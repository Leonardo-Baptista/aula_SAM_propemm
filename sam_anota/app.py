"""Interface do sam-anota em CustomTkinter.

Toda chamada ao modelo (carga, codificação, predição, modo texto) roda numa única
thread de trabalho, em fila. A thread devolve resultados por uma fila que a interface
lê a cada 30 ms; nenhum widget é tocado fora da thread principal. Downloads de pesos
e a exportação rodam em threads próprias e também respondem pela fila.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from . import __version__
from .dialogos import DialogoExportar
from .exportar import OpcoesTreino, exportar_treino
from .metricas import (LIMIAR_DIFERENCA, RelatorioSessao, caminho_referencia, carregar_referencia,
                       diferenca_grande, iou_contra_referencia, iou_uniao)
from .modelo import (CATALOGO, DISPOSITIVOS_ROTULO, MODELO_PADRAO, ErroModelo, ModeloSAM, Predicao,
                     baixar_checkpoint, carregar_imagem_rgb, descrever_dispositivo, dispositivos_disponiveis,
                     modelos_disponiveis, resolver_chave, sha1_arquivo)
from .projeto import RAIZ, Projeto, cor_da_classe
from .texto import (LIMIAR_CAIXA, LIMIAR_TEXTO, MOTORES, DetectorGdino, ErroTexto, SegmentadorSam3,
                    classe_para_texto, requisitos_gdino, requisitos_sam3, texto_com_gdino)

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

COR_ATUAL = (30, 144, 255)  # máscara em edição: preenchimento
COR_CONTORNO_ATUAL = (255, 255, 0)
LIMIAR_ARRASTO = 6  # px na tela: abaixo disso é clique, acima é caixa
FUNDO_CANVAS = {"dark": "#1b1b1b", "light": "#dcdcdc"}
COR_ALERTA = ("#b3261e", "#ff6b5e")
COR_OK = ("#1b7a33", "#5fd47a")
COR_TEXTO = ("gray10", "gray90")
FERRAMENTAS = {"Ponto +": "pos", "Ponto -": "neg", "Caixa": "caixa"}

ATALHOS = (
    "Ferramenta: P ponto +, N ponto -, C caixa\n"
    "Clique esquerdo: ponto da ferramenta\n"
    "Clique direito: ponto negativo (sempre)\n"
    "Arrastar: caixa (em qualquer ferramenta)\n"
    "Caixa e pontos se somam no mesmo objeto\n"
    "1, 2, 3: escolher candidata\n"
    "Enter: aceitar   Del: descartar instância\n"
    "Esc: limpar prompt   Ctrl+Z: desfazer\n"
    "PgUp / PgDn: imagem anterior / próxima\n"
    "Ctrl+Enter: concluir e ir à próxima\n"
    "Ctrl+E: exportar   F5: recarregar pasta"
)

TEXTO_SOBRE = f"""sam-anota {__version__}
Anotação de imagens com o Segment Anything Model, para aula.

Código do sam-anota: licença MIT.

Créditos e licenças de terceiros
* SAM 2 / SAM 2.1 (código e pesos): Meta AI (FAIR), licença Apache 2.0.
  https://github.com/facebookresearch/sam2
* SAM 1 (segment-anything, código e pesos): Meta AI, licença Apache 2.0.
  https://github.com/facebookresearch/segment-anything
* SAM 3 (opcional, modo texto): Meta AI, licença própria (SAM License),
  pesos com acesso controlado no Hugging Face (facebook/sam3).
* Grounding DINO tiny (opcional, modo texto): IDEA-Research, Apache 2.0.
* Hugging Face transformers: Apache 2.0.
* CustomTkinter: Tom Schimansky, licença MIT.
* PyTorch (BSD 3-Clause), NumPy (BSD 3-Clause), OpenCV (Apache 2.0),
  Pillow (MIT-CMU), Hydra (MIT), pycocotools (BSD 2-Clause).
* Imagens: licenças em dados/exemplos/FONTES.md e dados/dificeis/FONTES.md.

Referências
* Ravi, N. et al. SAM 2: Segment Anything in Images and Videos.
  arXiv:2408.00714, 2024 (ICLR 2025).
* Kirillov, A. et al. Segment Anything. ICCV 2023. arXiv:2304.02643.
* Carion, N. et al. SAM 3: Segment Anything with Concepts. Meta, 2025.
* Liu, S. et al. Grounding DINO: Marrying DINO with Grounded Pre-Training
  for Open-Set Object Detection. ECCV 2024. arXiv:2303.05499.

O "escore previsto" é a IoU que o próprio SAM estima para cada máscara
candidata; a "IoU real" só aparece quando existe <nome>_ref.png.
"""


def _fmt(x: float, casas: int = 2) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def _sobrepor(base: np.ndarray, mascara: np.ndarray, cor, alfa: float, cor_contorno=None, espessura=2):
    """Pinta a máscara sobre base (in place) com transparência e contorno."""
    if not mascara.any():
        return
    sel = mascara.astype(bool)
    base[sel] = (base[sel] * (1 - alfa) + np.array(cor, dtype=np.float32) * alfa).astype(np.uint8)
    contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base, contornos, -1, cor_contorno or cor, espessura, lineType=cv2.LINE_AA)


class App(ctk.CTk):
    def __init__(self, pasta: Path | None = None, modelo: str = MODELO_PADRAO, dispositivo: str | None = None,
                 cache_disco: bool = True, checkpoint: Path | None = None):
        super().__init__()
        self.title("sam-anota: anotação com SAM")
        self.geometry("1366x740")
        self.minsize(1100, 620)

        self.cache_disco = cache_disco
        self.dispositivo_pedido = dispositivo or "auto"
        self.modelo = ModeloSAM(modelo, checkpoint=checkpoint, dispositivo=self.dispositivo_pedido,
                                cache_disco=cache_disco)
        self.modelo_pronto = False
        self.erro_modelo: str | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam")
        self.fila: queue.Queue = queue.Queue()
        self.baixando: str | None = None
        self.detector: DetectorGdino | None = None
        self.sam3: SegmentadorSam3 | None = None
        self.ultima_exportacao = None

        self.projeto: Projeto | None = None
        self.relatorio: RelatorioSessao | None = None
        self.lista: list[str] = []
        self.botoes_lista: dict[str, ctk.CTkButton] = {}
        self.miniaturas: dict[str, ctk.CTkImage] = {}
        self.ferramenta = "pos"

        self._zerar_estado_imagem()
        self._montar_interface()
        self._ligar_atalhos()
        self.protocol("WM_DELETE_WINDOW", self.fechar)
        self.after(30, self._ler_fila)
        self.after(200, self._tique)

        if checkpoint is None and not self.modelo.info.baixado:
            # peso ainda não baixado: oferece o download (com barra e SHA) em vez de falhar
            self.rotulo_modelo.configure(text=f"{self.modelo.info.rotulo}: peso não baixado")
            self.after(300, lambda: self.trocar_modelo(self.modelo.nome))
        else:
            self._iniciar_carga(self.modelo)
        self.abrir_pasta(Path(pasta) if pasta else RAIZ / "dados")

    # ================================================================ estado

    def _zerar_estado_imagem(self):
        self.rel: str | None = None
        self.imagem_rgb: np.ndarray | None = None
        self.chave: str | None = None
        self.codificada = False
        self.inicio_codificacao: float | None = None
        self.tempo_codificacao: float | None = None
        self.origem_codificacao = ""
        self.referencia: np.ndarray | None = None
        self.instancias_texto: list = []
        self.indice_texto = 0
        self.texto_atual = ""
        self._zerar_prompt()
        self.escala = 1.0
        self.desloc = (0, 0)
        self.base_exibicao: np.ndarray | None = None
        self.camada_objetos: np.ndarray | None = None
        self.foto = None
        self.rotulos_objetos: list = []
        self.tempo_ultimo_clique: float | None = None
        self.ultimo_objeto_txt = ""
        self.iou_uniao = None

    def _zerar_prompt(self):
        self.pontos: list[tuple[float, float, int]] = []
        self.caixa: tuple[float, float, float, float] | None = None
        self.historico: list[tuple] = []
        self.predicao = None
        self.escolhida = 0
        self.id_pedido = 0
        self.cliques_objeto = 0
        self.inicio_objeto: float | None = None
        self.arrasto: tuple[int, int] | None = None
        self.origem_base = ""  # vazio: cliques no SAM; senão a origem da instância do modo texto
        self.ajustado = False

    def origem_atual(self) -> str:
        sam = self.modelo.info.rotulo
        if not self.origem_base:
            return f"{sam} (cliques)"
        base = f"{self.origem_base} (texto: {self.texto_atual})"
        return base + (f" + ajuste com cliques no {sam}" if self.ajustado else "")

    # ================================================================ interface

    def _montar_interface(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        negrito = ctk.CTkFont(weight="bold")
        pequena = ctk.CTkFont(size=12)

        # ---- barra superior
        topo = ctk.CTkFrame(self, corner_radius=0)
        topo.grid(row=0, column=0, columnspan=3, sticky="ew")
        ctk.CTkButton(topo, text="Abrir pasta...", width=100, command=self.escolher_pasta).pack(side="left", padx=(8, 4), pady=6)
        ctk.CTkButton(topo, text="Exportar...", width=90, command=self.exportar).pack(side="left", padx=4)
        ctk.CTkButton(topo, text="Sobre", width=60, command=self.mostrar_sobre).pack(side="left", padx=4)
        ctk.CTkLabel(topo, text="Modelo").pack(side="left", padx=(12, 4))
        self.mapa_modelos = {m.rotulo_combo: m.chave for m in modelos_disponiveis()}
        self.combo_modelo = ctk.CTkComboBox(topo, values=list(self.mapa_modelos), width=255, state="readonly",
                                            command=self._ao_escolher_modelo)
        self.combo_modelo.set(self.modelo.info.rotulo_combo)
        self.combo_modelo.pack(side="left")
        ctk.CTkLabel(topo, text="Dispositivo").pack(side="left", padx=(10, 4))
        self.mapa_disp = {DISPOSITIVOS_ROTULO[d]: d for d in dispositivos_disponiveis()}
        self.combo_disp = ctk.CTkComboBox(topo, values=list(self.mapa_disp), width=125, state="readonly",
                                          command=self._ao_escolher_dispositivo)
        self.combo_disp.set(DISPOSITIVOS_ROTULO.get(self.dispositivo_pedido, "Automático"))
        self.combo_disp.pack(side="left")
        self.tema = ctk.CTkSegmentedButton(topo, values=["Escuro", "Claro"], command=self._mudar_tema, width=100)
        self.tema.set("Escuro" if ctk.get_appearance_mode() == "Dark" else "Claro")
        self.tema.pack(side="left", padx=10)
        self.rotulo_modelo = ctk.CTkLabel(topo, text="", anchor="e", font=pequena)
        self.rotulo_modelo.pack(side="right", padx=10)

        # ---- lista de imagens (esquerda)
        esq = ctk.CTkFrame(self, width=205, corner_radius=0)
        esq.grid(row=1, column=0, sticky="ns")
        esq.grid_propagate(False)
        esq.grid_rowconfigure(1, weight=1)
        esq.grid_columnconfigure(0, weight=1)
        self.rotulo_pasta = ctk.CTkLabel(esq, text="Imagens", font=negrito, anchor="w")
        self.rotulo_pasta.grid(row=0, column=0, sticky="w", padx=10, pady=(6, 0))
        self.quadro_lista = ctk.CTkScrollableFrame(esq, width=180)
        self.quadro_lista.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # ---- canvas (centro)
        self.canvas = tk.Canvas(self, highlightthickness=0, cursor="crosshair",
                                bg=FUNDO_CANVAS["dark" if ctk.get_appearance_mode() == "Dark" else "light"])
        self.canvas.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self.canvas.bind("<Configure>", lambda e: self._ajustar_exibicao())
        self.canvas.bind("<ButtonPress-1>", self._ao_pressionar)
        self.canvas.bind("<B1-Motion>", self._ao_arrastar)
        self.canvas.bind("<ButtonRelease-1>", self._ao_soltar)
        self.canvas.bind("<ButtonPress-3>", self._ao_clique_direito)
        if sys.platform == "darwin":
            self.canvas.bind("<ButtonPress-2>", self._ao_clique_direito)

        # ---- painel direito
        dir_ = ctk.CTkScrollableFrame(self, width=290, corner_radius=0)
        dir_.grid(row=1, column=2, sticky="ns")

        self.seletor = ctk.CTkSegmentedButton(dir_, values=list(FERRAMENTAS), command=self._ao_escolher_ferramenta)
        self.seletor.set("Ponto +")
        self.seletor.pack(fill="x", padx=6, pady=(4, 4))

        ctk.CTkLabel(dir_, text="Máscaras candidatas (1, 2, 3)", font=negrito).pack(anchor="w", padx=6)
        linha = ctk.CTkFrame(dir_, fg_color="transparent")
        linha.pack(fill="x", padx=2)
        self.botoes_cand = []
        self._img_vazia = ctk.CTkImage(Image.new("RGB", (84, 56), (90, 90, 90)), size=(84, 56))
        for i in range(3):
            b = ctk.CTkButton(linha, text=f"{i + 1}: -", image=self._img_vazia, compound="top", width=90,
                              height=88, fg_color="transparent", border_width=2,
                              border_color=("gray70", "gray30"), text_color=COR_TEXTO,
                              command=lambda i=i: self.escolher(i))
            b.pack(side="left", padx=2)
            self.botoes_cand.append(b)
        self.rotulo_origem = ctk.CTkLabel(dir_, text="", font=pequena, anchor="w", justify="left", wraplength=280)
        self.rotulo_origem.pack(fill="x", padx=6)

        linha_c = ctk.CTkFrame(dir_, fg_color="transparent")
        linha_c.pack(fill="x", padx=6, pady=(4, 0))
        ctk.CTkLabel(linha_c, text="Classe", font=negrito).pack(side="left")
        self.combo_classe = ctk.CTkComboBox(linha_c, values=["objeto"], state="readonly", width=215,
                                            command=lambda _v: self.canvas.focus_set())
        self.combo_classe.pack(side="right")
        linha2 = ctk.CTkFrame(dir_, fg_color="transparent")
        linha2.pack(fill="x", padx=6, pady=2)
        self.entrada_classe = ctk.CTkEntry(linha2, placeholder_text="nova classe", width=150)
        self.entrada_classe.pack(side="left")
        self.entrada_classe.bind("<Return>", lambda e: self.adicionar_classe())
        ctk.CTkButton(linha2, text="+", width=32, command=self.adicionar_classe).pack(side="left", padx=4)
        ctk.CTkButton(linha2, text="Remover", width=76, fg_color="gray40",
                      command=self.remover_classe).pack(side="left")

        linha3 = ctk.CTkFrame(dir_, fg_color="transparent")
        linha3.pack(fill="x", padx=6, pady=(6, 2))
        self.botao_aceitar = ctk.CTkButton(linha3, text="Aceitar (Enter)", height=32, width=150, command=self.aceitar)
        self.botao_aceitar.pack(side="left")
        ctk.CTkButton(linha3, text="Limpar (Esc)", fg_color="gray40", height=32, width=110,
                      command=self.limpar).pack(side="right")

        self.rotulo_avaliacao = ctk.CTkLabel(dir_, text="", wraplength=280, justify="left", anchor="w", font=pequena)
        self.rotulo_avaliacao.pack(fill="x", padx=6, pady=(2, 2))

        abas = ctk.CTkTabview(dir_, height=300, width=285)
        abas.pack(fill="x", padx=2, pady=(0, 4))
        self.abas = abas
        a_obj, a_txt, a_aj = abas.add("Objetos"), abas.add("Texto"), abas.add("Atalhos")

        self.quadro_objetos = ctk.CTkFrame(a_obj)
        self.quadro_objetos.pack(fill="x", pady=2)
        self.chave_concluida = ctk.CTkSwitch(a_obj, text="Imagem concluída", command=self._alternar_concluida)
        self.chave_concluida.pack(anchor="w", pady=6)

        # aba Texto
        self.entrada_texto = ctk.CTkEntry(a_txt, placeholder_text="em inglês: smoke, car, person")
        self.entrada_texto.pack(fill="x", pady=(2, 2))
        self.entrada_texto.bind("<Return>", lambda e: self.buscar_texto())
        self.combo_motor = ctk.CTkComboBox(a_txt, values=list(MOTORES.values()), state="readonly")
        self.combo_motor.set(MOTORES["gdino"])
        self.combo_motor.pack(fill="x", pady=2)
        linha_l = ctk.CTkFrame(a_txt, fg_color="transparent")
        linha_l.pack(fill="x", pady=2)
        ctk.CTkLabel(linha_l, text="caixa", font=pequena).pack(side="left")
        self.limiar_caixa = ctk.CTkEntry(linha_l, width=48)
        self.limiar_caixa.insert(0, _fmt(LIMIAR_CAIXA))
        self.limiar_caixa.pack(side="left", padx=(4, 10))
        ctk.CTkLabel(linha_l, text="texto", font=pequena).pack(side="left")
        self.limiar_texto = ctk.CTkEntry(linha_l, width=48)
        self.limiar_texto.insert(0, _fmt(LIMIAR_TEXTO))
        self.limiar_texto.pack(side="left", padx=4)
        self.botao_buscar = ctk.CTkButton(linha_l, text="Buscar", width=70, command=self.buscar_texto)
        self.botao_buscar.pack(side="right")
        self.mapear_classe = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(a_txt, text="Classe com o nome do texto", variable=self.mapear_classe,
                        font=pequena).pack(anchor="w", pady=2)
        self.rotulo_texto = ctk.CTkLabel(a_txt, text="Cada instância achada vira um objeto para revisar.",
                                         wraplength=260, justify="left", anchor="w", font=pequena)
        self.rotulo_texto.pack(fill="x", pady=2)
        linha_i = ctk.CTkFrame(a_txt, fg_color="transparent")
        linha_i.pack(fill="x")
        ctk.CTkButton(linha_i, text="Descartar (Del)", width=120, fg_color="gray40",
                      command=self.descartar_instancia).pack(side="left")
        ctk.CTkButton(linha_i, text="Próxima", width=80, fg_color="gray40",
                      command=lambda: self._ir_instancia(self.indice_texto + 1)).pack(side="right")

        ctk.CTkLabel(a_aj, text=ATALHOS, justify="left", anchor="w", font=pequena).pack(anchor="w")

        # ---- barra de status
        base = ctk.CTkFrame(self, corner_radius=0, height=28)
        base.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.status = {}
        for nome in ("codificacao", "clique", "cliques", "tempo_obj", "objetos", "mensagem"):
            r = ctk.CTkLabel(base, text="", anchor="w")
            r.pack(side="left", padx=(10, 6))
            self.status[nome] = r
        self.status["mensagem"].pack_configure(fill="x", expand=True)
        self.barra_download = ctk.CTkProgressBar(base, width=160)

    def _ligar_atalhos(self):
        b = self.bind
        b("<Escape>", lambda e: self._se_livre(e, self.limpar))
        b("<Control-z>", lambda e: self._se_livre(e, self.desfazer))
        b("<Control-Z>", lambda e: self._se_livre(e, self.desfazer))
        for i in range(3):
            b(f"<Key-{i + 1}>", lambda e, i=i: self._se_livre(e, lambda: self.escolher(i)))
            b(f"<KP_{i + 1}>", lambda e, i=i: self._se_livre(e, lambda: self.escolher(i)))
        for tecla, nome in (("p", "Ponto +"), ("n", "Ponto -"), ("c", "Caixa")):
            b(f"<Key-{tecla}>", lambda e, n=nome: self._se_livre(e, lambda: self.definir_ferramenta(n)))
        b("<Return>", lambda e: self._se_livre(e, self.aceitar))
        b("<KP_Enter>", lambda e: self._se_livre(e, self.aceitar))
        b("<Delete>", lambda e: self._se_livre(e, self.descartar_instancia))
        b("<Control-Return>", lambda e: self._se_livre(e, self.concluir_e_avancar))
        b("<Next>", lambda e: self._se_livre(e, lambda: self.navegar(1)))
        b("<Prior>", lambda e: self._se_livre(e, lambda: self.navegar(-1)))
        b("<Control-e>", lambda e: self.exportar())
        b("<Control-s>", lambda e: self.salvar())
        b("<F5>", lambda e: self.abrir_pasta(self.projeto.pasta_imagens) if self.projeto else None)

    def _se_livre(self, evento, acao):
        """Ignora atalhos quando o foco está num campo de texto."""
        if isinstance(evento.widget, (tk.Entry, tk.Text)):
            return None
        acao()
        return "break"

    # ================================================================ thread de trabalho

    def _enviar(self, funcao, *args):
        def envelope():
            try:
                funcao(*args)
            except Exception as e:  # erro na thread vai para a interface
                self.fila.put((self._erro_tarefa, (e, traceback.format_exc())))
        self.executor.submit(envelope)

    def _devolver(self, funcao, *args):
        self.fila.put((funcao, args))

    def _ler_fila(self):
        try:
            while True:
                funcao, args = self.fila.get_nowait()
                funcao(*args)
        except queue.Empty:
            pass
        self.after(30, self._ler_fila)

    def _erro_tarefa(self, e, rastro):
        print(rastro, file=sys.stderr)
        self._mensagem(f"Erro: {e}", alerta=True)

    # ================================================================ modelo e dispositivo

    def _iniciar_carga(self, modelo: ModeloSAM):
        self.modelo_pronto = False
        self.erro_modelo = None
        self.rotulo_modelo.configure(text=f"Carregando {modelo.info.rotulo}...", text_color=COR_TEXTO)
        self._enviar(self._tarefa_carregar, modelo)

    def _tarefa_carregar(self, modelo: ModeloSAM):
        try:
            t = modelo.carregar()
        except ErroModelo as e:
            self._devolver(self._falha_modelo, modelo, str(e))
            return
        self._devolver(self._modelo_carregado, modelo, t)

    def _modelo_carregado(self, modelo, t):
        if modelo is not self.modelo:
            return
        self.modelo_pronto = True
        self.rotulo_modelo.configure(text=descrever_dispositivo(modelo.dispositivo), text_color=COR_TEXTO)
        self._mensagem(f"{modelo.info.rotulo} carregado em {_fmt(t, 1)} s.")
        if self.relatorio:
            self.relatorio.dispositivo = modelo.dispositivo
            self.relatorio.modelo = modelo.info.rotulo
        if modelo.aviso_dispositivo:
            self.combo_disp.set(DISPOSITIVOS_ROTULO["cpu"])
            self.dispositivo_pedido = "cpu"
            messagebox.showwarning("Dispositivo", modelo.aviso_dispositivo)

    def _falha_modelo(self, modelo, msg):
        if modelo is not self.modelo:
            return
        self.erro_modelo = msg
        self.rotulo_modelo.configure(text="SAM indisponível", text_color=COR_ALERTA)
        self.inicio_codificacao = None
        self._redesenhar()
        messagebox.showerror("SAM indisponível", msg + "\n\nA interface abre, mas não segmenta.")

    def _ao_escolher_modelo(self, rotulo: str):
        self.canvas.focus_set()
        chave = self.mapa_modelos[rotulo]
        if chave == self.modelo.nome and (self.modelo_pronto or self.erro_modelo is None):
            return
        self.trocar_modelo(chave)

    def _ao_escolher_dispositivo(self, rotulo: str):
        self.canvas.focus_set()
        self.dispositivo_pedido = self.mapa_disp[rotulo]
        self.trocar_modelo(self.modelo.nome, forcar=True)

    def trocar_modelo(self, chave: str, forcar: bool = False, perguntar: bool = True):
        """Troca o modelo (ou recarrega no novo dispositivo). Limpa o cache de embeddings."""
        chave = resolver_chave(chave)
        info = CATALOGO[chave]
        if self.baixando:
            messagebox.showinfo("Modelo", f"Aguarde o download de {self.baixando} terminar.")
            self.combo_modelo.set(self.modelo.info.rotulo_combo)
            return
        if not info.baixado:
            if perguntar and not messagebox.askyesno(
                    "Baixar pesos", f"{info.rotulo} ainda não foi baixado.\nBaixar agora "
                    f"{info.tamanho_bytes / 1e6:.0f} MB de {info.url}?\nO SHA-256 é conferido no fim."):
                self.combo_modelo.set(self.modelo.info.rotulo_combo)
                return
            self._baixar(info)
            return
        antigo = self.modelo
        self.modelo = ModeloSAM(chave, dispositivo=self.dispositivo_pedido, cache_disco=self.cache_disco)
        self.combo_modelo.set(info.rotulo_combo)
        self._enviar(antigo.liberar)
        self.detector = None  # recriado no dispositivo novo quando precisar
        self.sam3 = None
        self._iniciar_carga(self.modelo)
        if self.rel:
            self.abrir_imagem(self.rel)

    def _baixar(self, info):
        self.baixando = info.rotulo
        self.barra_download.set(0)
        self.barra_download.pack(side="right", padx=10)
        self._mensagem(f"Baixando {info.rotulo}...")
        ultimo = [-1.0]

        def progresso(feito, total):
            frac = feito / max(total, 1)
            if frac - ultimo[0] >= 0.01 or feito == total:
                ultimo[0] = frac
                self.fila.put((self._progresso_download, (info, feito, total)))

        def tarefa():
            try:
                baixar_checkpoint(info, progresso)
                self.fila.put((self._download_concluido, (info, None)))
            except Exception as e:
                self.fila.put((self._download_concluido, (info, e)))
        threading.Thread(target=tarefa, daemon=True, name="download").start()

    def _progresso_download(self, info, feito, total):
        self.barra_download.set(feito / max(total, 1))
        self._mensagem(f"Baixando {info.rotulo}: {feito / 1e6:.0f} de {total / 1e6:.0f} MB "
                       f"({100 * feito / max(total, 1):.0f}%)")

    def _download_concluido(self, info, erro):
        self.baixando = None
        self.barra_download.pack_forget()
        if erro is not None:
            self.combo_modelo.set(self.modelo.info.rotulo_combo)
            self._mensagem(f"Download de {info.rotulo} falhou: {erro}", alerta=True)
            messagebox.showerror("Download", f"Download de {info.rotulo} falhou:\n{erro}")
            return
        self._mensagem(f"{info.rotulo} baixado, SHA-256 conferido.")
        self.trocar_modelo(info.chave, perguntar=False)

    # ================================================================ pasta e lista

    def escolher_pasta(self):
        inicial = self.projeto.pasta_imagens if self.projeto else RAIZ / "dados"
        pasta = filedialog.askdirectory(initialdir=str(inicial), title="Pasta de imagens")
        if pasta:
            self.abrir_pasta(Path(pasta))

    def abrir_pasta(self, pasta: Path):
        pasta = Path(pasta)
        if not pasta.is_dir():
            self._mensagem(f"Pasta não encontrada: {pasta}", alerta=True)
            return
        if self.projeto:
            self.salvar()
        self.projeto = Projeto.abrir_ou_criar(pasta)
        self.relatorio = RelatorioSessao(self.projeto.pasta_saida / "relatorios",
                                         self.modelo.dispositivo if self.modelo_pronto else "",
                                         self.modelo.info.rotulo)
        self.lista = self.projeto.sincronizar()
        self.rotulo_pasta.configure(text=f"Imagens: {pasta.name}")
        self._atualizar_combo()
        self._montar_lista()
        self._zerar_estado_imagem()
        self.canvas.delete("all")
        self._atualizar_objetos()
        if self.lista:
            primeira = next((r for r in self.lista if not self.projeto.imagens[r].concluida), self.lista[0])
            self.abrir_imagem(primeira)
        else:
            self._mensagem("Nenhuma imagem nesta pasta.", alerta=True)

    def _montar_lista(self):
        for w in self.quadro_lista.winfo_children():
            w.destroy()
        self.botoes_lista.clear()
        grupo_atual = None
        for rel in self.lista:
            grupo = rel.rsplit("/", 1)[0] if "/" in rel else ""
            if grupo != grupo_atual:
                grupo_atual = grupo
                ctk.CTkLabel(self.quadro_lista, text=(grupo or ".") + "/", anchor="w",
                             font=ctk.CTkFont(size=12, weight="bold")).pack(fill="x", padx=2, pady=(6, 0))
            b = ctk.CTkButton(self.quadro_lista, text=self._texto_item(rel), anchor="w", height=44,
                              fg_color="transparent", text_color=COR_TEXTO,
                              hover_color=("gray75", "gray28"), command=lambda r=rel: self.abrir_imagem(r))
            b.pack(fill="x", padx=2, pady=1)
            self.botoes_lista[rel] = b
        self._carregar_miniaturas(list(self.lista))

    def _carregar_miniaturas(self, pendentes: list[str]):
        """Gera miniaturas aos poucos, para a janela não congelar com pastas grandes."""
        if not pendentes or self.projeto is None:
            return
        rel = pendentes.pop(0)
        b = self.botoes_lista.get(rel)
        if b is not None:
            try:
                if rel not in self.miniaturas:
                    with Image.open(self.projeto.caminho_absoluto(rel)) as im:
                        im.draft("RGB", (112, 84))
                        im = im.convert("RGB")
                        im.thumbnail((56, 42))
                        self.miniaturas[rel] = ctk.CTkImage(im, size=im.size)
                b.configure(image=self.miniaturas[rel], compound="left")
            except Exception:
                pass
        self.after(1, self._carregar_miniaturas, pendentes)

    def _texto_item(self, rel: str) -> str:
        img = self.projeto.imagens[rel]
        nome = rel.rsplit("/", 1)[-1]
        if len(nome) > 17:
            nome = nome[:15] + "..."
        marca = "✔ " if img.concluida else ""
        n = len(img.objetos)
        return f"{marca}{nome}\n{n} objeto{'s' if n != 1 else ''}"

    def _atualizar_item(self, rel: str):
        b = self.botoes_lista.get(rel)
        if b is None:
            return
        atual = rel == self.rel
        b.configure(text=self._texto_item(rel),
                    fg_color=("gray78", "gray25") if atual else "transparent",
                    text_color=COR_OK if self.projeto.imagens[rel].concluida else COR_TEXTO)

    def navegar(self, passo: int):
        if not self.lista:
            return
        i = self.lista.index(self.rel) if self.rel in self.lista else -1
        self.abrir_imagem(self.lista[(i + passo) % len(self.lista)])

    # ================================================================ imagem

    def abrir_imagem(self, rel: str):
        anterior = self.rel
        self._zerar_estado_imagem()
        self.rel = rel
        if anterior in self.botoes_lista:
            self._atualizar_item(anterior)
        self._atualizar_item(rel)
        caminho = self.projeto.caminho_absoluto(rel)
        try:
            self.imagem_rgb = carregar_imagem_rgb(caminho)
            self.chave = sha1_arquivo(caminho)
        except Exception as e:
            self._mensagem(f"Não consegui abrir {rel}: {e}", alerta=True)
            return
        img = self.projeto.imagens[rel]
        img.altura, img.largura = self.imagem_rgb.shape[:2]
        ref = caminho_referencia(caminho)
        if ref is not None:
            try:
                self.referencia = carregar_referencia(ref, self.imagem_rgb.shape[:2])
            except Exception as e:
                self._mensagem(f"Referência ilegível ({ref.name}): {e}", alerta=True)
        self.chave_concluida.select() if img.concluida else self.chave_concluida.deselect()
        self.rotulo_avaliacao.configure(
            text=("Esta imagem tem máscara de referência: a IoU real aparece ao aceitar."
                  if self.referencia is not None else "Sem máscara de referência (<nome>_ref.png)."),
            text_color=COR_TEXTO)
        self.rotulo_texto.configure(text="Cada instância achada vira um objeto para revisar.")
        self._atualizar_objetos()
        self._ajustar_exibicao()
        self._atualizar_candidatas()
        if self.erro_modelo:
            return
        self.inicio_codificacao = time.perf_counter()
        self._mensagem(f"{rel}: codificando...")
        self._enviar(self._tarefa_codificar, self.modelo, rel, self.chave, self.imagem_rgb)

    def _tarefa_codificar(self, modelo, rel, chave, imagem):
        if rel != self.rel or modelo is not self.modelo or not modelo.carregado:
            return  # o aluno já trocou de imagem ou de modelo: não gasta tempo
        e = modelo.codificar(chave, imagem)
        self._devolver(self._codificada, modelo, rel, e)

    def _codificada(self, modelo, rel, e):
        if rel != self.rel or modelo is not self.modelo:
            return
        self.codificada = True
        self.inicio_codificacao = None
        img = self.projeto.imagens[rel]
        if e.origem == "calculado":
            img.tempo_codificacao_s = e.tempo_s
            if self.relatorio:
                self.relatorio.codificacoes.append(e.tempo_s)
        self.tempo_codificacao = e.tempo_s
        self.origem_codificacao = e.origem
        self._mensagem(f"{rel}: pronta ({img.largura}x{img.altura}, {modelo.info.rotulo}). Clique no objeto.")
        if self.pontos or self.caixa:
            self._pedir_predicao()
        self._redesenhar()

    # ================================================================ exibição

    def _ajustar_exibicao(self):
        if self.imagem_rgb is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        h, w = self.imagem_rgb.shape[:2]
        self.escala = min(cw / w, ch / h)
        dw, dh = max(1, int(w * self.escala)), max(1, int(h * self.escala))
        self.desloc = ((cw - dw) // 2, (ch - dh) // 2)
        interp = cv2.INTER_AREA if self.escala < 1 else cv2.INTER_LINEAR
        self.base_exibicao = cv2.resize(self.imagem_rgb, (dw, dh), interpolation=interp)
        self._refazer_camada_objetos()

    def _reduzir(self, mascara: np.ndarray) -> np.ndarray:
        dh, dw = self.base_exibicao.shape[:2]
        return cv2.resize(mascara.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)

    def _cor_classe(self, classe: str):
        return cor_da_classe(self.projeto.classes.index(classe) if classe in self.projeto.classes else 0)

    def _refazer_camada_objetos(self):
        """Imagem reduzida já com os objetos aceitos pintados (refeita só quando eles mudam)."""
        if self.base_exibicao is None or self.rel is None:
            return
        camada = self.base_exibicao.copy()
        self.rotulos_objetos = []
        for obj in self.projeto.imagens[self.rel].objetos:
            m = self._reduzir(obj.mascara())
            _sobrepor(camada, m, self._cor_classe(obj.classe), 0.40)
            ys, xs = np.nonzero(m)
            if len(xs):
                self.rotulos_objetos.append((int(xs.min()), int(ys.min()), f"#{obj.id} {obj.classe}",
                                             self._cor_classe(obj.classe)))
        self.camada_objetos = camada
        self._redesenhar()

    def _redesenhar(self):
        c = self.canvas
        c.delete("all")
        if self.camada_objetos is None:
            return
        quadro = self.camada_objetos.copy()
        m = self._mascara_escolhida()
        if m is not None:
            _sobrepor(quadro, self._reduzir(m), COR_ATUAL, 0.45, COR_CONTORNO_ATUAL, 2)
        self.foto = ImageTk.PhotoImage(Image.fromarray(quadro))
        ox, oy = self.desloc
        c.create_image(ox, oy, image=self.foto, anchor="nw")
        for x, y, texto, cor in self.rotulos_objetos:
            t = c.create_text(ox + x + 3, oy + y + 2, text=texto, anchor="nw", fill="white",
                              font=("Segoe UI", 9, "bold"))
            caixa = c.bbox(t)
            r = c.create_rectangle(caixa[0] - 2, caixa[1] - 1, caixa[2] + 2, caixa[3] + 1,
                                   fill="#%02x%02x%02x" % tuple(int(v * 0.6) for v in cor), outline="")
            c.tag_lower(r, t)
        # instâncias do modo texto ainda por revisar: caixas tracejadas finas
        for i, inst in enumerate(self.instancias_texto):
            if i == self.indice_texto:
                continue
            x0, y0 = self._para_tela(inst.caixa[0], inst.caixa[1])
            x1, y1 = self._para_tela(inst.caixa[2], inst.caixa[3])
            c.create_rectangle(x0, y0, x1, y1, outline="#c0c0c0", width=1, dash=(3, 3))
        if self.caixa is not None:
            x0, y0 = self._para_tela(self.caixa[0], self.caixa[1])
            x1, y1 = self._para_tela(self.caixa[2], self.caixa[3])
            c.create_rectangle(x0, y0, x1, y1, outline="#ffff00", width=2, dash=(6, 3))
        raio = 6 * self._escala_dpi()
        for x, y, r in self.pontos:
            sx, sy = self._para_tela(x, y)
            cor = "#20d040" if r == 1 else "#ff3030"
            c.create_oval(sx - raio, sy - raio, sx + raio, sy + raio, fill=cor, outline="white", width=2)
            c.create_text(sx, sy, text="+" if r == 1 else "-", fill="white", font=("Segoe UI", 9, "bold"))
        if self.inicio_codificacao is not None:
            self._desenhar_aviso_codificando()

    def _desenhar_aviso_codificando(self):
        c = self.canvas
        c.delete("aviso")
        if self.inicio_codificacao is None:
            return
        dt = time.perf_counter() - self.inicio_codificacao
        txt = (f"codificando com {self.modelo.info.rotulo}... {_fmt(dt, 1)} s" if self.modelo_pronto
               else f"carregando {self.modelo.info.rotulo}...")
        cx = c.winfo_width() // 2
        t = c.create_text(cx, 24, text=txt, fill="white", font=("Segoe UI", 13, "bold"), tags="aviso")
        b = c.bbox(t)
        r = c.create_rectangle(b[0] - 10, b[1] - 4, b[2] + 10, b[3] + 4, fill="#000000",
                               stipple="gray50", outline="", tags="aviso")
        c.tag_lower(r, t)

    def _escala_dpi(self) -> float:
        """Fator de escala do Windows (1,0 em 100%, 1,5 em 150%): o canvas usa pixels físicos."""
        try:
            return float(ctk.ScalingTracker.get_window_scaling(self))
        except Exception:
            return 1.0

    def _para_tela(self, x, y):
        return self.desloc[0] + x * self.escala, self.desloc[1] + y * self.escala

    def _para_imagem(self, sx, sy):
        h, w = self.imagem_rgb.shape[:2]
        x = (sx - self.desloc[0]) / self.escala
        y = (sy - self.desloc[1]) / self.escala
        dentro = 0 <= x < w and 0 <= y < h
        return float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1)), dentro

    # ================================================================ prompts

    def _ao_escolher_ferramenta(self, nome: str):
        self.ferramenta = FERRAMENTAS[nome]
        self.canvas.configure(cursor="tcross" if self.ferramenta == "caixa" else "crosshair")
        self.canvas.focus_set()

    def definir_ferramenta(self, nome: str):
        self.seletor.set(nome)
        self._ao_escolher_ferramenta(nome)

    def _ao_pressionar(self, e):
        self.canvas.focus_set()
        if self.imagem_rgb is not None:
            self.arrasto = (e.x, e.y)

    def _ao_arrastar(self, e):
        if self.arrasto is None:
            return
        self.canvas.delete("elastico")
        x0, y0 = self.arrasto
        if abs(e.x - x0) > LIMIAR_ARRASTO or abs(e.y - y0) > LIMIAR_ARRASTO:
            self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="#ffff00", width=2, tags="elastico")

    def _ao_soltar(self, e):
        if self.arrasto is None:
            return
        x0, y0 = self.arrasto
        self.arrasto = None
        self.canvas.delete("elastico")
        if abs(e.x - x0) > LIMIAR_ARRASTO or abs(e.y - y0) > LIMIAR_ARRASTO:
            ax, ay, _ = self._para_imagem(x0, y0)
            bx, by, _ = self._para_imagem(e.x, e.y)
            self.definir_caixa(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
            return
        x, y, dentro = self._para_imagem(e.x, e.y)
        if not dentro:
            return
        if self.ferramenta == "caixa":
            self._mensagem("Ferramenta Caixa: arraste para desenhar (P volta para pontos).")
        else:
            self.adicionar_ponto(x, y, 1 if self.ferramenta == "pos" else 0)

    def _ao_clique_direito(self, e):
        self.canvas.focus_set()
        if self.imagem_rgb is None:
            return
        x, y, dentro = self._para_imagem(e.x, e.y)
        if dentro:
            self.adicionar_ponto(x, y, 0)

    def adicionar_ponto(self, x: float, y: float, rotulo: int):
        """Ponto em pixels da imagem; rotulo 1 = positivo, 0 = negativo."""
        self.pontos.append((x, y, rotulo))
        self.historico.append(("ponto",))
        self._novo_prompt()

    def definir_caixa(self, x0, y0, x1, y1):
        """Uma caixa por objeto; uma nova substitui a anterior. Os pontos continuam valendo."""
        self.historico.append(("caixa", self.caixa))
        self.caixa = (x0, y0, x1, y1)
        self._novo_prompt()

    def _novo_prompt(self):
        if self.inicio_objeto is None:
            self.inicio_objeto = time.perf_counter()
        self.cliques_objeto += 1
        if self.origem_base:
            self.ajustado = True
        self._pedir_predicao()
        self._redesenhar()

    def desfazer(self):
        if not self.historico:
            return
        acao = self.historico.pop()
        if acao[0] == "ponto":
            self.pontos.pop()
        else:
            self.caixa = acao[1]
        self._pedir_predicao()
        self._redesenhar()

    def limpar(self):
        """Esc: descarta o objeto em edição (zera cliques e cronômetro) e a fila do modo texto."""
        self.instancias_texto = []
        self.indice_texto = 0
        self._zerar_prompt()
        self._atualizar_candidatas()
        self._redesenhar()
        self._atualizar_status()

    def _pedir_predicao(self):
        self.id_pedido += 1
        if not self.pontos and self.caixa is None:
            self.predicao = None
            self._atualizar_candidatas()
            return
        if not self.codificada:
            return  # _codificada chama de novo quando o embedding chegar
        self._enviar(self._tarefa_prever, self.modelo, self.rel, self.chave, self.id_pedido,
                     list(self.pontos), self.caixa)

    def _tarefa_prever(self, modelo, rel, chave, id_pedido, pontos, caixa):
        if rel != self.rel or id_pedido != self.id_pedido or modelo is not self.modelo or not modelo.carregado:
            return  # já há pedido mais novo na fila
        p = modelo.prever(chave, pontos, caixa)
        self._devolver(self._predicao_pronta, rel, id_pedido, p)

    def _predicao_pronta(self, rel, id_pedido, p):
        if rel != self.rel or id_pedido != self.id_pedido:
            return
        self.predicao = p
        self.escolhida = int(np.argmax(p.escores))
        self.tempo_ultimo_clique = p.tempo_s
        if self.relatorio:
            self.relatorio.predicoes.append(p.tempo_s)
        self._atualizar_candidatas()
        self._redesenhar()
        self._atualizar_status()

    def _mascara_escolhida(self):
        if self.predicao is None:
            return None
        return self.predicao.mascaras[min(self.escolhida, len(self.predicao.mascaras) - 1)]

    def escolher(self, i: int):
        if self.predicao is None or i >= len(self.predicao.mascaras):
            return
        self.escolhida = i
        self._atualizar_candidatas()
        self._redesenhar()

    def _atualizar_candidatas(self):
        p = self.predicao
        if p is None or self.imagem_rgb is None:
            for i, b in enumerate(self.botoes_cand):
                b.configure(image=self._img_vazia, text=f"{i + 1}: -", border_color=("gray70", "gray30"))
            self.rotulo_origem.configure(text="")
            return
        self.rotulo_origem.configure(text=f"Origem: {self.origem_atual()}")
        # recorte em torno da união, para a diferença entre as candidatas ficar visível
        uniao = p.mascaras.any(axis=0)
        ys, xs = np.nonzero(uniao)
        h, w = uniao.shape
        if len(xs):
            mx = int(0.15 * (xs.max() - xs.min() + 1)) + 8
            my = int(0.15 * (ys.max() - ys.min() + 1)) + 8
            x0, x1 = max(0, xs.min() - mx), min(w, xs.max() + mx + 1)
            y0, y1 = max(0, ys.min() - my), min(h, ys.max() + my + 1)
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        self._imgs_cand = []
        for i, b in enumerate(self.botoes_cand):
            if i >= len(p.mascaras):
                b.configure(image=self._img_vazia, text=f"{i + 1}: -", border_color=("gray70", "gray30"))
                continue
            rec = self.imagem_rgb[y0:y1, x0:x1].copy()
            _sobrepor(rec, p.mascaras[i, y0:y1, x0:x1].astype(np.uint8), COR_ATUAL, 0.5,
                      COR_CONTORNO_ATUAL, max(1, int(max(rec.shape[:2]) / 120)))
            im = Image.fromarray(rec)
            im.thumbnail((84, 56))
            ci = ctk.CTkImage(im, size=im.size)
            self._imgs_cand.append(ci)
            escolhida = i == self.escolhida
            b.configure(image=ci, text=f"{i + 1}: {_fmt(float(p.escores[i]), 2)}",
                        border_color=("#c8a600", "#ffff00") if escolhida else ("gray70", "gray30"))

    # ================================================================ modo texto

    def _motor_texto(self) -> str:
        rotulo = self.combo_motor.get()
        return next(k for k, v in MOTORES.items() if v == rotulo)

    def buscar_texto(self, texto: str | None = None, motor: str | None = None):
        texto = (texto if texto is not None else self.entrada_texto.get()).strip()
        if not texto:
            self._mensagem("Escreva o que procurar, em inglês (smoke, car, person).", alerta=True)
            return
        if self.imagem_rgb is None or self.erro_modelo:
            return
        motor = motor or self._motor_texto()
        motivo = requisitos_gdino() if motor == "gdino" else requisitos_sam3(verificar_acesso=False)
        if motivo:
            self._oferecer_gdino(motor, motivo, texto)
            return
        try:
            lc = float(self.limiar_caixa.get().replace(",", "."))
            lt = float(self.limiar_texto.get().replace(",", "."))
        except ValueError:
            self._mensagem("Limiares precisam ser números entre 0 e 1.", alerta=True)
            return
        self.canvas.focus_set()
        self.botao_buscar.configure(state="disabled")
        self.rotulo_texto.configure(text=f"Buscando '{texto}' com {MOTORES[motor]}..."
                                         + (" (a primeira vez carrega o detector)" if motor == "gdino" and self.detector is None else ""))
        self._enviar(self._tarefa_texto, self.modelo, self.rel, self.chave, self.imagem_rgb, texto, motor, lc, lt)

    def _tarefa_texto(self, modelo, rel, chave, imagem, texto, motor, lc, lt):
        try:
            if not modelo.carregado:
                raise ErroTexto("O SAM ainda não carregou.")
            if motor == "gdino":
                if self.detector is None or self.detector.dispositivo != modelo.dispositivo:
                    self.detector = DetectorGdino(modelo.dispositivo)
                    self.detector.carregar()
                modelo.codificar(chave, imagem)  # garante o embedding (vem do cache)
                r = texto_com_gdino(self.detector, modelo, chave, imagem, texto, lc, lt)
            else:
                from .texto import ResultadoTexto
                if self.sam3 is None:
                    self.sam3 = SegmentadorSam3(modelo.dispositivo)
                    self.sam3.carregar()
                inst, t = self.sam3.segmentar(imagem, texto, lc)
                r = ResultadoTexto(texto, "sam3", inst, t, 0.0)
        except ErroTexto as e:
            self._devolver(self._texto_falhou, motor, str(e), texto)
            return
        self._devolver(self._texto_pronto, rel, r)

    def _oferecer_gdino(self, motor, motivo, texto):
        self.rotulo_texto.configure(text=motivo.splitlines()[0])
        if motor == "sam3" and requisitos_gdino() is None:
            if messagebox.askyesno("SAM 3 indisponível", motivo + "\n\nUsar agora o Grounding DINO tiny + SAM?"):
                self.combo_motor.set(MOTORES["gdino"])
                self.buscar_texto(texto, "gdino")
        else:
            messagebox.showerror("Modo texto indisponível", motivo)

    def _texto_falhou(self, motor, motivo, texto):
        self.botao_buscar.configure(state="normal")
        self._oferecer_gdino(motor, motivo, texto)

    def _texto_pronto(self, rel, r):
        self.botao_buscar.configure(state="normal")
        if rel != self.rel:
            return
        self.texto_atual = r.texto
        self.instancias_texto = r.instancias
        self._mensagem(f"Texto '{r.texto}': {len(r.instancias)} instâncias; detecção {_fmt(r.tempo_deteccao_s, 2)} s"
                       + (f", SAM {_fmt(r.tempo_sam_s, 2)} s" if r.motor == "gdino" else ""))
        if self.mapear_classe.get():
            classe = classe_para_texto(r.texto, self.projeto.classes)
            if classe:
                self.combo_classe.set(classe)
        if not r.instancias:
            self.rotulo_texto.configure(text=f"Nada achado para '{r.texto}'. Baixe os limiares ou troque o texto.")
            self._zerar_prompt()
            self._atualizar_candidatas()
            self._redesenhar()
            return
        self._ir_instancia(0)

    def _ir_instancia(self, i: int):
        if not self.instancias_texto:
            return
        if i >= len(self.instancias_texto):
            self.rotulo_texto.configure(text="Fim das instâncias deste texto.")
            self.instancias_texto = []
            self._zerar_prompt()
            self._atualizar_candidatas()
            self._redesenhar()
            return
        self.indice_texto = i
        inst = self.instancias_texto[i]
        self._zerar_prompt()
        self.origem_base = inst.origem
        self.caixa = inst.caixa
        self.predicao = Predicao(inst.mascaras, inst.escores, 0.0)
        self.escolhida = int(np.argmax(inst.escores))
        self.inicio_objeto = time.perf_counter()
        self.rotulo_texto.configure(
            text=f"Instância {i + 1} de {len(self.instancias_texto)}: '{inst.rotulo}', "
                 f"confiança {_fmt(inst.escore_caixa, 2)}. Enter aceita, Del descarta, cliques ajustam.")
        self.abas.set("Texto")
        self._atualizar_candidatas()
        self._redesenhar()

    def descartar_instancia(self):
        if self.instancias_texto:
            self._ir_instancia(self.indice_texto + 1)

    # ================================================================ objetos

    def aceitar(self):
        m = self._mascara_escolhida()
        if m is None:
            self._mensagem("Nada para aceitar: clique no objeto primeiro.", alerta=True)
            return
        if not m.any():
            self._mensagem("A máscara escolhida está vazia; escolha outra (1, 2, 3).", alerta=True)
            return
        classe = self.combo_classe.get() or self.projeto.classes[0]
        tempo = time.perf_counter() - (self.inicio_objeto or time.perf_counter())
        escore = float(self.predicao.escores[self.escolhida])
        iou_real = iou_contra_referencia(m, self.referencia) if self.referencia is not None else None
        origem = self.origem_atual()
        obj = self.projeto.adicionar_objeto(
            self.rel, m, classe, escore, self.escolhida, self.cliques_objeto, tempo,
            pontos=self.pontos, caixa=self.caixa, iou_real=iou_real, origem=origem)
        img = self.projeto.imagens[self.rel]
        self.iou_uniao = None
        if self.referencia is not None:
            self.iou_uniao = iou_uniao([o.mascara() for o in img.objetos], self.referencia)
        if self.relatorio:
            self.relatorio.adicionar(self.rel, obj.id, classe, obj.cliques, tempo, escore, iou_real,
                                     img.tempo_codificacao_s, self.iou_uniao, origem)
        self.ultimo_objeto_txt = f"Último: {obj.cliques} cliques, {_fmt(tempo, 1)} s"
        self._mostrar_avaliacao(obj)
        texto_pendente = bool(self.instancias_texto)
        self._zerar_prompt()
        self.salvar()
        self._atualizar_objetos()
        self._atualizar_item(self.rel)
        self._refazer_camada_objetos()
        self._mensagem(f"Objeto #{obj.id} ({classe}) aceito e salvo. Origem: {origem}.")
        if texto_pendente:
            self._ir_instancia(self.indice_texto + 1)
        else:
            self._atualizar_candidatas()

    def _mostrar_avaliacao(self, obj):
        if obj.iou_real is None:
            self.rotulo_avaliacao.configure(
                text=f"#{obj.id}: escore previsto {_fmt(obj.escore_previsto, 3)}. Sem referência para medir a IoU real.",
                text_color=COR_TEXTO)
            return
        dif = obj.escore_previsto - obj.iou_real
        grande = diferenca_grande(obj.escore_previsto, obj.iou_real)
        aviso = (f"\nDIFERENÇA GRANDE ({_fmt(dif, 2)}): o modelo {'superestimou' if dif > 0 else 'subestimou'} "
                 "a própria máscara." if grande else f"\nDiferença {_fmt(dif, 2)} (abaixo de {_fmt(LIMIAR_DIFERENCA, 2)}).")
        uniao = self.iou_uniao
        if uniao is not None and len(self.projeto.imagens[self.rel].objetos) > 1:
            aviso += f"\nUnião dos objetos desta imagem contra a referência: IoU {_fmt(uniao, 3)}."
        self.rotulo_avaliacao.configure(
            text=f"#{obj.id}: escore previsto {_fmt(obj.escore_previsto, 3)}  |  IoU real {_fmt(obj.iou_real, 3)}{aviso}",
            text_color=COR_ALERTA if grande else COR_OK)

    def _atualizar_objetos(self):
        for w in self.quadro_objetos.winfo_children():
            w.destroy()
        if self.rel is None or self.projeto is None:
            return
        objs = self.projeto.imagens[self.rel].objetos
        if not objs:
            ctk.CTkLabel(self.quadro_objetos, text="(nenhum)", text_color="gray50").pack(anchor="w", padx=6)
        for obj in objs:
            linha = ctk.CTkFrame(self.quadro_objetos, fg_color="transparent")
            linha.pack(fill="x", pady=1)
            cor = "#%02x%02x%02x" % self._cor_classe(obj.classe)
            ctk.CTkLabel(linha, text="", width=10, height=34, fg_color=cor, corner_radius=3).pack(side="left", padx=(4, 4))
            txt = f"#{obj.id} {obj.classe}  esc {_fmt(obj.escore_previsto, 2)}"
            if obj.iou_real is not None:
                txt += f"  IoU {_fmt(obj.iou_real, 2)}"
            origem = obj.origem or "SAM 2.1 (cliques)"
            txt += "\n" + (origem if len(origem) <= 34 else origem[:32] + "...")
            grande = diferenca_grande(obj.escore_previsto, obj.iou_real)
            ctk.CTkLabel(linha, text=txt, justify="left", anchor="w", font=ctk.CTkFont(size=11),
                         text_color=COR_ALERTA if grande else COR_TEXTO).pack(side="left", fill="x", expand=True)
            ctk.CTkButton(linha, text="Apagar", width=50, height=24, fg_color="gray40",
                          command=lambda i=obj.id: self.apagar_objeto(i)).pack(side="right", padx=2)
        self._atualizar_status()

    def apagar_objeto(self, id_objeto: int):
        if self.projeto.remover_objeto(self.rel, id_objeto):
            if self.relatorio:
                self.relatorio.marcar_apagado(self.rel, id_objeto)
            self.salvar()
            self._atualizar_objetos()
            self._atualizar_item(self.rel)
            self._refazer_camada_objetos()
            self._mensagem(f"Objeto #{id_objeto} apagado.")

    def _alternar_concluida(self):
        if self.rel is None:
            return
        self.projeto.imagens[self.rel].concluida = bool(self.chave_concluida.get())
        self.salvar()
        self._atualizar_item(self.rel)

    def concluir_e_avancar(self):
        if self.rel is None:
            return
        self.projeto.imagens[self.rel].concluida = True
        self.salvar()
        self._atualizar_item(self.rel)
        pendentes = [r for r in self.lista if not self.projeto.imagens[r].concluida]
        if pendentes:
            i = self.lista.index(self.rel)
            depois = [r for r in self.lista[i + 1:] + self.lista[:i] if r in pendentes]
            self.abrir_imagem(depois[0])
        else:
            self._mensagem("Todas as imagens estão concluídas. Exporte com Ctrl+E.")

    # ================================================================ classes

    def _atualizar_combo(self):
        atual = self.combo_classe.get()
        self.combo_classe.configure(values=self.projeto.classes)
        self.combo_classe.set(atual if atual in self.projeto.classes else self.projeto.classes[0])

    def adicionar_classe(self):
        nome = self.entrada_classe.get().strip()
        if not nome:
            return
        if nome not in self.projeto.classes:
            self.projeto.classes.append(nome)
            self.salvar()
        self._atualizar_combo()
        self.combo_classe.set(nome)
        self.entrada_classe.delete(0, "end")
        self.canvas.focus_set()

    def remover_classe(self):
        nome = self.combo_classe.get()
        if self.projeto.classe_em_uso(nome):
            messagebox.showinfo("Classe em uso", f"A classe '{nome}' tem objetos. Apague-os antes.")
            return
        if len(self.projeto.classes) <= 1:
            return
        self.projeto.classes.remove(nome)
        self.salvar()
        self._atualizar_combo()

    # ================================================================ salvar, exportar, sobre

    def salvar(self):
        if self.projeto is None:
            return
        try:
            self.projeto.salvar()
            if self.relatorio:
                self.relatorio.salvar()
        except OSError as e:
            self._mensagem(f"Falha ao salvar: {e}", alerta=True)

    def exportar(self):
        """Abre o diálogo de exportação para treino."""
        if self.projeto is None:
            return None
        return DialogoExportar(self)

    def exportar_treino(self, opcoes: OpcoesTreino | None = None, pasta: Path | None = None):
        """Mesma exportação do diálogo, sem janela (usada nos testes)."""
        self.salvar()
        r = exportar_treino(self.projeto, pasta, opcoes)
        self.ultima_exportacao = r
        self._mensagem(f"Exportado para treino em {r.pasta}")
        return r

    def mostrar_sobre(self):
        j = ctk.CTkToplevel(self)
        j.title("Sobre o sam-anota")
        j.geometry("600x520")
        j.transient(self)
        t = ctk.CTkTextbox(j, wrap="word")
        t.pack(fill="both", expand=True, padx=10, pady=10)
        t.insert("1.0", TEXTO_SOBRE)
        t.configure(state="disabled")
        ctk.CTkButton(j, text="Fechar", command=j.destroy).pack(pady=(0, 10))
        j.after(100, j.focus_set)

    def _mudar_tema(self, valor):
        modo = "dark" if valor == "Escuro" else "light"
        ctk.set_appearance_mode(modo)
        self.canvas.configure(bg=FUNDO_CANVAS[modo])

    # ================================================================ status

    def _mensagem(self, texto: str, alerta: bool = False):
        self.status["mensagem"].configure(text=texto, text_color=COR_ALERTA if alerta else COR_TEXTO)

    def _atualizar_status(self):
        s = self.status
        if self.tempo_codificacao is not None:
            extra = ""
            if self.origem_codificacao != "calculado":
                orig = self.projeto.imagens[self.rel].tempo_codificacao_s if self.rel else None
                extra = f" ({self.origem_codificacao}" + (f"; calculada em {_fmt(orig, 2)} s)" if orig else ")")
            s["codificacao"].configure(text=f"Codificação: {_fmt(self.tempo_codificacao, 2)} s{extra}")
        elif self.inicio_codificacao is not None:
            s["codificacao"].configure(text=f"Codificando... {_fmt(time.perf_counter() - self.inicio_codificacao, 1)} s")
        else:
            s["codificacao"].configure(text="Codificação: -")
        s["clique"].configure(text="Último clique: " + (
            f"{_fmt(self.tempo_ultimo_clique, 3)} s" if self.tempo_ultimo_clique is not None else "-"))
        s["cliques"].configure(text=f"Cliques no objeto: {self.cliques_objeto}")
        t = time.perf_counter() - self.inicio_objeto if self.inicio_objeto else 0.0
        s["tempo_obj"].configure(text=f"Tempo no objeto: {_fmt(t, 1)} s" +
                                 (f"  ({self.ultimo_objeto_txt})" if self.ultimo_objeto_txt else ""))
        if self.projeto:
            n_img = len(self.projeto.imagens[self.rel].objetos) if self.rel else 0
            s["objetos"].configure(text=f"Objetos: {n_img} nesta imagem, {self.projeto.total_objetos()} no total")

    def _tique(self):
        self._atualizar_status()
        if self.inicio_codificacao is not None:
            self._desenhar_aviso_codificando()
        self.after(200, self._tique)

    def fechar(self):
        self.salvar()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()


def principal(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m sam_anota", description="Anotação de imagens com SAM")
    ap.add_argument("--pasta", type=Path, default=None, help="pasta de imagens (padrão: dados/ do repositório)")
    ap.add_argument("--modelo", default=MODELO_PADRAO,
                    help="sam2.1_tiny (padrão), sam2.1_small, sam2.1_base_plus, sam2.1_large, sam1_vit_b, "
                         "sam1_vit_l, sam1_vit_h (ou tiny, small, base_plus, large, vit_b, vit_l, vit_h)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="caminho do peso (padrão: checkpoints/)")
    ap.add_argument("--dispositivo", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--sem-cache-disco", action="store_true", help="não grava embeddings em saidas/")
    ap.add_argument("--tema", choices=["escuro", "claro", "sistema"], default="escuro")
    a = ap.parse_args(argv)
    ctk.set_appearance_mode({"escuro": "dark", "claro": "light", "sistema": "system"}[a.tema])
    app = App(pasta=a.pasta, modelo=resolver_chave(a.modelo), dispositivo=a.dispositivo,
              cache_disco=not a.sem_cache_disco, checkpoint=a.checkpoint)
    app.mainloop()
