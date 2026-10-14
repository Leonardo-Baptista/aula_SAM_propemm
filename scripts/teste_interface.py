"""Teste de ponta a ponta da interface, com o SAM de verdade.

Abre a janela, carrega imagens, simula cliques e arrastos no canvas (eventos Tk),
usa o seletor de ferramenta, troca de modelo e de dispositivo, roda o modo texto,
aceita objetos, exporta para treino e confere os arquivos. Mede os tempos reais.

    python scripts/teste_interface.py                     # saídas numa pasta temporária
    python scripts/teste_interface.py --captura docs/tela.png
    python scripts/teste_interface.py --sem-texto         # pula o modo texto (sem transformers)

As caixas de diálogo (avisos, perguntas) são trocadas por registradores, para o teste
não parar esperando clique; o texto de cada uma é conferido.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
DIALOGOS: list[tuple[str, str, str]] = []


def esperar(app, condicao, limite_s=180.0, passo_s=0.02):
    t0 = time.perf_counter()
    while not condicao():
        app.update()
        if time.perf_counter() - t0 > limite_s:
            raise TimeoutError("tempo esgotado esperando a interface")
        time.sleep(passo_s)
    return time.perf_counter() - t0


def pausa(app, s):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < s:
        app.update()
        time.sleep(0.01)


def clicar(app, x, y, botao=1):
    """Clique simulado em coordenadas da IMAGEM, convertidas para a tela."""
    sx, sy = (int(round(v)) for v in app._para_tela(x, y))
    app.canvas.event_generate(f"<ButtonPress-{botao}>", x=sx, y=sy)
    if botao == 1:
        app.canvas.event_generate("<ButtonRelease-1>", x=sx, y=sy)
    app.update()


def arrastar(app, x0, y0, x1, y1):
    a = [int(round(v)) for v in app._para_tela(x0, y0)]
    b = [int(round(v)) for v in app._para_tela(x1, y1)]
    app.canvas.event_generate("<ButtonPress-1>", x=a[0], y=a[1])
    app.canvas.event_generate("<B1-Motion>", x=(a[0] + b[0]) // 2, y=(a[1] + b[1]) // 2)
    app.canvas.event_generate("<B1-Motion>", x=b[0], y=b[1])
    app.canvas.event_generate("<ButtonRelease-1>", x=b[0], y=b[1])
    app.update()


def esperar_predicao(app):
    pedido = app.id_pedido
    return esperar(app, lambda: app.predicao is not None and app.id_pedido == pedido
                   and getattr(app, "_ultimo_id_aplicado", None) == pedido, 60)


def esperar_modelo(app):
    return esperar(app, lambda: (app.modelo_pronto and app.codificada) or app.erro_modelo, 600)


def cena_sintetica(pasta: Path) -> Path:
    """Fundo texturizado, um disco (referência) e um retângulo distrator."""
    rng = np.random.default_rng(7)
    fundo = (rng.normal(120, 18, (600, 800, 3))).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(fundo)
    d = ImageDraw.Draw(im)
    d.ellipse((180, 160, 420, 400), fill=(220, 60, 50))
    d.rectangle((500, 220, 700, 420), fill=(40, 90, 200))
    caminho = pasta / "sintetica_disco.png"
    im.save(caminho)
    ref = Image.new("L", (800, 600), 0)
    ImageDraw.Draw(ref).ellipse((180, 160, 420, 400), fill=255)
    ref.save(pasta / "sintetica_disco_ref.png")
    return caminho


def substituir_dialogos(modulo_app):
    mb = modulo_app.messagebox

    def registrar(tipo, resposta):
        def f(titulo, texto="", **k):
            DIALOGOS.append((tipo, titulo, texto))
            print(f"  [diálogo {tipo}] {titulo}: {texto.splitlines()[0] if texto else ''}")
            return resposta
        return f
    mb.showwarning = registrar("aviso", None)
    mb.showerror = registrar("erro", None)
    mb.showinfo = registrar("info", None)
    mb.askyesno = registrar("pergunta", False)  # não baixa nada e não troca de motor sozinho


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--captura", type=Path, default=None, help="salva a tela da janela neste PNG")
    ap.add_argument("--sem-texto", action="store_true")
    ap.add_argument("--modelos", default="sam2.1_small,sam1_vit_b",
                    help="modelos extras para testar a troca (só os já baixados)")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="sam_anota_teste_"))
    os.environ["SAM_ANOTA_SAIDAS"] = str(tmp / "saidas")
    import customtkinter as ctk

    from sam_anota import app as modulo_app
    from sam_anota.app import App
    from sam_anota.exportar import OpcoesTreino
    from sam_anota.modelo import CATALOGO, familia_instalada

    substituir_dialogos(modulo_app)
    original = App._predicao_pronta

    def marcado(self, rel, id_pedido, p):
        original(self, rel, id_pedido, p)
        if rel == self.rel and id_pedido == self.id_pedido:
            self._ultimo_id_aplicado = id_pedido
    App._predicao_pronta = marcado

    ctk.set_appearance_mode("dark")
    t_janela = time.perf_counter()
    app = App(pasta=RAIZ / "dados" / "exemplos")
    app.geometry("1366x740+0+0")
    app.update()
    print(f"janela aberta em {time.perf_counter() - t_janela:.2f} s")
    tempos = {"predicao_s": []}

    esperar_modelo(app)
    if app.erro_modelo:
        print("FALHA: modelo não carregou:", app.erro_modelo)
        return 1
    print(f"modelo: {app.modelo.descricao()}, carga {app.modelo.tempo_carga_s:.2f} s")
    assert app.rel == "cao_e_gato.jpg", app.rel
    tempos["codificacao_s"] = app.tempo_codificacao
    print(f"codificação cao_e_gato: {app.tempo_codificacao:.2f} s ({app.origem_codificacao})")

    # ---------------- cliques: gato com dois pontos positivos (ferramenta Ponto +)
    app.combo_classe.set("gato")
    clicar(app, 420, 650)
    esperar_predicao(app)
    tempos["predicao_s"].append(app.tempo_ultimo_clique)
    clicar(app, 800, 450)
    esperar_predicao(app)
    tempos["predicao_s"].append(app.tempo_ultimo_clique)
    print(f"gato, 2 cliques: escores {np.round(app.predicao.escores, 3)}")
    app.event_generate("<Return>")
    app.update()
    assert app.projeto.total_objetos() == 1, "Enter não aceitou o objeto"
    assert "SAM 2.1 tiny (cliques)" in app.projeto.imagens["cao_e_gato.jpg"].objetos[0].origem

    # ---------------- ferramenta Ponto -: clique esquerdo vira negativo; tecla P volta
    app.combo_classe.set("cao")
    clicar(app, 1250, 650)
    esperar_predicao(app)
    clicar(app, 560, 180)
    esperar_predicao(app)
    app.event_generate("<Key-n>")
    app.update()
    assert app.ferramenta == "neg" and app.seletor.get() == "Ponto -"
    clicar(app, 500, 650)  # esquerdo, mas negativo pela ferramenta
    esperar_predicao(app)
    assert app.pontos[-1][2] == 0
    app.desfazer()
    esperar_predicao(app)
    clicar(app, 500, 650, botao=3)
    esperar_predicao(app)
    app.event_generate("<Key-p>")
    app.update()
    assert app.ferramenta == "pos"
    app.escolher(int(np.argmax(app.predicao.escores)))
    app.aceitar()
    cao = app.projeto.imagens["cao_e_gato.jpg"].objetos[-1]
    print(f"cão aceito: {cao.cliques} cliques, {cao.tempo_s:.1f} s, escore {cao.escore_previsto:.3f}")

    # ---------------- ferramenta Caixa: caixa + ponto no mesmo objeto, caixa fica visível
    app.abrir_imagem("transito_pune.jpg")
    esperar_modelo(app)
    tempos["codificacao_pune_s"] = app.tempo_codificacao
    app.definir_ferramenta("Caixa")
    clicar(app, 700, 500)  # clique com a ferramenta Caixa não cria ponto
    assert not app.pontos and app.caixa is None
    arrastar(app, 560, 330, 1010, 700)
    esperar_predicao(app)
    app.definir_ferramenta("Ponto +")
    clicar(app, 780, 520)
    esperar_predicao(app)
    assert app.caixa is not None and len(app.pontos) == 1
    assert app.canvas.find_overlapping(*app._para_tela(560, 330), *app._para_tela(562, 332))
    print(f"pune, caixa + ponto: escores {np.round(app.predicao.escores, 3)}")
    app.combo_classe.set("carro")
    app.aceitar()
    assert app.caixa is None  # a caixa some ao aceitar

    # ---------------- dispositivo: pedir GPU
    import torch
    n_dialogos = len(DIALOGOS)
    app._ao_escolher_dispositivo("GPU (CUDA)")
    esperar_modelo(app)
    if torch.cuda.is_available():
        assert app.modelo.dispositivo == "cuda"
        print(f"GPU: {app.rotulo_modelo.cget('text')}; codificação {app.tempo_codificacao:.2f} s")
    else:
        assert app.modelo.dispositivo == "cpu", app.modelo.dispositivo
        aviso = [d for d in DIALOGOS[n_dialogos:] if d[0] == "aviso"]
        assert aviso and "CUDA" in aviso[0][2], DIALOGOS[n_dialogos:]
        assert app.combo_disp.get() == "CPU"
        print("GPU recusada com aviso e volta para CPU (torch sem CUDA)")
    tempos["dispositivo"] = app.modelo.dispositivo
    print(f"barra: {app.rotulo_modelo.cget('text')}")

    # ---------------- troca de modelo (só os já baixados)
    for chave in [c for c in a.modelos.split(",") if c]:
        info = CATALOGO[chave]
        if not (info.baixado and familia_instalada(info.familia)):
            print(f"pulando {chave} (não baixado ou pacote ausente)")
            continue
        app.trocar_modelo(chave)
        esperar_modelo(app)
        clicar(app, 780, 520)
        esperar_predicao(app)
        print(f"{info.rotulo}: carga {app.modelo.tempo_carga_s:.2f} s, codificação {app.tempo_codificacao:.2f} s, "
              f"predição {app.tempo_ultimo_clique:.3f} s")
        tempos[f"codificacao_{chave}_s"] = app.tempo_codificacao
        app.limpar()
    app.trocar_modelo("sam2.1_tiny")
    esperar_modelo(app)

    # ---------------- modo texto
    if not a.sem_texto:
        app.abrir_imagem("cao_e_gato.jpg")
        esperar_modelo(app)
        n = len(DIALOGOS)
        app.buscar_texto("smoke", "sam3")  # sem token: tem de explicar e oferecer o outro motor
        esperar(app, lambda: len(DIALOGOS) > n, 120)
        motivo = DIALOGOS[n][2]
        assert "facebook/sam3" in motivo, motivo
        print(f"SAM 3: recusado com motivo: {motivo.splitlines()[0]}")
        t0 = time.perf_counter()
        app.entrada_texto.delete(0, "end")
        app.entrada_texto.insert(0, "cat")
        app.buscar_texto(motor="gdino")
        esperar(app, lambda: app.botao_buscar.cget("state") == "normal" and app.instancias_texto, 600)
        tempos["texto_gdino_s"] = time.perf_counter() - t0
        print(f"texto 'cat': {len(app.instancias_texto)} instâncias em {tempos['texto_gdino_s']:.1f} s "
              f"({app.status['mensagem'].cget('text')})")
        assert app.combo_classe.get() == "gato"  # classe mapeada pelo texto
        assert app.predicao is not None and app.caixa is not None
        app.aceitar()
        obj = app.projeto.imagens["cao_e_gato.jpg"].objetos[-1]
        assert "Grounding DINO tiny + SAM 2.1 tiny" in obj.origem, obj.origem
        print(f"objeto do texto aceito: origem '{obj.origem}', escore {obj.escore_previsto:.3f}")

    # ---------------- exportação para treino (mesma função do diálogo)
    app.projeto.imagens["transito_pune.jpg"].concluida = True
    r = app.exportar_treino(OpcoesTreino(frac_val=0.34, semente=42))
    for formato in ("yolo", "coco", "semantica"):
        assert (r.pasta / formato / "README_TREINO.md").exists()
    coco = json.loads((r.pasta / "coco" / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
    print(f"exportação: treino {r.treino}, val {r.val}, {r.objetos} objetos; "
          f"COCO train com {len(coco['annotations'])} anotações")
    dlg = app.exportar()  # o diálogo abre e fecha
    app.update()
    assert dlg.winfo_exists()
    dlg.destroy()

    # ---------------- captura: modo texto numa imagem difícil, com IoU real
    if a.captura and not a.sem_texto:
        app.abrir_pasta(RAIZ / "dados")
        rel = "dificeis/fumaca/02_ijmond_coqueria2_pluma_chamine.jpg"
        app.entrada_classe.insert(0, "fumaca")
        app.adicionar_classe()
        app.abrir_imagem(rel)
        esperar_modelo(app)
        app.entrada_texto.delete(0, "end")
        app.entrada_texto.insert(0, "smoke")
        app.buscar_texto(motor="gdino")
        esperar(app, lambda: app.botao_buscar.cget("state") == "normal", 600)
        if app.instancias_texto:
            app.aceitar()  # primeira instância: mostra escore previsto contra IoU real
        app.abas.set("Texto")
        app.lift()
        app.attributes("-topmost", True)
        pausa(app, 1.2)
        from PIL import ImageGrab
        x, y = app.winfo_rootx(), app.winfo_rooty()
        w, h = app.winfo_width(), app.winfo_height()
        a.captura.parent.mkdir(parents=True, exist_ok=True)
        ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(a.captura)
        app.attributes("-topmost", False)
        print(f"captura salva em {a.captura} ({w}x{h})")

    # ---------------- cena sintética com referência
    pasta_sint = tmp / "sintetica"
    pasta_sint.mkdir()
    cena_sintetica(pasta_sint)
    app.abrir_pasta(pasta_sint)
    assert app.lista == ["sintetica_disco.png"], app.lista
    esperar_modelo(app)
    clicar(app, 300, 280)
    esperar_predicao(app)
    app.escolher(int(np.argmax(app.predicao.escores)))
    app.aceitar()
    obj = app.projeto.imagens["sintetica_disco.png"].objetos[0]
    print(f"disco sintético: escore previsto {obj.escore_previsto:.3f}, IoU real {obj.iou_real:.3f}")
    assert obj.iou_real > 0.9
    rel_csv = app.relatorio.salvar()
    assert "origem" in rel_csv[0].read_text(encoding="utf-8").splitlines()[0]

    app.fechar()
    pred = tempos["predicao_s"]
    print(f"\nRESUMO: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in tempos.items() if k != 'predicao_s'}, ensure_ascii=False)}"
          f" | predição {min(pred):.3f} a {max(pred):.3f} s")
    print(f"saidas do teste em {tmp}")
    print("Teste da interface ok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
