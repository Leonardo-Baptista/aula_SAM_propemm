"""Diálogos da interface: exportação para treino."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from .exportar import FORMATOS, OpcoesTreino, exportar_treino, selecionar_imagens


def abrir_no_sistema(pasta: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(pasta)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(pasta)])
        else:
            subprocess.Popen(["xdg-open", str(pasta)])
    except Exception:
        pass


class DialogoExportar(ctk.CTkToplevel):
    """Escolhe formatos, divisão treino e validação e quais imagens entram."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Exportar para treino")
        self.geometry("470x520")
        self.resizable(False, False)
        self.transient(app)
        self.resultado = None
        negrito = ctk.CTkFont(weight="bold")

        ctk.CTkLabel(self, text="Formatos (cada um numa pasta própria)", font=negrito).pack(anchor="w", padx=14, pady=(12, 2))
        self.formatos = {}
        for chave, rotulo in FORMATOS.items():
            v = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(self, text=rotulo, variable=v).pack(anchor="w", padx=24, pady=2)
            self.formatos[chave] = v

        ctk.CTkLabel(self, text="Divisão por imagem", font=negrito).pack(anchor="w", padx=14, pady=(10, 2))
        linha = ctk.CTkFrame(self, fg_color="transparent")
        linha.pack(fill="x", padx=24)
        ctk.CTkLabel(linha, text="Validação (%)").pack(side="left")
        self.val = ctk.CTkEntry(linha, width=50)
        self.val.insert(0, "20")
        self.val.pack(side="left", padx=(6, 18))
        ctk.CTkLabel(linha, text="Semente").pack(side="left")
        self.semente = ctk.CTkEntry(linha, width=60)
        self.semente.insert(0, "42")
        self.semente.pack(side="left", padx=6)

        ctk.CTkLabel(self, text="Quais imagens", font=negrito).pack(anchor="w", padx=14, pady=(10, 2))
        self.so_concluidas = ctk.StringVar(value="todas")
        ctk.CTkRadioButton(self, text="Todas as anotadas (com objeto ou concluídas)", variable=self.so_concluidas,
                           value="todas", command=self._atualizar).pack(anchor="w", padx=24, pady=2)
        ctk.CTkRadioButton(self, text="Só as marcadas como concluídas", variable=self.so_concluidas,
                           value="concluidas", command=self._atualizar).pack(anchor="w", padx=24, pady=2)
        self.vazias = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self, text="Incluir imagens concluídas sem objeto (negativas)", variable=self.vazias,
                        command=self._atualizar).pack(anchor="w", padx=24, pady=(6, 2))

        self.rotulo_info = ctk.CTkLabel(self, text="", justify="left", anchor="w", wraplength=430)
        self.rotulo_info.pack(fill="x", padx=14, pady=(10, 4))
        self.barra = ctk.CTkProgressBar(self, mode="indeterminate")
        botoes = ctk.CTkFrame(self, fg_color="transparent")
        botoes.pack(side="bottom", fill="x", padx=14, pady=12)
        self.botao = ctk.CTkButton(botoes, text="Exportar", command=self.exportar)
        self.botao.pack(side="right")
        ctk.CTkButton(botoes, text="Fechar", fg_color="gray40", command=self.destroy).pack(side="right", padx=8)
        self._atualizar()
        self.after(150, self.grab_set)

    def opcoes(self) -> OpcoesTreino:
        formatos = tuple(k for k, v in self.formatos.items() if v.get())
        try:
            frac = float(self.val.get().replace(",", ".")) / 100
            semente = int(self.semente.get())
        except ValueError:
            raise ValueError("Validação (%) e semente precisam ser números.") from None
        if not 0 <= frac < 1:
            raise ValueError("Validação precisa estar entre 0 e 99%.")
        return OpcoesTreino(formatos=formatos, frac_val=frac, semente=semente,
                            incluir_vazias=self.vazias.get(), so_concluidas=self.so_concluidas.get() == "concluidas")

    def _atualizar(self):
        try:
            n = len(selecionar_imagens(self.app.projeto, self.opcoes()))
        except ValueError:
            n = 0
        self.rotulo_info.configure(text=f"{n} imagens entram. Saída: {self.app.projeto.pasta_saida}"
                                        f"{os.sep}treino_<data e hora>")

    def exportar(self):
        try:
            op = self.opcoes()
        except ValueError as e:
            messagebox.showerror("Exportar", str(e), parent=self)
            return
        if not op.formatos:
            messagebox.showerror("Exportar", "Escolha ao menos um formato.", parent=self)
            return
        self.app.salvar()
        self.botao.configure(state="disabled")
        self.barra.pack(fill="x", padx=14)
        self.barra.start()

        def tarefa():
            try:
                r = exportar_treino(self.app.projeto, op=op,
                                    progresso=lambda t: self.app.fila.put((self._progresso, (t,))))
                self.app.fila.put((self._concluido, (r, None)))
            except Exception as e:
                self.app.fila.put((self._concluido, (None, e)))
        threading.Thread(target=tarefa, daemon=True).start()

    def _progresso(self, texto):
        if self.winfo_exists():
            self.rotulo_info.configure(text=texto)

    def _concluido(self, r, erro):
        if not self.winfo_exists():
            return
        self.barra.stop()
        self.barra.pack_forget()
        self.botao.configure(state="normal")
        if erro is not None:
            messagebox.showerror("Exportar", str(erro), parent=self)
            return
        self.resultado = r
        self.app.ultima_exportacao = r
        texto = (f"Pronto: {len(r.treino)} imagens de treino, {len(r.val)} de validação, {r.objetos} objetos.\n"
                 f"{r.pasta}\nCada formato tem um README_TREINO.md com o comando de treino.")
        self.rotulo_info.configure(text=texto)
        self.app._mensagem(f"Exportado para treino em {r.pasta}")
        if messagebox.askyesno("Exportar", texto + "\n\nAbrir a pasta?", parent=self):
            abrir_no_sistema(r.pasta)
