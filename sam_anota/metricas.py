"""IoU contra máscara de referência e relatório da sessão.

Referência: arquivo <nome>_ref.png ao lado da imagem. Aceita dois formatos:
  * binária (0 e 255, ou 0 e 1): cada componente conexa vira uma instância;
  * rotulada (0 = fundo, 1, 2, 3... = um valor por objeto, ou PNG com paleta).
A IoU real de um objeto anotado é a melhor entre: a referência inteira e cada
instância. Assim um objeto que cobre a pluma toda e um objeto que cobre uma
ilha só são comparados com o alvo que faz sentido para cada um.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .projeto import SUFIXO_REFERENCIA

LIMIAR_DIFERENCA = 0.15  # |escore previsto - IoU real| acima disso ganha destaque


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Interseção sobre união de duas máscaras booleanas do mesmo tamanho.

    Duas máscaras vazias dão 1.0 (concordam perfeitamente em não ter nada).
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"tamanhos diferentes: {a.shape} e {b.shape}")
    uniao = np.logical_or(a, b).sum()
    if uniao == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / uniao)


def caminho_referencia(caminho_imagem: Path) -> Path | None:
    p = Path(caminho_imagem)
    ref = p.with_name(f"{p.stem}{SUFIXO_REFERENCIA}.png")
    return ref if ref.exists() else None


def carregar_referencia(caminho: Path, forma: tuple[int, int] | None = None) -> np.ndarray:
    """Devolve uma matriz inteira: 0 = fundo, 1..N = instâncias.

    Se forma=(altura, largura) for dada e diferente, redimensiona por vizinho mais próximo.
    """
    im = Image.open(caminho)
    rotulada = im.mode == "P"
    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=2)
        rotulada = False
    if rotulada:
        rotulos = arr.astype(np.int32)
    elif 1 < arr.max() < 128:
        rotulos = arr.astype(np.int32)  # valores pequenos: um por instância
    else:
        binaria = (arr > 0) if arr.max() <= 1 else (arr >= 128)
        _, rotulos = cv2.connectedComponents(binaria.astype(np.uint8), connectivity=8)
        rotulos = rotulos.astype(np.int32)
    if forma is not None and rotulos.shape != tuple(forma):
        im_r = Image.fromarray(rotulos.astype(np.uint16)).resize((forma[1], forma[0]), Image.NEAREST)
        rotulos = np.array(im_r).astype(np.int32)
    return rotulos


def iou_contra_referencia(mascara: np.ndarray, rotulos: np.ndarray) -> float:
    """Melhor IoU da máscara contra a referência inteira ou contra uma instância dela."""
    valores = [v for v in np.unique(rotulos) if v != 0]
    melhor = iou(mascara, rotulos > 0)
    for v in valores if len(valores) > 1 else []:
        melhor = max(melhor, iou(mascara, rotulos == v))
    return melhor


def iou_uniao(mascaras: list[np.ndarray], rotulos: np.ndarray) -> float | None:
    """IoU da união de todos os objetos da imagem contra a referência inteira."""
    if not mascaras:
        return None
    return iou(np.logical_or.reduce([np.asarray(m, bool) for m in mascaras]), rotulos > 0)


def diferenca_grande(escore: float, iou_real: float | None, limiar: float = LIMIAR_DIFERENCA) -> bool:
    return iou_real is not None and abs(float(escore) - float(iou_real)) >= limiar


# ---------------------------------------------------------------- relatório da sessão

CAMPOS = ["horario", "imagem", "id_objeto", "classe", "cliques", "tempo_s",
          "escore_previsto", "iou_real", "diferenca", "iou_uniao_imagem", "tempo_codificacao_s",
          "modelo", "dispositivo", "origem", "apagado"]


class RelatorioSessao:
    """CSV + resumo em Markdown, reescritos a cada objeto aceito."""

    def __init__(self, pasta: Path, dispositivo: str = "", modelo: str = ""):
        self.pasta = Path(pasta)
        self.inicio = datetime.now()
        carimbo = self.inicio.strftime("%Y%m%d_%H%M%S")
        self.caminho_csv = self.pasta / f"sessao_{carimbo}.csv"
        self.caminho_resumo = self.pasta / f"sessao_{carimbo}_resumo.md"
        self.dispositivo = dispositivo
        self.modelo = modelo
        self.registros: list[dict] = []
        self.codificacoes: list[float] = []
        self.predicoes: list[float] = []

    def adicionar(self, imagem: str, id_objeto: int, classe: str, cliques: int, tempo_s: float,
                  escore: float, iou_real: float | None, tempo_codificacao_s: float | None,
                  iou_uniao_imagem: float | None = None, origem: str = "") -> None:
        self.registros.append({
            "horario": datetime.now().isoformat(timespec="seconds"),
            "imagem": imagem, "id_objeto": id_objeto, "classe": classe, "cliques": cliques,
            "tempo_s": round(float(tempo_s), 2), "escore_previsto": round(float(escore), 4),
            "iou_real": None if iou_real is None else round(float(iou_real), 4),
            "diferenca": None if iou_real is None else round(float(escore) - float(iou_real), 4),
            "iou_uniao_imagem": None if iou_uniao_imagem is None else round(float(iou_uniao_imagem), 4),
            "tempo_codificacao_s": None if tempo_codificacao_s is None else round(tempo_codificacao_s, 3),
            "modelo": self.modelo, "dispositivo": self.dispositivo, "origem": origem,
            "apagado": False,
        })

    def marcar_apagado(self, imagem: str, id_objeto: int) -> None:
        for r in self.registros:
            if r["imagem"] == imagem and r["id_objeto"] == id_objeto:
                r["apagado"] = True

    def salvar(self) -> tuple[Path, Path] | None:
        if not self.registros and not self.codificacoes:
            return None
        self.pasta.mkdir(parents=True, exist_ok=True)
        with open(self.caminho_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS)
            w.writeheader()
            for r in self.registros:
                w.writerow({k: ("" if r[k] is None else r[k]) for k in CAMPOS})
        self.caminho_resumo.write_text(self.resumo(), encoding="utf-8")
        return self.caminho_csv, self.caminho_resumo

    def resumo(self) -> str:
        ativos = [r for r in self.registros if not r["apagado"]]
        com_ref = [r for r in ativos if r["iou_real"] is not None]
        linhas = [
            f"# Sessão de anotação {self.inicio.strftime('%d/%m/%Y %H:%M')}",
            "",
            f"Dispositivo: {self.dispositivo or 'n/d'}. Modelo: {self.modelo or 'n/d'}.",
            "",
            "| Medida | Valor |",
            "| --- | --- |",
            f"| Objetos aceitos | {len(ativos)} |",
            f"| Imagens com objeto | {len({r['imagem'] for r in ativos})} |",
        ]
        if ativos:
            linhas += [
                f"| Cliques por objeto (média) | {_fmt(np.mean([r['cliques'] for r in ativos]), 1)} |",
                f"| Tempo por objeto (média, s) | {_fmt(np.mean([r['tempo_s'] for r in ativos]), 1)} |",
                f"| Escore previsto (média) | {_fmt(np.mean([r['escore_previsto'] for r in ativos]), 3)} |",
            ]
        if self.codificacoes:
            linhas.append(f"| Codificação da imagem (mediana, s) | {_fmt(np.median(self.codificacoes), 2)} "
                          f"({len(self.codificacoes)} imagens) |")
        if self.predicoes:
            linhas.append(f"| Predição por clique (mediana, s) | {_fmt(np.median(self.predicoes), 3)} |")
        if com_ref:
            ious = np.array([r["iou_real"] for r in com_ref])
            esc = np.array([r["escore_previsto"] for r in com_ref])
            grandes = int((np.abs(esc - ious) >= LIMIAR_DIFERENCA).sum())
            linhas += [
                f"| Objetos com referência | {len(com_ref)} |",
                f"| IoU real (média) | {_fmt(ious.mean(), 3)} |",
                f"| Escore previsto nesses objetos (média) | {_fmt(esc.mean(), 3)} |",
                f"| Diferença escore menos IoU (média) | {_fmt((esc - ious).mean(), 3)} |",
                f"| Objetos com diferença de {_fmt(LIMIAR_DIFERENCA, 2)} ou mais | {grandes} |",
            ]
            linhas += ["", "## Objetos com referência", "",
                       "| Imagem | Classe | Cliques | Tempo (s) | Escore previsto | IoU real |",
                       "| --- | --- | --- | --- | --- | --- |"]
            for r in com_ref:
                linhas.append(f"| {r['imagem']} | {r['classe']} | {r['cliques']} | {_fmt(r['tempo_s'], 1)} | "
                              f"{_fmt(r['escore_previsto'], 3)} | {_fmt(r['iou_real'], 3)} |")
        linhas += ["", f"Dados completos: `{self.caminho_csv.name}`.", ""]
        return "\n".join(linhas)


def _fmt(x: float, casas: int) -> str:
    """Número com vírgula decimal."""
    return f"{float(x):.{casas}f}".replace(".", ",")
