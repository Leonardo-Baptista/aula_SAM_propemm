"""Mede carga, codificação e predição de cada modelo baixado, no dispositivo pedido.

    python scripts/medir_modelos.py --dispositivo cpu
    python scripts/medir_modelos.py --dispositivo cuda --texto     # inclui o Grounding DINO

Imagem: dados/exemplos/cao_e_gato.jpg (1600x1200). Codificação: 3 vezes sem cache;
a primeira é mostrada à parte (inclui aquecimento) e as demais viram a mediana.
Predição: mediana de 5, com 2 pontos. Sai uma tabela em Markdown e um JSON.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")


def _v(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispositivo", default="cpu")
    ap.add_argument("--modelos", default="")
    ap.add_argument("--texto", action="store_true")
    ap.add_argument("--saida", type=Path, default=None)
    a = ap.parse_args()
    from sam_anota.modelo import CATALOGO, ModeloSAM, carregar_imagem_rgb, descrever_dispositivo, familia_instalada

    img = carregar_imagem_rgb(RAIZ / "dados" / "exemplos" / "cao_e_gato.jpg")
    chaves = [c for c in a.modelos.split(",") if c] or list(CATALOGO)
    linhas, dados = [], []
    disp_desc = None
    for chave in chaves:
        info = CATALOGO[chave]
        if not info.baixado or not familia_instalada(info.familia):
            print(f"pulando {info.rotulo}")
            continue
        m = ModeloSAM(chave, dispositivo=a.dispositivo, cache_disco=False)
        carga = m.carregar()
        disp_desc = descrever_dispositivo(m.dispositivo)
        cod = [m.codificar(f"k{i}", img).tempo_s for i in range(3)]
        pred = [m.prever("k0", [(420, 650, 1), (800, 450, 1)]).tempo_s for _ in range(5)]
        r = {"modelo": info.rotulo, "dispositivo": m.dispositivo, "carga_s": carga, "codificacao_primeira_s": cod[0],
             "codificacao_mediana_s": statistics.median(cod[1:]), "predicao_mediana_s": statistics.median(pred)}
        dados.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        linhas.append(f"| {info.rotulo} | {m.dispositivo.upper()} | {_v(carga, 1)} | {_v(cod[0])} | "
                      f"{_v(r['codificacao_mediana_s'])} | {_v(r['predicao_mediana_s'], 3)} |")
        m.liberar()
        del m
    if a.texto:
        from sam_anota.texto import DetectorGdino
        disp = "cuda" if a.dispositivo == "cuda" else "cpu"
        d = DetectorGdino(disp)
        carga = d.carregar()
        tempos = [d.detectar(img, t)[1] for t in ("cat", "dog", "cat", "dog")]
        r = {"modelo": "Grounding DINO tiny", "dispositivo": disp, "carga_s": carga,
             "deteccao_primeira_s": tempos[0], "deteccao_mediana_s": statistics.median(tempos[1:])}
        dados.append(r)
        print(json.dumps(r, ensure_ascii=False))
        linhas.append(f"| Grounding DINO tiny (texto para caixas) | {disp.upper()} | {_v(carga, 1)} | {_v(tempos[0])} | "
                      f"{_v(r['deteccao_mediana_s'])} | n/a |")
    print(f"\n{disp_desc}\n")
    print("| Modelo | Dispositivo | Carga (s) | 1ª codificação (s) | Codificação, mediana (s) | Predição (s) |")
    print("| --- | --- | --- | --- | --- | --- |")
    print("\n".join(linhas))
    if a.saida:
        a.saida.write_text(json.dumps({"ambiente": disp_desc, "medidas": dados}, ensure_ascii=False, indent=1),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
