"""Confere o ambiente e mede os tempos do SAM nesta máquina, sem abrir janela.

    python scripts/verificar.py

Saída esperada termina em "Ambiente ok."
"""
from __future__ import annotations

import platform
import sys
import time
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"Python {platform.python_version()} em {platform.system()} {platform.release()}")
    falhas = []
    for nome in ("numpy", "PIL", "cv2", "customtkinter", "torch", "sam2"):
        try:
            m = __import__(nome)
            print(f"  {nome:14s} {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            print(f"  {nome:14s} FALTA ({e})")
            falhas.append(nome)
    try:
        import tkinter
        tkinter.Tcl()
        print(f"  {'tkinter':14s} Tk {tkinter.TkVersion}")
    except Exception as e:
        print(f"  tkinter        FALTA ({e}); no Linux: sudo apt install python3-tk")
        falhas.append("tkinter")
    import importlib.util
    for nome, uso in (("segment_anything", "SAM 1, opcional"), ("transformers", "modo texto, opcional"),
                      ("pycocotools", "COCO com RLE")):
        tem = importlib.util.find_spec(nome) is not None
        print(f"  {nome:14s} {'ok' if tem else 'ausente'} ({uso})")
    if falhas:
        print("Faltam pacotes:", ", ".join(falhas), "(veja o README)")
        return 1

    from sam_anota.modelo import ErroModelo, ModeloSAM, carregar_imagem_rgb

    m = ModeloSAM(cache_disco=False)
    try:
        t = m.carregar()
    except ErroModelo as e:
        print(e)
        return 1
    print(f"Modelo: {m.descricao()}, carga {t:.2f} s")
    img = carregar_imagem_rgb(RAIZ / "dados" / "exemplos" / "cao_e_gato.jpg")
    e = m.codificar("verificacao", img)
    print(f"Codificação de {img.shape[1]}x{img.shape[0]}: {e.tempo_s:.2f} s")
    tempos = []
    for _ in range(5):
        p = m.prever("verificacao", [(420, 650, 1), (800, 450, 1)])
        tempos.append(p.tempo_s)
    print(f"Predição (5 vezes): {min(tempos):.3f} a {max(tempos):.3f} s; escores {[round(float(s), 3) for s in p.escores]}")
    if not p.mascaras.any():
        print("Máscaras vazias: algo errado no modelo.")
        return 1
    print("Ambiente ok.")
    return 0


if __name__ == "__main__":
    t0 = time.perf_counter()
    r = main()
    print(f"({time.perf_counter() - t0:.1f} s no total)")
    sys.exit(r)
