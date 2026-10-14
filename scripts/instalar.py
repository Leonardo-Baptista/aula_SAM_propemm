"""Instala tudo o que o aplicativo precisa, no Python que rodar este script.

Uso (de dentro da pasta do repositório, com o venv já criado):
    Windows:      .venv\\Scripts\\python scripts\\instalar.py
    Linux/macOS:  .venv/bin/python scripts/instalar.py

Opções:
    --gpu      força torch com CUDA 12.4 (padrão: só se o nvidia-smi responder)
    --cpu      força torch só CPU (menor download, cerca de 200 MB)
    --sam1     requirements-sam1.txt (segment-anything) e baixa o ViT-B (375 MB)
    --texto    requirements-texto.txt (transformers; o Grounding DINO tiny, 690 MB, baixa no primeiro uso)
    --notebooks  requirements-notebooks.txt (jupyter, matplotlib, segment-anything)
    --opcionais  os três acima
    --sem-opcionais    não pergunta pelos opcionais
    --sem-checkpoint   não baixa os pesos
Sem nenhuma dessas opções, num terminal interativo, o script pergunta.

Passos (os mesmos da instalação manual de docs/INSTALACAO.md):
  1. torch e torchvision, versões fixas, do índice oficial do PyTorch (CPU ou CUDA 12.4);
  2. requirements.txt (customtkinter, pillow, numpy, opencv, pycocotools, pytest, setuptools);
  3. sam2 do GitHub oficial, commit fixo, com SAM2_BUILD_CUDA=0 e --no-build-isolation
     (sem compilar extensão C, o que no Windows exigiria Visual Studio e CUDA toolkit);
  4. checkpoint sam2.1_hiera_tiny.pt com conferência de SHA-256;
  5. teste rápido: importa o sam2 e carrega o modelo.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAM2_GIT = "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
INDICE_CPU = "https://download.pytorch.org/whl/cpu"
INDICE_CUDA = "https://download.pytorch.org/whl/cu124"
# versões testadas em 14/10/2026: CPU no Windows com 2.14.1; CUDA 12.4 na RTX 3060 com 2.6.0
# (2.6.0 é a última publicada para cu124)
TORCH_CPU = ["torch==2.14.1", "torchvision==0.29.1"]
TORCH_CUDA = ["torch==2.6.0", "torchvision==0.21.0"]


def rodar(cmd: list[str], env=None) -> None:
    print("\n>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(f"Falhou (código {r.returncode}): {' '.join(cmd)}")


def perguntar(texto: str) -> bool:
    try:
        return input(f"{texto} [s/N] ").strip().lower() in ("s", "sim", "y", "yes")
    except EOFError:
        return False


def tem_nvidia() -> bool:
    if platform.system() == "Darwin" or shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true")
    g.add_argument("--cpu", action="store_true")
    ap.add_argument("--sem-checkpoint", action="store_true")
    ap.add_argument("--sam1", action="store_true")
    ap.add_argument("--texto", action="store_true")
    ap.add_argument("--notebooks", action="store_true")
    ap.add_argument("--opcionais", action="store_true")
    ap.add_argument("--sem-opcionais", action="store_true")
    a = ap.parse_args()
    if a.opcionais:
        a.sam1 = a.texto = a.notebooks = True
    if not (a.sam1 or a.texto or a.notebooks or a.sem_opcionais) and sys.stdin.isatty():
        a.sam1 = perguntar("Instalar também o SAM 1 (ViT-B, ViT-L, ViT-H)? Baixa 375 MB a mais")
        a.texto = perguntar("Instalar o modo texto (Grounding DINO + SAM; SAM 3 se você tiver acesso)?")
        a.notebooks = perguntar("Instalar o necessário para rodar os notebooks localmente (jupyter)?")

    if sys.version_info < (3, 10):
        sys.exit("Use Python 3.10 ou mais novo (testado com 3.12).")
    if sys.version_info[:2] != (3, 12):
        print(f"AVISO: Python {platform.python_version()}; o material foi testado com 3.12.")
    if sys.prefix == sys.base_prefix:
        print("AVISO: não parece um venv. Recomendo: py -3.12 -m venv .venv (Windows) ou python3 -m venv .venv")
    if shutil.which("git") is None:
        sys.exit("O git não está no PATH; ele é necessário para instalar o sam2 do GitHub.")

    py = [sys.executable, "-m", "pip"]
    rodar(py + ["install", "--upgrade", "pip"])

    usar_gpu = a.gpu or (not a.cpu and tem_nvidia())
    if platform.system() == "Darwin":
        rodar(py + ["install", "torch", "torchvision"])
    else:
        indice = INDICE_CUDA if usar_gpu else INDICE_CPU
        print(f"\nTorch: {'CUDA 12.4' if usar_gpu else 'só CPU'} ({indice})")
        rodar(py + ["install", "torch", "torchvision", "--index-url", indice])

    rodar(py + ["install", "-r", str(RAIZ / "requirements.txt")])

    env = dict(os.environ, SAM2_BUILD_CUDA="0")
    rodar(py + ["install", "--no-build-isolation", SAM2_GIT], env=env)

    for flag, arquivo in ((a.sam1, "requirements-sam1.txt"), (a.texto, "requirements-texto.txt"),
                          (a.notebooks, "requirements-notebooks.txt")):
        if flag:
            rodar(py + ["install", "-r", str(RAIZ / arquivo)])

    if not a.sem_checkpoint:
        modelos = "sam2.1_tiny" + (",sam1_vit_b" if a.sam1 else "")
        rodar([sys.executable, str(RAIZ / "scripts" / "obter_checkpoint.py"), "--modelo", modelos])

    rodar([sys.executable, str(RAIZ / "scripts" / "verificar.py")])
    print("\nInstalação concluída. Para abrir: python -m sam_anota")
    return 0


if __name__ == "__main__":
    sys.exit(main())
