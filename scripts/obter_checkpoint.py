"""Baixa pesos oficiais do SAM (Meta) para checkpoints/ e confere o SHA-256.

Uso:
    python scripts/obter_checkpoint.py                         # SAM 2.1 tiny (padrão do app)
    python scripts/obter_checkpoint.py --modelo sam2.1_large
    python scripts/obter_checkpoint.py --modelo sam1_vit_b,sam1_vit_h
    python scripts/obter_checkpoint.py --modelo todos
    python scripts/obter_checkpoint.py --listar

Arquivo que já existe com o tamanho certo é só conferido (hash), não baixado de novo.
URLs, tamanhos e hashes ficam em sam_anota/modelo.py (CATALOGO).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from sam_anota.modelo import CATALOGO, ErroModelo, baixar_checkpoint, resolver_chave, sha256_arquivo  # noqa: E402

_ultimo = [-1]


def _progresso(feito: int, total: int) -> None:
    pct = int(100 * feito / max(total, 1)) // 5 * 5
    if pct != _ultimo[0]:  # imprime de 5 em 5%
        _ultimo[0] = pct
        print(f"\r  {pct:3d}% de {total / 1e6:.0f} MB", end="", flush=True)


def obter(chave: str) -> bool:
    info = CATALOGO[chave]
    destino = info.caminho
    if info.baixado:
        print(f"{info.rotulo}: já existe, conferindo o SHA-256...")
    else:
        print(f"{info.rotulo}: baixando {info.url} ({info.tamanho_bytes / 1e6:.0f} MB)")
        _ultimo[0] = -1
        try:
            baixar_checkpoint(info, _progresso)
        except ErroModelo as e:
            print(f"\nERRO: {e}")
            return False
        print()
    h = sha256_arquivo(destino)
    if h != info.sha256:
        print(f"ATENÇÃO: {destino.name} com hash {h}, esperado {info.sha256}. Apague o arquivo e rode de novo.")
        return False
    print(f"  {destino}\n  SHA-256 ok: {h}")
    return True


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="sam2.1_tiny", help="chave(s) separadas por vírgula, ou 'todos'")
    ap.add_argument("--listar", action="store_true")
    a = ap.parse_args()
    if a.listar:
        for c, i in CATALOGO.items():
            print(f"{c:18s} {i.rotulo_combo:34s} {'baixado' if i.baixado else '-'}")
        return 0
    chaves = list(CATALOGO) if a.modelo == "todos" else [resolver_chave(c.strip()) for c in a.modelo.split(",")]
    ok = all([obter(c) for c in chaves])
    print("Checkpoint pronto." if ok else "Houve falha.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
