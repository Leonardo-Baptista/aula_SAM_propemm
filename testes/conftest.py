import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def saidas_temporarias(tmp_path, monkeypatch):
    """Nenhum teste escreve em saidas/ do repositório."""
    import sam_anota.projeto as projeto

    monkeypatch.setattr(projeto, "PASTA_SAIDAS", tmp_path / "saidas")


def retangulo(altura=60, largura=80, x0=10, y0=20, x1=40, y1=50):
    """Máscara sintética: retângulo cheio de [x0, x1) por [y0, y1)."""
    m = np.zeros((altura, largura), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m
