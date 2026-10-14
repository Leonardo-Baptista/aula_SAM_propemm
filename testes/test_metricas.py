import numpy as np
import pytest
from PIL import Image

from conftest import retangulo
from sam_anota.metricas import (RelatorioSessao, caminho_referencia, carregar_referencia,
                                diferenca_grande, iou, iou_contra_referencia)


def test_iou_identica_e_disjunta():
    a = retangulo()
    assert iou(a, a) == 1.0
    b = retangulo(x0=50, x1=70)
    assert iou(a, b) == 0.0


def test_iou_meia_sobreposicao():
    a = retangulo(x0=0, x1=20, y0=0, y1=10)  # 200 px
    b = retangulo(x0=10, x1=30, y0=0, y1=10)  # 200 px, 100 em comum
    assert abs(iou(a, b) - 100 / 300) < 1e-12


def test_iou_vazias_concordam():
    z = np.zeros((5, 5), bool)
    assert iou(z, z) == 1.0


def test_iou_tamanhos_diferentes_falha():
    with pytest.raises(ValueError):
        iou(np.zeros((3, 3)), np.zeros((4, 4)))


def test_referencia_binaria_255(tmp_path):
    (tmp_path / "foto.jpg").write_bytes(b"")
    ref = retangulo().astype(np.uint8) * 255
    Image.fromarray(ref).save(tmp_path / "foto_ref.png")
    caminho = caminho_referencia(tmp_path / "foto.jpg")
    assert caminho is not None and caminho.name == "foto_ref.png"
    rot = carregar_referencia(caminho)
    assert set(np.unique(rot)) == {0, 1}  # uma componente só
    assert iou_contra_referencia(retangulo(), rot) == 1.0


def test_referencia_rotulada_escolhe_melhor_instancia(tmp_path):
    rot = np.zeros((60, 80), np.uint8)
    rot[retangulo(x0=0, x1=20)] = 1
    rot[retangulo(x0=50, x1=70)] = 2
    Image.fromarray(rot).save(tmp_path / "x_ref.png")
    r = carregar_referencia(tmp_path / "x_ref.png")
    assert iou_contra_referencia(retangulo(x0=50, x1=70), r) == 1.0


def test_referencia_redimensionada(tmp_path):
    ref = retangulo(altura=30, largura=40, x0=5, x1=20, y0=10, y1=25).astype(np.uint8) * 255
    Image.fromarray(ref).save(tmp_path / "y_ref.png")
    r = carregar_referencia(tmp_path / "y_ref.png", forma=(60, 80))
    assert r.shape == (60, 80) and r.sum() == 4 * ref.astype(bool).sum()


def test_sem_referencia(tmp_path):
    assert caminho_referencia(tmp_path / "nada.jpg") is None


def test_diferenca_grande():
    assert diferenca_grande(0.95, 0.70)
    assert not diferenca_grande(0.95, 0.90)
    assert not diferenca_grande(0.95, None)


def test_relatorio_csv_e_resumo(tmp_path):
    r = RelatorioSessao(tmp_path, "cpu", "SAM 2.1 tiny")
    r.codificacoes.append(1.5)
    r.predicoes += [0.08, 0.09]
    r.adicionar("a.jpg", 1, "gato", 3, 12.0, 0.95, 0.70, 1.5)
    r.adicionar("b.jpg", 2, "cao", 1, 4.0, 0.90, None, 1.4)
    r.adicionar("b.jpg", 3, "cao", 2, 5.0, 0.80, None, 1.4)
    r.marcar_apagado("b.jpg", 3)
    csv_, resumo = r.salvar()
    linhas = csv_.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 4 and linhas[0].startswith("horario,imagem")
    texto = resumo.read_text(encoding="utf-8")
    assert "| Objetos aceitos | 2 |" in texto
    assert "IoU real (média) | 0,700" in texto
    assert "Objetos com diferença de 0,15 ou mais | 1 |" in texto


def test_referencia_binaria_com_ilhas(tmp_path):
    """Máscara binária com duas ilhas: um objeto numa ilha tem IoU 1 com ela;
    a união das duas tem IoU 1 com a referência inteira."""
    from sam_anota.metricas import iou_uniao

    ilha_a = retangulo(x0=0, x1=20)
    ilha_b = retangulo(x0=50, x1=70)
    Image.fromarray((ilha_a | ilha_b).astype(np.uint8) * 255).save(tmp_path / "z_ref.png")
    rot = carregar_referencia(tmp_path / "z_ref.png")
    assert rot.max() == 2
    assert iou_contra_referencia(ilha_a, rot) == 1.0
    assert iou_contra_referencia(ilha_a | ilha_b, rot) == 1.0
    assert iou_uniao([ilha_a, ilha_b], rot) == 1.0
    assert iou_uniao([ilha_a], rot) == 0.5
    assert iou_uniao([], rot) is None
