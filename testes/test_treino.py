import json

import numpy as np
import pytest
from PIL import Image

from conftest import retangulo
from sam_anota.exportar import (OpcoesTreino, dividir, exportar_treino, juntar_pedacos, linha_yolo,
                                mascara_para_poligonos, nome_plano, segmentacao_coco, selecionar_imagens, tem_furo)
from sam_anota.projeto import Projeto


def projeto_treino(tmp_path, n=6):
    pasta = tmp_path / "imgs"
    (pasta / "sub").mkdir(parents=True)
    nomes = [f"img{i}.png" for i in range(n - 1)] + ["sub/vazia.png"]
    for nome in nomes:
        Image.new("RGB", (80, 60), (40, 40, 40)).save(pasta / nome)
    p = Projeto.abrir_ou_criar(pasta)
    p.classes = ["carro", "pessoa"]
    for i, nome in enumerate(nomes[:-1]):
        p.adicionar_objeto(nome, retangulo(x0=5 + i, x1=30 + i), "carro", 0.9, 0, 1, 1.0)
        p.adicionar_objeto(nome, retangulo(x0=20, x1=60, y0=30, y1=55), "pessoa", 0.8, 0, 1, 1.0)
    p.imagens["sub/vazia.png"].concluida = True
    p.imagens["sub/vazia.png"].largura, p.imagens["sub/vazia.png"].altura = 80, 60
    return p


# ---------------------------------------------------------------- geometria

def test_juntar_pedacos_cobre_os_dois():
    m = retangulo(x0=2, x1=20, y0=2, y1=20) | retangulo(x0=50, x1=70, y0=30, y1=50)
    polis = mascara_para_poligonos(m)
    assert len(polis) == 2
    junto = np.array(juntar_pedacos(polis)).reshape(-1, 2)
    assert len(junto) == sum(len(p) // 2 for p in polis) + 2  # dois pontos repetidos na ponte
    raster = np.zeros_like(m, dtype=np.uint8)
    import cv2
    cv2.fillPoly(raster, [np.round(junto).astype(np.int32)], 1)
    assert raster[10, 10] and raster[40, 60]  # os dois pedaços estão no polígono único


def test_linha_yolo_um_objeto_uma_linha():
    m = retangulo(x0=2, x1=20) | retangulo(x0=50, x1=70)
    linha = linha_yolo(m, 3, 80, 60)
    assert linha.startswith("3 ") and "\n" not in linha
    v = [float(x) for x in linha.split()[1:]]
    assert len(v) % 2 == 0 and all(0 <= x <= 1 for x in v)
    assert max(v[0::2]) > 0.8 and min(v[0::2]) < 0.1


def test_furo_vira_rle_e_poligono_simples_nao():
    pytest.importorskip("pycocotools")
    from pycocotools import mask as mascara_coco

    anel = retangulo(x0=10, x1=60, y0=10, y1=50)
    anel[20:40, 25:45] = False
    assert tem_furo(anel) and not tem_furo(retangulo())
    seg = segmentacao_coco(anel)
    assert isinstance(seg, dict) and isinstance(seg["counts"], str)
    rle = {"size": seg["size"], "counts": seg["counts"].encode("ascii")}
    assert np.array_equal(mascara_coco.decode(rle).astype(bool), anel)
    assert isinstance(segmentacao_coco(retangulo()), list)


def test_objeto_fino_vira_rle():
    pytest.importorskip("pycocotools")
    fino = np.zeros((60, 80), bool)
    fino[30, 5:75] = True  # trinca de 1 pixel
    fino[31, 5:75] = True
    assert isinstance(segmentacao_coco(fino), dict)


# ---------------------------------------------------------------- seleção e divisão

def test_divisao_reprodutivel_e_disjunta():
    rels = [f"{i}.png" for i in range(10)]
    t1, v1 = dividir(rels, 0.2, 42)
    t2, v2 = dividir(rels, 0.2, 42)
    assert (t1, v1) == (t2, v2) and len(v1) == 2 and not set(t1) & set(v1)
    assert set(t1) | set(v1) == set(rels)
    assert dividir(rels, 0.2, 7) != (t1, v1)
    assert len(dividir(["a", "b"], 0.2, 1)[1]) == 1  # com 2 imagens, ao menos 1 em val


def test_selecao_de_imagens(tmp_path):
    p = projeto_treino(tmp_path)
    assert "sub/vazia.png" not in selecionar_imagens(p, OpcoesTreino())
    assert "sub/vazia.png" in selecionar_imagens(p, OpcoesTreino(incluir_vazias=True))
    assert selecionar_imagens(p, OpcoesTreino(so_concluidas=True)) == []
    p.imagens["img0.png"].concluida = True
    assert selecionar_imagens(p, OpcoesTreino(so_concluidas=True, incluir_vazias=True)) == ["img0.png", "sub/vazia.png"]


# ---------------------------------------------------------------- os três formatos

def test_exportar_treino_tres_formatos(tmp_path):
    p = projeto_treino(tmp_path)
    r = exportar_treino(p, tmp_path / "out", OpcoesTreino(frac_val=0.34, semente=1, incluir_vazias=True))
    assert len(r.treino) + len(r.val) == 6 and len(r.val) == 2
    raiz = tmp_path / "out"

    # YOLO
    y = raiz / "yolo"
    dados = (y / "data.yaml").read_text(encoding="utf-8")
    assert "path: ." in dados and "train: images/train" in dados and "val: images/val" in dados
    assert dados.index("0: carro") < dados.index("1: pessoa")
    for split, lista in (("train", r.treino), ("val", r.val)):
        for rel in lista:
            assert (y / "images" / split / nome_plano(rel)).exists()
            txt = (y / "labels" / split / nome_plano(rel).replace(".png", ".txt")).read_text()
            assert len([l for l in txt.splitlines() if l]) == len(p.imagens[rel].objetos)
    assert (y / "README_TREINO.md").exists()

    # COCO
    c = raiz / "coco"
    total = 0
    for split, lista in (("train", r.treino), ("val", r.val)):
        d = json.loads((c / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
        assert [cat["id"] for cat in d["categories"]] == [1, 2]
        assert [cat["name"] for cat in d["categories"]] == ["carro", "pessoa"]
        assert len(d["images"]) == len(lista)
        for a in d["annotations"]:
            assert a["iscrowd"] == 0 and len(a["bbox"]) == 4 and a["area"] > 0
            assert (c / "images" / split / next(i["file_name"] for i in d["images"] if i["id"] == a["image_id"])).exists()
        total += len(d["annotations"])
    assert total == p.total_objetos()
    assert (c / "treinar_torchvision.py").exists() and (c / "README_TREINO.md").exists()

    # semântica: valor = índice da classe, 0 fundo, o último aceito vence
    s = raiz / "semantica"
    info = json.loads((s / "classes.json").read_text(encoding="utf-8"))
    assert [i["nome"] for i in info] == ["fundo", "carro", "pessoa"]
    rel = (r.treino + r.val)[0] if (r.treino + r.val)[0] != "sub/vazia.png" else (r.treino + r.val)[1]
    split = "train" if rel in r.treino else "val"
    m = np.array(Image.open(s / "masks" / split / nome_plano(rel).replace(".png", ".png")))
    assert m.dtype == np.uint8 and m.shape == (60, 80)
    assert set(np.unique(m)) == {0, 1, 2}
    assert m[40, 25] == 2  # sobreposição carro e pessoa: pessoa foi aceita depois
    assert m[25, 12] == 1
    assert (s / "visualizacao" / split / nome_plano(rel)).exists()
    assert (s / "treinar_unet.py").exists()
    vazia = "train" if "sub/vazia.png" in r.treino else "val"
    assert np.array(Image.open(s / "masks" / vazia / "sub__vazia.png")).max() == 0

    div = json.loads((raiz / "divisao.json").read_text(encoding="utf-8"))
    assert div["semente"] == 1 and sorted(div["treino"] + div["val"]) == sorted(r.treino + r.val)


def test_exportar_sem_imagens_falha_claro(tmp_path):
    pasta = tmp_path / "v"
    pasta.mkdir()
    Image.new("RGB", (10, 10)).save(pasta / "a.png")
    with pytest.raises(ValueError):
        exportar_treino(Projeto.abrir_ou_criar(pasta), tmp_path / "o")
