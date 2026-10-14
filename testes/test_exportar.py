import json

import numpy as np
from PIL import Image

from conftest import retangulo
from sam_anota.exportar import (bbox_xywh, exportar_coco, exportar_mascaras_png, exportar_tudo,
                                exportar_yolo, mascara_para_poligonos)
from sam_anota.projeto import Projeto


def projeto_sintetico(tmp_path):
    pasta = tmp_path / "imgs"
    (pasta / "sub").mkdir(parents=True)
    Image.new("RGB", (80, 60)).save(pasta / "a.png")
    Image.new("RGB", (80, 60)).save(pasta / "sub" / "b.png")
    p = Projeto.abrir_ou_criar(pasta)
    p.adicionar_objeto("a.png", retangulo(), "carro", 0.9, 2, 3, 5.0)
    disco = np.zeros((60, 80), bool)
    yy, xx = np.mgrid[:60, :80]
    disco[(xx - 55) ** 2 + (yy - 30) ** 2 <= 10 ** 2] = True
    p.adicionar_objeto("a.png", disco, "pessoa", 0.8, 0, 1, 2.0)
    p.adicionar_objeto("sub/b.png", retangulo(x0=0, x1=80, y0=0, y1=60), "carro", 0.99, 0, 1, 1.0)
    return p


def test_poligono_do_retangulo():
    polis = mascara_para_poligonos(retangulo())
    assert len(polis) == 1
    xy = np.array(polis[0]).reshape(-1, 2)
    assert xy[:, 0].min() == 10 and xy[:, 0].max() == 39
    assert xy[:, 1].min() == 20 and xy[:, 1].max() == 49


def test_dois_pedacos_viram_dois_poligonos_maior_primeiro():
    m = retangulo(x0=0, x1=10, y0=0, y1=10) | retangulo(x0=30, x1=70, y0=20, y1=50)
    polis = mascara_para_poligonos(m)
    assert len(polis) == 2
    assert min(np.array(polis[0]).reshape(-1, 2)[:, 0]) == 30


def test_bbox_xywh():
    assert bbox_xywh(retangulo()) == [10.0, 20.0, 30.0, 30.0]
    assert bbox_xywh(np.zeros((5, 5), bool)) == [0.0, 0.0, 0.0, 0.0]


def test_coco(tmp_path):
    p = projeto_sintetico(tmp_path)
    caminho = exportar_coco(p, tmp_path / "out" / "coco.json")
    c = json.loads(caminho.read_text(encoding="utf-8"))
    assert [i["file_name"] for i in c["images"]] == ["a.png", "sub/b.png"]
    assert c["images"][0]["width"] == 80 and c["images"][0]["height"] == 60
    nomes = {cat["id"]: cat["name"] for cat in c["categories"]}
    assert len(c["annotations"]) == 3
    ret = c["annotations"][0]
    assert nomes[ret["category_id"]] == "carro"
    assert ret["area"] == 900.0  # 30 x 30 pixels
    assert ret["bbox"] == [10.0, 20.0, 30.0, 30.0]
    assert ret["iscrowd"] == 0 and len(ret["segmentation"][0]) >= 8
    disco = c["annotations"][1]
    assert nomes[disco["category_id"]] == "pessoa"
    assert abs(disco["area"] - np.pi * 100) < 40
    assert {a["image_id"] for a in c["annotations"]} == {1, 2}
    assert len({a["id"] for a in c["annotations"]}) == 3


def test_yolo(tmp_path):
    p = projeto_sintetico(tmp_path)
    exportar_yolo(p, tmp_path / "yolo")
    linhas = (tmp_path / "yolo" / "labels" / "a.txt").read_text().strip().splitlines()
    assert len(linhas) == 2
    campos = linhas[0].split()
    assert campos[0] == str(p.classes.index("carro"))
    xy = np.array([float(v) for v in campos[1:]]).reshape(-1, 2)
    assert np.isclose(xy[:, 0].min(), 10 / 80) and np.isclose(xy[:, 0].max(), 39 / 80)
    assert np.isclose(xy[:, 1].min(), 20 / 60) and np.isclose(xy[:, 1].max(), 49 / 60)
    assert (tmp_path / "yolo" / "labels" / "sub" / "b.txt").exists()
    classes = (tmp_path / "yolo" / "classes.txt").read_text().split()
    assert classes == p.classes
    assert "names:" in (tmp_path / "yolo" / "data.yaml").read_text()


def test_mascaras_png(tmp_path):
    p = projeto_sintetico(tmp_path)
    arquivos = exportar_mascaras_png(p, tmp_path / "m")
    assert len(arquivos) == 3
    m = np.array(Image.open(arquivos[0]))
    assert set(np.unique(m)) == {0, 255} and (m > 0).sum() == 900


def test_exportar_tudo_vai_para_saidas(tmp_path):
    p = projeto_sintetico(tmp_path)
    r = exportar_tudo(p)
    assert r["coco"].exists() and str(tmp_path) in str(r["pasta"])
