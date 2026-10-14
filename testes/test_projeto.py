import json

import numpy as np
from PIL import Image

from conftest import retangulo
from sam_anota.projeto import Projeto, mascara_para_rle, rle_para_mascara


def test_rle_ida_e_volta():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.random((17, 23)) > 0.6
        assert np.array_equal(rle_para_mascara(mascara_para_rle(m)), m)


def test_rle_comeca_com_1_e_vazia():
    m = np.ones((3, 4), bool)
    rle = mascara_para_rle(m)
    assert rle["contagens"] == [0, 12]
    assert np.array_equal(rle_para_mascara(rle), m)
    z = np.zeros((3, 4), bool)
    assert np.array_equal(rle_para_mascara(mascara_para_rle(z)), z)


def pasta_com_imagens(tmp_path):
    pasta = tmp_path / "imgs"
    (pasta / "faceis").mkdir(parents=True)
    (pasta / "dificeis").mkdir()
    for nome in ("faceis/um.jpg", "dificeis/dois.png", "dificeis/dois_ref.png"):
        Image.new("RGB", (80, 60)).save(pasta / nome)
    (pasta / "leia.txt").write_text("x")
    return pasta


def test_lista_ignora_referencias_e_nao_imagens(tmp_path):
    pasta = pasta_com_imagens(tmp_path)
    assert Projeto.listar_imagens(pasta) == ["dificeis/dois.png", "faceis/um.jpg"]


def test_salvar_e_carregar(tmp_path):
    pasta = pasta_com_imagens(tmp_path)
    p = Projeto.abrir_ou_criar(pasta)
    p.classes.append("chapa")
    o = p.adicionar_objeto("faceis/um.jpg", retangulo(), "chapa", 0.91, 1, 4, 7.5,
                           pontos=[(20, 30, 1), (5, 5, 0)], caixa=(1, 2, 3, 4), iou_real=0.88)
    p.imagens["faceis/um.jpg"].concluida = True
    caminho = p.salvar()
    assert caminho.name == "projeto.json" and str(tmp_path / "saidas") in str(caminho)
    assert not list(pasta.rglob("*.json")), "nada pode ser escrito na pasta de imagens"
    d = json.loads(caminho.read_text(encoding="utf-8"))
    assert d["formato"] == "sam-anota"

    q = Projeto.abrir_ou_criar(pasta)  # reabre do disco
    assert q.classes == p.classes and q.proximo_id == 2
    img = q.imagens["faceis/um.jpg"]
    assert img.concluida and img.largura == 80 and img.altura == 60
    o2 = img.objetos[0]
    assert (o2.id, o2.classe, o2.cliques, o2.iou_real, o2.caixa) == (o.id, "chapa", 4, 0.88, [1.0, 2.0, 3.0, 4.0])
    assert o2.pontos[1] == [5.0, 5.0, 0.0]
    assert np.array_equal(o2.mascara(), retangulo()) and o2.area == 900


def test_remover_e_classe_em_uso(tmp_path):
    p = Projeto.abrir_ou_criar(pasta_com_imagens(tmp_path))
    o = p.adicionar_objeto("faceis/um.jpg", retangulo(), "gato", 0.9, 0, 1, 1.0)
    assert p.classe_em_uso("gato") and p.total_objetos() == 1
    assert p.remover_objeto("faceis/um.jpg", o.id)
    assert not p.classe_em_uso("gato") and p.total_objetos() == 0
    assert not p.remover_objeto("faceis/um.jpg", 999)


def test_imagem_nova_aparece_ao_sincronizar(tmp_path):
    pasta = pasta_com_imagens(tmp_path)
    p = Projeto.abrir_ou_criar(pasta)
    Image.new("RGB", (10, 10)).save(pasta / "faceis" / "tres.png")
    assert "faceis/tres.png" in p.sincronizar()
