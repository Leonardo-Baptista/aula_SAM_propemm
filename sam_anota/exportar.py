"""Exportação das anotações.

Duas camadas:
  * funções básicas (polígonos, bbox, COCO/YOLO/PNG de tudo, sem divisão), usadas pelos testes;
  * exportar_treino: o diálogo de exportação da interface. Divide por imagem em treino e
    validação (semente fixa) e escreve, cada formato na sua pasta, prontos para treinar:
      yolo/       YOLO-seg (Ultralytics YOLO11 e YOLO26)
      coco/       COCO de instâncias (Mask R-CNN: torchvision, Detectron2, MMDetection)
      semantica/  máscaras PNG de índice de classe (U-Net, DeepLabV3, SegFormer)
    Cada pasta leva um README_TREINO.md e, quando faz sentido, um script de treino curto.

Regras documentadas:
  * polígonos: cv2.findContours, contornos externos;
  * objeto em vários pedaços: no COCO, um polígono por pedaço na mesma anotação; no YOLO,
    os pedaços são costurados num polígono só por pontes de largura zero entre os pontos
    mais próximos (uma linha por objeto, como a Ultralytics espera);
  * objeto com furo, ou fino demais para polígono (o polígono rasterizado pelo pycocotools
    fica com IoU abaixo de 0,9 contra a máscara, caso das trincas): no COCO vira RLE
    (pycocotools) se o pacote estiver instalado; sem ele, vai como polígono e perde detalhe.
    No YOLO só existe polígono: furos se perdem e objetos finos encolhem;
  * semântica: objetos pintados na ordem em que foram aceitos; na sobreposição, o último
    aceito vence.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .projeto import Projeto, cor_da_classe

AREA_MINIMA_CONTORNO = 4.0  # contornos menores que isso (em px²) são descartados
FORMATOS = {"yolo": "YOLO-seg (Ultralytics)", "coco": "COCO de instâncias (Mask R-CNN)",
            "semantica": "Máscaras semânticas PNG (U-Net)"}
MODELOS_DIR = Path(__file__).resolve().parent / "modelos_treino"


# ================================================================ geometria

def mascara_para_poligonos(mascara: np.ndarray) -> list[list[float]]:
    """Lista de polígonos [x0, y0, x1, y1, ...] em pixels, do maior para o menor."""
    m = np.ascontiguousarray(np.asarray(mascara, dtype=np.uint8))
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poligonos = []
    for c in sorted(contornos, key=cv2.contourArea, reverse=True):
        if len(c) < 3 or cv2.contourArea(c) < AREA_MINIMA_CONTORNO:
            continue
        poligonos.append([float(v) for v in c.reshape(-1, 2).ravel()])
    return poligonos


def tem_furo(mascara: np.ndarray) -> bool:
    """True se algum contorno interno (furo) tem área relevante."""
    m = np.ascontiguousarray(np.asarray(mascara, dtype=np.uint8))
    contornos, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    return any(h[3] != -1 and cv2.contourArea(c) >= AREA_MINIMA_CONTORNO for c, h in zip(contornos, hier[0]))


def juntar_pedacos(poligonos: list[list[float]]) -> list[float]:
    """Costura vários polígonos num só, por pontes de largura zero.

    Cada pedaço novo entra no ponto do contorno acumulado mais próximo dele: o caminho
    vai até o pedaço, dá a volta nele e retorna pelo mesmo segmento. Rasterizado, cobre
    a união dos pedaços mais as linhas finas das pontes.
    """
    if not poligonos:
        return []
    caminho = np.array(poligonos[0], dtype=np.float64).reshape(-1, 2)
    for p in poligonos[1:]:
        q = np.array(p, dtype=np.float64).reshape(-1, 2)
        d = ((caminho[:, None, :] - q[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        volta = np.concatenate([q[j:], q[:j + 1]])  # começa e termina em q[j]
        caminho = np.concatenate([caminho[:i + 1], volta, caminho[i:]])
    return caminho.ravel().tolist()


def bbox_xywh(mascara: np.ndarray) -> list[float]:
    """Caixa COCO [x, y, largura, altura] cobrindo todos os pixels da máscara."""
    ys, xs = np.nonzero(mascara)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def poligono_yolo(poligono: list[float], largura: int, altura: int) -> list[float]:
    xy = np.array(poligono, dtype=np.float64).reshape(-1, 2)
    xy[:, 0] = np.clip(xy[:, 0] / largura, 0.0, 1.0)
    xy[:, 1] = np.clip(xy[:, 1] / altura, 0.0, 1.0)
    return xy.ravel().tolist()


def linha_yolo(mascara: np.ndarray, classe: int, largura: int, altura: int) -> str | None:
    polis = mascara_para_poligonos(mascara)
    if not polis:
        return None
    coords = poligono_yolo(juntar_pedacos(polis), largura, altura)
    return " ".join([str(classe)] + [f"{v:.6f}" for v in coords])


def pycocotools_disponivel() -> bool:
    try:
        import pycocotools.mask  # noqa: F401
        return True
    except Exception:
        return False


LIMIAR_FIDELIDADE_POLIGONO = 0.9  # abaixo disso o objeto vai em RLE (objetos finos, como trincas)


def fidelidade_poligono(mascara: np.ndarray, poligonos: list[list[float]]) -> float:
    """IoU entre a máscara e os polígonos rasterizados pelo próprio pycocotools (annToMask)."""
    from pycocotools import mask as mascara_coco

    from .metricas import iou

    h, w = mascara.shape
    validos = [p for p in poligonos if len(p) >= 6]
    if not validos:
        return 0.0
    raster = mascara_coco.decode(mascara_coco.merge(mascara_coco.frPyObjects(validos, h, w)))
    return iou(raster > 0, mascara)


def segmentacao_coco(mascara: np.ndarray, usar_rle: bool | None = None) -> list | dict:
    """Polígonos; RLE compactado (pycocotools) quando o objeto tem furo ou quando o polígono
    perde muito (IoU do polígono rasterizado abaixo de LIMIAR_FIDELIDADE_POLIGONO)."""
    if usar_rle is None:
        usar_rle = pycocotools_disponivel()
    poligonos = mascara_para_poligonos(mascara)
    if usar_rle and (tem_furo(mascara) or fidelidade_poligono(mascara, poligonos) < LIMIAR_FIDELIDADE_POLIGONO):
        from pycocotools import mask as mascara_coco

        rle = mascara_coco.encode(np.asfortranarray(np.asarray(mascara, dtype=np.uint8)))
        return {"size": [int(v) for v in rle["size"]], "counts": rle["counts"].decode("ascii")}
    return poligonos


def _anotacao_coco(obj, id_anotacao: int, id_imagem: int, id_categoria: int, usar_rle=None) -> dict:
    m = obj.mascara()
    return {
        "id": id_anotacao, "image_id": id_imagem, "category_id": id_categoria,
        "segmentation": segmentacao_coco(m, usar_rle), "area": float(m.sum()), "bbox": bbox_xywh(m),
        "iscrowd": 0,
        # campos extras (o COCO e os frameworks ignoram chaves desconhecidas)
        "sam_escore_previsto": round(obj.escore_previsto, 4), "sam_cliques": obj.cliques,
        "sam_tempo_s": round(obj.tempo_s, 2), "sam_origem": obj.origem,
        "iou_referencia": None if obj.iou_real is None else round(obj.iou_real, 4),
    }


# ================================================================ exportação simples (sem divisão)

def _imagens_anotadas(projeto: Projeto):
    for rel in sorted(projeto.imagens):
        img = projeto.imagens[rel]
        if img.objetos:
            yield rel, img


def _categorias(projeto: Projeto) -> list[dict]:
    return [{"id": i + 1, "name": c, "supercategory": "objeto"} for i, c in enumerate(projeto.classes)]


def exportar_coco(projeto: Projeto, destino: Path) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    id_classe = {c: i + 1 for i, c in enumerate(projeto.classes)}
    imagens, anotacoes = [], []
    id_anotacao = 1
    for id_imagem, (rel, img) in enumerate(_imagens_anotadas(projeto), start=1):
        imagens.append({"id": id_imagem, "file_name": rel, "width": img.largura, "height": img.altura})
        for obj in img.objetos:
            anotacoes.append(_anotacao_coco(obj, id_anotacao, id_imagem, id_classe[obj.classe]))
            id_anotacao += 1
    coco = {"info": {"description": "Anotações feitas com sam-anota",
                     "date_created": datetime.now().isoformat(timespec="seconds"),
                     "pasta_imagens": str(projeto.pasta_imagens)},
            "licenses": [], "images": imagens, "annotations": anotacoes, "categories": _categorias(projeto)}
    destino.write_text(json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
    return destino


def exportar_yolo(projeto: Projeto, pasta: Path) -> list[Path]:
    """Um .txt por imagem em pasta/labels/, mantendo as subpastas (sem divisão treino e validação)."""
    pasta = Path(pasta)
    gerados = []
    for rel, img in _imagens_anotadas(projeto):
        linhas = [linha_yolo(o.mascara(), projeto.classes.index(o.classe), img.largura, img.altura)
                  for o in img.objetos]
        linhas = [l for l in linhas if l]
        destino = pasta / "labels" / Path(rel).with_suffix(".txt")
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")
        gerados.append(destino)
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "classes.txt").write_text("\n".join(projeto.classes) + "\n", encoding="utf-8")
    nomes = "\n".join(f"  {i}: {c}" for i, c in enumerate(projeto.classes))
    (pasta / "data.yaml").write_text(f"path: .\ntrain: images\nval: images\nnames:\n{nomes}\n", encoding="utf-8")
    return gerados


def exportar_mascaras_png(projeto: Projeto, pasta: Path) -> list[Path]:
    """Uma PNG binária (0 e 255) por objeto: <imagem>_obj<id>_<classe>.png."""
    pasta = Path(pasta)
    gerados = []
    for rel, img in _imagens_anotadas(projeto):
        base = Path(rel).with_suffix("")
        for obj in img.objetos:
            classe = re.sub(r"[^A-Za-z0-9_-]+", "_", obj.classe)
            destino = pasta / base.parent / f"{base.name}_obj{obj.id:04d}_{classe}.png"
            destino.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(obj.mascara().astype(np.uint8) * 255).save(destino)
            gerados.append(destino)
    return gerados


def exportar_tudo(projeto: Projeto, pasta: Path | None = None) -> dict:
    pasta = Path(pasta or projeto.pasta_saida / "exportacao")
    coco = exportar_coco(projeto, pasta / "coco.json")
    yolo = exportar_yolo(projeto, pasta / "yolo")
    pngs = exportar_mascaras_png(projeto, pasta / "mascaras")
    return {"pasta": pasta, "coco": coco, "yolo": yolo, "mascaras": pngs}


# ================================================================ exportação para treino

@dataclass
class OpcoesTreino:
    formatos: tuple[str, ...] = ("yolo", "coco", "semantica")
    frac_val: float = 0.2
    semente: int = 42
    incluir_vazias: bool = False  # imagens concluídas sem objeto entram como exemplos negativos
    so_concluidas: bool = False  # False: todas as imagens anotadas (com objeto ou concluídas)


@dataclass
class ResultadoTreino:
    pasta: Path
    treino: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    objetos: int = 0
    arquivos: dict = field(default_factory=dict)


def selecionar_imagens(projeto: Projeto, op: OpcoesTreino) -> list[str]:
    """Imagens que entram no conjunto.

    Candidatas: só as concluídas (so_concluidas) ou todas as anotadas (com objeto ou concluídas).
    Imagens nunca tocadas não entram: não sabemos se estão vazias ou só pendentes.
    Sem incluir_vazias, as candidatas sem objeto saem.
    """
    sel = []
    for rel in sorted(projeto.imagens):
        img = projeto.imagens[rel]
        candidata = img.concluida if op.so_concluidas else (img.concluida or bool(img.objetos))
        if not candidata:
            continue
        if not img.objetos and not op.incluir_vazias:
            continue
        sel.append(rel)
    return sel


def dividir(rels: list[str], frac_val: float, semente: int) -> tuple[list[str], list[str]]:
    """Divisão por imagem, reprodutível. Com 2 imagens ou mais, a validação tem pelo menos 1."""
    ordem = sorted(rels)
    random.Random(semente).shuffle(ordem)
    n_val = int(round(len(ordem) * frac_val))
    if frac_val > 0 and len(ordem) >= 2:
        n_val = min(max(n_val, 1), len(ordem) - 1)
    val = sorted(ordem[:n_val])
    treino = sorted(ordem[n_val:])
    return treino, val


def nome_plano(rel: str) -> str:
    """dificeis/fumaca/01.jpg -> dificeis__fumaca__01.jpg (sem subpastas, sem colisão)."""
    return rel.replace("/", "__")


def _copiar_imagem(origem: Path, destino: Path) -> tuple[int, int]:
    """Copia; se a foto tiver rotação EXIF, grava já girada (as máscaras estão nessa orientação)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(origem) as im:
        orientacao = im.getexif().get(0x0112, 1)
        if orientacao not in (1, None):
            girada = ImageOps.exif_transpose(im).convert("RGB")
            girada.save(destino, quality=95)
            return girada.size
        tamanho = im.size
    shutil.copy2(origem, destino)
    return tamanho


def _texto_modelo(nome: str, **valores) -> str:
    texto = (MODELOS_DIR / nome).read_text(encoding="utf-8")
    for k, v in valores.items():
        texto = texto.replace("{{" + k + "}}", str(v))
    return texto


def exportar_treino(projeto: Projeto, pasta: Path | None = None, op: OpcoesTreino | None = None,
                    progresso=None) -> ResultadoTreino:
    """Escreve os formatos escolhidos, já divididos em treino e validação.

    progresso(texto) recebe mensagens curtas (para a interface).
    """
    op = op or OpcoesTreino()
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta = Path(pasta or projeto.pasta_saida / f"treino_{carimbo}")
    rels = selecionar_imagens(projeto, op)
    if not rels:
        raise ValueError("Nenhuma imagem para exportar com essas opções (anote ou marque imagens como concluídas).")
    treino, val = dividir(rels, op.frac_val, op.semente)
    res = ResultadoTreino(pasta, treino, val, sum(len(projeto.imagens[r].objetos) for r in rels))
    divisoes = {"train": treino, "val": val}
    classes = list(projeto.classes)
    avisar = progresso or (lambda _t: None)

    # cada objeto decodificado uma vez só
    mascaras = {r: [(o, o.mascara()) for o in projeto.imagens[r].objetos] for r in rels}

    for formato in op.formatos:
        raiz = pasta / formato
        avisar(f"{FORMATOS[formato]}: copiando imagens...")
        for split, lista in divisoes.items():
            for rel in lista:
                _copiar_imagem(projeto.caminho_absoluto(rel), raiz / "images" / split / nome_plano(rel))
        if formato == "yolo":
            res.arquivos["yolo"] = _escrever_yolo(projeto, raiz, divisoes, mascaras, classes)
        elif formato == "coco":
            res.arquivos["coco"] = _escrever_coco(projeto, raiz, divisoes, mascaras, classes)
        elif formato == "semantica":
            res.arquivos["semantica"] = _escrever_semantica(projeto, raiz, divisoes, mascaras, classes)
        avisar(f"{FORMATOS[formato]}: pronto.")

    resumo = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "pasta_imagens": str(projeto.pasta_imagens),
        "classes": classes, "formatos": list(op.formatos),
        "frac_val": op.frac_val, "semente": op.semente, "incluir_vazias": op.incluir_vazias,
        "so_concluidas": op.so_concluidas, "objetos": res.objetos,
        "treino": treino, "val": val,
    }
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "divisao.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def _escrever_yolo(projeto, raiz: Path, divisoes, mascaras, classes) -> dict:
    n_linhas = 0
    for split, lista in divisoes.items():
        for rel in lista:
            img = projeto.imagens[rel]
            linhas = [linha_yolo(m, classes.index(o.classe), img.largura, img.altura) for o, m in mascaras[rel]]
            linhas = [l for l in linhas if l]
            n_linhas += len(linhas)
            destino = raiz / "labels" / split / Path(nome_plano(rel)).with_suffix(".txt")
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")
    nomes = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    # path: "." é resolvido a partir da pasta onde o comando roda (rode dentro de yolo/)
    (raiz / "data.yaml").write_text(
        "# Gerado pelo sam-anota. Rode o treino de dentro desta pasta.\n"
        f"path: .\ntrain: images/train\nval: images/val\nnames:\n{nomes}\n", encoding="utf-8")
    (raiz / "README_TREINO.md").write_text(_texto_modelo("yolo_README.md", N=len(classes)), encoding="utf-8")
    return {"linhas": n_linhas, "data_yaml": raiz / "data.yaml"}


def _escrever_coco(projeto, raiz: Path, divisoes, mascaras, classes) -> dict:
    id_classe = {c: i + 1 for i, c in enumerate(classes)}
    categorias = [{"id": i + 1, "name": c, "supercategory": "objeto"} for i, c in enumerate(classes)]
    usar_rle = pycocotools_disponivel()
    arquivos, n_rle = {}, 0
    for split, lista in divisoes.items():
        imagens, anotacoes = [], []
        id_anotacao = 1
        for id_imagem, rel in enumerate(lista, start=1):
            img = projeto.imagens[rel]
            imagens.append({"id": id_imagem, "file_name": nome_plano(rel), "width": img.largura,
                            "height": img.altura, "sam_anota_origem": rel})
            for o, m in mascaras[rel]:
                a = {"id": id_anotacao, "image_id": id_imagem, "category_id": id_classe[o.classe],
                     "segmentation": segmentacao_coco(m, usar_rle), "area": float(m.sum()),
                     "bbox": bbox_xywh(m), "iscrowd": 0,
                     "sam_escore_previsto": round(o.escore_previsto, 4), "sam_origem": o.origem,
                     "iou_referencia": None if o.iou_real is None else round(o.iou_real, 4)}
                n_rle += isinstance(a["segmentation"], dict)
                anotacoes.append(a)
                id_anotacao += 1
        coco = {"info": {"description": f"sam-anota, divisão {split}",
                         "date_created": datetime.now().isoformat(timespec="seconds")},
                "licenses": [], "images": imagens, "annotations": anotacoes, "categories": categorias}
        destino = raiz / "annotations" / f"instances_{split}.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        arquivos[split] = destino
    aviso_furo = (f"Objetos com furo, ou finos demais para polígono, foram gravados em RLE ({n_rle} de "
                  f"{sum(len(mascaras[r]) for l in divisoes.values() for r in l)} anotações): o polígono "
                  "rasterizado pelo pycocotools precisa ter IoU de 0,9 ou mais com a máscara para ficar como "
                  "polígono." if usar_rle else
                  "pycocotools não estava instalado: todos os objetos foram gravados como polígono externo "
                  "(furos se perderam e objetos finos encolhem).")
    (raiz / "README_TREINO.md").write_text(
        _texto_modelo("coco_README.md", N=len(classes), AVISO_FURO=aviso_furo), encoding="utf-8")
    shutil.copy2(MODELOS_DIR / "treinar_torchvision.py", raiz / "treinar_torchvision.py")
    return {**arquivos, "rle": n_rle}


def _escrever_semantica(projeto, raiz: Path, divisoes, mascaras, classes) -> dict:
    cores = np.array([(0, 0, 0)] + [cor_da_classe(i) for i in range(len(classes))], dtype=np.uint8)
    for split, lista in divisoes.items():
        for rel in lista:
            img = projeto.imagens[rel]
            indice = np.zeros((img.altura, img.largura), dtype=np.uint8)
            for o, m in mascaras[rel]:  # ordem de aceitação: o último aceito vence
                indice[m] = classes.index(o.classe) + 1
            nome = Path(nome_plano(rel)).with_suffix(".png").name
            destino = raiz / "masks" / split / nome
            destino.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(indice).save(destino)  # PNG de 8 bits (modo L), valor = índice da classe
            dvis = raiz / "visualizacao" / split / nome  # só para olhar: cores da classes.json
            dvis.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cores[indice]).save(dvis)
    info = [{"indice": 0, "nome": "fundo", "cor": [0, 0, 0]}] + [
        {"indice": i + 1, "nome": c, "cor": list(cor_da_classe(i))} for i, c in enumerate(classes)]
    (raiz / "classes.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    (raiz / "README_TREINO.md").write_text(_texto_modelo("semantica_README.md", N=len(classes) + 1),
                                           encoding="utf-8")
    shutil.copy2(MODELOS_DIR / "treinar_unet.py", raiz / "treinar_unet.py")
    return {"classes_json": raiz / "classes.json"}
