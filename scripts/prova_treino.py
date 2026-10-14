"""Prova de que a exportação está pronta para treinar.

1. Anota com o SAM, sem interface, imagens de dados/exemplos e dados/dificeis (3 classes:
   animal, fumaca, trinca; nas difíceis o prompt é a caixa da referência mais um ponto).
2. Exporta nos três formatos (80/20, semente 42, com uma imagem vazia como negativa).
3. Valida: COCO com pycocotools (carrega; annToMask contra a máscara do projeto),
   máscaras semânticas (valores, tamanhos) e rótulos YOLO (linhas, faixa 0 a 1).
4. Treina de verdade, poucas iterações: YOLO11n-seg (python com ultralytics, passado em
   --python-yolo), Mask R-CNN do torchvision e U-Net (scripts gerados na exportação).

    python scripts/prova_treino.py --saida <pasta> [--python-yolo <python com ultralytics>]
           [--yolo-pesos yolo11n-seg.pt] [--dispositivo cuda] [--etapas anotar,exportar,validar,treinar]

Nada é escrito em dados/: o projeto e as saídas vão para --saida.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))


def anotar(dispositivo: str, modelo_nome: str):
    import cv2
    from PIL import Image

    from sam_anota.metricas import caminho_referencia, carregar_referencia, iou_contra_referencia
    from sam_anota.modelo import ModeloSAM, carregar_imagem_rgb
    from sam_anota.projeto import Projeto

    proj = Projeto.abrir_ou_criar(RAIZ / "dados")
    proj.classes = ["animal", "fumaca", "trinca"]
    for img in proj.imagens.values():
        img.objetos, img.concluida = [], False
    m = ModeloSAM(modelo_nome, dispositivo=dispositivo, cache_disco=False)
    m.carregar()
    print(f"anotando com {m.descricao()}")

    def anotar_imagem(rel, prompts, classe):
        caminho = proj.caminho_absoluto(rel)
        img = carregar_imagem_rgb(caminho)
        proj.imagens[rel].altura, proj.imagens[rel].largura = img.shape[:2]
        m.codificar(rel, img)
        ref = caminho_referencia(caminho)
        rot = carregar_referencia(ref, img.shape[:2]) if ref else None
        for pontos, caixa in prompts:
            p = m.prever(rel, pontos, caixa)
            i = int(np.argmax(p.escores))
            iou = iou_contra_referencia(p.mascaras[i], rot) if rot is not None else None
            proj.adicionar_objeto(rel, p.mascaras[i], classe, float(p.escores[i]), i, len(pontos or []) + (caixa is not None),
                                  0.0, pontos=pontos, caixa=caixa, iou_real=iou,
                                  origem=f"{m.info.rotulo} (prova_treino)")
        proj.imagens[rel].concluida = True

    anotar_imagem("exemplos/cao_e_gato.jpg", [([(420, 650, 1), (800, 450, 1)], None),
                                              ([(1250, 650, 1), (560, 180, 1), (500, 650, 0)], None)], "animal")
    proj.imagens["exemplos/transito_pune.jpg"].concluida = True  # imagem sem objeto (negativa)
    with Image.open(proj.caminho_absoluto("exemplos/transito_pune.jpg")) as im:
        proj.imagens["exemplos/transito_pune.jpg"].largura, proj.imagens["exemplos/transito_pune.jpg"].altura = im.size
    for rel in sorted(proj.imagens):
        if not rel.startswith(("dificeis/fumaca/", "dificeis/trinca/")):
            continue
        caminho = proj.caminho_absoluto(rel)
        rot = carregar_referencia(caminho_referencia(caminho))
        valores, contagens = np.unique(rot[rot > 0], return_counts=True)
        comp = (rot == valores[contagens.argmax()]).astype(np.uint8)
        d = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(d.argmax()), d.shape)
        ys, xs = np.nonzero(comp)
        caixa = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        anotar_imagem(rel, [([(float(x), float(y), 1)], caixa)], "trinca" if "/trinca/" in rel else "fumaca")
    proj.salvar()
    total = proj.total_objetos()
    ious = [o.iou_real for i in proj.imagens.values() for o in i.objetos if o.iou_real is not None]
    print(f"{total} objetos em {sum(1 for i in proj.imagens.values() if i.concluida)} imagens concluídas; "
          f"IoU média contra a referência (caixa + ponto) {np.mean(ious):.3f}")
    return proj


def exportar(proj, pasta: Path):
    from sam_anota.exportar import OpcoesTreino, exportar_treino

    t0 = time.perf_counter()
    r = exportar_treino(proj, pasta, OpcoesTreino(frac_val=0.2, semente=42, incluir_vazias=True, so_concluidas=True),
                        progresso=print)
    print(f"exportado em {time.perf_counter() - t0:.1f} s: treino {len(r.treino)} imagens, val {len(r.val)}: {r.val}")
    return r


def validar(proj, pasta: Path) -> dict:
    from PIL import Image
    from pycocotools.coco import COCO

    from sam_anota.exportar import nome_plano
    from sam_anota.metricas import iou

    div = json.loads((pasta / "divisao.json").read_text(encoding="utf-8"))
    div["train"] = div["treino"]  # nomes das pastas usam train e val
    relatorio = {}
    # COCO
    ious, n_rle = [], 0
    for split in ("train", "val"):
        coco = COCO(str(pasta / "coco" / "annotations" / f"instances_{split}.json"))
        por_nome = {nome_plano(r): r for r in div[split]}
        for id_img, info in coco.imgs.items():
            assert (pasta / "coco" / "images" / split / info["file_name"]).exists()
            objs = proj.imagens[por_nome[info["file_name"]]].objetos
            anns = coco.loadAnns(coco.getAnnIds(imgIds=id_img))
            assert len(anns) == len(objs), (info["file_name"], len(anns), len(objs))
            for a, o in zip(anns, objs):
                n_rle += isinstance(a["segmentation"], dict)
                ious.append(iou(coco.annToMask(a) > 0, o.mascara()))
    relatorio["coco"] = {"anotacoes": len(ious), "rle": n_rle, "iou_annToMask_min": round(min(ious), 4),
                         "iou_annToMask_media": round(float(np.mean(ious)), 4)}
    # semântica
    n_classes = len(div["classes"])
    for split in ("train", "val"):
        for rel in div[split]:
            nome = Path(nome_plano(rel)).with_suffix(".png").name
            m = np.array(Image.open(pasta / "semantica" / "masks" / split / nome))
            img = proj.imagens[rel]
            assert m.dtype == np.uint8 and m.shape == (img.altura, img.largura)
            assert m.max() <= n_classes
    relatorio["semantica"] = {"mascaras": len(div["train"]) + len(div["val"]), "ok": True}
    # YOLO
    linhas = 0
    for split in ("train", "val"):
        for rel in div[split]:
            txt = (pasta / "yolo" / "labels" / split / Path(nome_plano(rel)).with_suffix(".txt")).read_text()
            ls = [l for l in txt.splitlines() if l.strip()]
            assert len(ls) == len(proj.imagens[rel].objetos)
            for l in ls:
                v = [float(x) for x in l.split()[1:]]
                assert len(v) % 2 == 0 and len(v) >= 6 and all(0 <= x <= 1 for x in v)
            linhas += len(ls)
    relatorio["yolo"] = {"linhas": linhas, "ok": True}
    print("validação:", json.dumps(relatorio, ensure_ascii=False))
    return relatorio


def _rodar(cmd, cwd):
    print("\n>", " ".join(str(c) for c in cmd), f"(em {cwd})", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    saida = (r.stdout or "") + (r.stderr or "")
    print("\n".join(saida.strip().splitlines()[-8:]))
    if r.returncode != 0:
        raise SystemExit(f"falhou (código {r.returncode})")
    return saida, time.perf_counter() - t0


def treinar(pasta: Path, python_yolo: str | None, yolo_pesos: str, dispositivo: str) -> dict:
    res = {}
    if python_yolo:
        codigo = (f"from ultralytics import YOLO; r = YOLO(r'{yolo_pesos}').train(data='data.yaml', epochs=1, imgsz=320, "
                  f"batch=4, workers=0, device='{0 if dispositivo == 'cuda' else 'cpu'}', project='runs', name='prova', "
                  "exist_ok=True, plots=False, verbose=False); print('MASK_MAP50', r.seg.map50, 'BOX_MAP50', r.box.map50)")
        saida, t = _rodar([python_yolo, "-c", codigo], pasta / "yolo")
        linha = [l for l in saida.splitlines() if l.startswith("MASK_MAP50")]
        res["yolo"] = {"tempo_s": round(t, 1), "resultado": linha[-1] if linha else "treinou (sem métrica impressa)"}
    py = sys.executable
    saida, t = _rodar([py, "treinar_torchvision.py", "--iteracoes", "4", "--lote", "2", "--dispositivo", dispositivo],
                      pasta / "coco")
    res["maskrcnn"] = json.loads(saida.strip().splitlines()[-1])
    saida, t = _rodar([py, "treinar_unet.py", "--iteracoes", "4", "--lote", "2", "--tamanho", "256",
                       "--dispositivo", dispositivo, "--sem-pretreino"], pasta / "semantica")
    res["unet"] = json.loads(saida.strip().splitlines()[-1])
    print("\ntreinos:", json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--python-yolo", default=None, help="python de um venv com ultralytics (AGPL, fora do app)")
    ap.add_argument("--yolo-pesos", default="yolo11n-seg.pt")
    ap.add_argument("--dispositivo", default="cpu")
    ap.add_argument("--modelo", default="sam2.1_tiny")
    a = ap.parse_args()
    os.environ["SAM_ANOTA_SAIDAS"] = str(a.saida / "saidas")
    import sam_anota.projeto as projeto
    projeto.PASTA_SAIDAS = a.saida / "saidas"

    proj = anotar(a.dispositivo, a.modelo)
    pasta = a.saida / "exportacao"
    exportar(proj, pasta)
    validacao = validar(proj, pasta)
    treinos = treinar(pasta, a.python_yolo, a.yolo_pesos, a.dispositivo)
    (a.saida / "prova_treino.json").write_text(json.dumps({"validacao": validacao, "treinos": treinos},
                                                          ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nProva concluída. Resumo em {a.saida / 'prova_treino.json'}")


if __name__ == "__main__":
    main()
