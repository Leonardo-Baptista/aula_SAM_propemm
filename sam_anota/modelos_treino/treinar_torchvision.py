"""Treino mínimo de Mask R-CNN (torchvision) sobre a exportação COCO do sam-anota.

Rode de dentro da pasta coco/ da exportação:
    pip install torch torchvision pycocotools
    python treinar_torchvision.py --iteracoes 300

Lê annotations/instances_train.json e images/train; no fim mede a perda média em val.
Opções: --iteracoes, --lote, --lr, --dispositivo (auto, cpu, cuda), --sem-pretreino.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class CocoInstancias(torch.utils.data.Dataset):
    def __init__(self, raiz: str, divisao: str):
        self.coco = COCO(os.path.join(raiz, "annotations", f"instances_{divisao}.json"))
        self.ids = sorted(self.coco.imgs)
        self.pasta = os.path.join(raiz, "images", divisao)
        # categorias do COCO (1..N) viram rótulos 1..N; 0 é o fundo no torchvision
        self.rotulo = {c: i + 1 for i, c in enumerate(sorted(self.coco.cats))}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        info = self.coco.imgs[self.ids[i]]
        img = torchvision.io.read_image(os.path.join(self.pasta, info["file_name"]),
                                        torchvision.io.ImageReadMode.RGB).float() / 255
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=info["id"], iscrowd=False))
        h, w = info["height"], info["width"]
        if anns:
            masks = torch.as_tensor(np.stack([self.coco.annToMask(a) for a in anns]), dtype=torch.uint8)
            boxes = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                                   a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=torch.float32)
            labels = torch.tensor([self.rotulo[a["category_id"]] for a in anns], dtype=torch.int64)
        else:  # imagem sem objeto: exemplo negativo
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels, "masks": masks}


def criar_modelo(num_classes: int, pretreino: bool):
    modelo = maskrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretreino else None,
                                      weights_backbone=None)
    entrada = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(entrada, num_classes)
    modelo.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, num_classes)
    return modelo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=".")
    ap.add_argument("--iteracoes", type=int, default=300)
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--sem-pretreino", action="store_true", help="não baixa os pesos COCO (cerca de 180 MB)")
    a = ap.parse_args()
    disp = ("cuda" if torch.cuda.is_available() else "cpu") if a.dispositivo == "auto" else a.dispositivo

    treino = CocoInstancias(a.raiz, "train")
    num_classes = len(treino.rotulo) + 1
    carregador = torch.utils.data.DataLoader(treino, batch_size=a.lote, shuffle=True,
                                             collate_fn=lambda b: tuple(zip(*b)))
    modelo = criar_modelo(num_classes, not a.sem_pretreino).to(disp)
    otim = torch.optim.SGD([p for p in modelo.parameters() if p.requires_grad], lr=a.lr,
                           momentum=0.9, weight_decay=1e-4)
    modelo.train()
    it, t0, perdas = 0, time.perf_counter(), []
    while it < a.iteracoes:
        for imgs, alvos in carregador:
            imgs = [i.to(disp) for i in imgs]
            alvos = [{k: v.to(disp) for k, v in t.items()} for t in alvos]
            perda = sum(modelo(imgs, alvos).values())
            otim.zero_grad()
            perda.backward()
            otim.step()
            perdas.append(float(perda))
            it += 1
            print(f"iteração {it}/{a.iteracoes}  perda {float(perda):.3f}", flush=True)
            if it >= a.iteracoes:
                break

    val = CocoInstancias(a.raiz, "val")
    perdas_val = []
    with torch.no_grad():  # perda em val (modo treino só para o modelo devolver as perdas)
        for i in range(len(val)):
            img, alvo = val[i]
            perdas_val.append(float(sum(modelo([img.to(disp)], [{k: v.to(disp) for k, v in alvo.items()}]).values())))
    modelo.eval()
    with torch.no_grad():
        saida = modelo([val[0][0].to(disp)])[0] if len(val) else {"scores": torch.zeros(0)}
    resultado = {"dispositivo": disp, "iteracoes": it, "tempo_s": round(time.perf_counter() - t0, 1),
                 "perda_inicial": round(perdas[0], 4), "perda_final": round(perdas[-1], 4),
                 "perda_val_media": round(float(np.mean(perdas_val)), 4) if perdas_val else None,
                 "deteccoes_na_primeira_val": int((saida["scores"] > 0.5).sum())}
    torch.save(modelo.state_dict(), "maskrcnn_treinado.pt")
    print(json.dumps(resultado, ensure_ascii=False))


if __name__ == "__main__":
    main()
