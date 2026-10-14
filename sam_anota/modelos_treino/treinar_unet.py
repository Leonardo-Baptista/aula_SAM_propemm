"""Treino mínimo de U-Net sobre a exportação semântica do sam-anota.

Rode de dentro da pasta semantica/ da exportação:
    pip install torch segmentation-models-pytorch opencv-python-headless
    python treinar_unet.py --iteracoes 500

Usa segmentation_models_pytorch (Unet com encoder resnet34) se estiver instalado;
senão, uma U-Net mínima definida aqui mesmo. No fim mede a mIoU em val.
Opções: --iteracoes, --lote, --tamanho (múltiplo de 32), --dispositivo, --sem-pretreino.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Semantica(torch.utils.data.Dataset):
    def __init__(self, raiz: str, divisao: str, tamanho: int):
        self.imagens = sorted(glob.glob(os.path.join(raiz, "images", divisao, "*")))
        self.pasta_mascaras = os.path.join(raiz, "masks", divisao)
        self.tamanho = tamanho

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, i):
        caminho = self.imagens[i]
        nome = os.path.splitext(os.path.basename(caminho))[0] + ".png"
        img = cv2.cvtColor(cv2.imread(caminho), cv2.COLOR_BGR2RGB)
        mascara = cv2.imread(os.path.join(self.pasta_mascaras, nome), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.tamanho, self.tamanho), interpolation=cv2.INTER_AREA)
        mascara = cv2.resize(mascara, (self.tamanho, self.tamanho), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
        img = (img - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return img, torch.from_numpy(mascara.astype(np.int64))


class UNetMinima(nn.Module):
    """U-Net de 4 níveis, para quando o segmentation_models_pytorch não estiver instalado."""

    def __init__(self, classes: int, base: int = 32):
        super().__init__()

        def bloco(e, s):
            return nn.Sequential(nn.Conv2d(e, s, 3, padding=1), nn.BatchNorm2d(s), nn.ReLU(inplace=True),
                                 nn.Conv2d(s, s, 3, padding=1), nn.BatchNorm2d(s), nn.ReLU(inplace=True))
        c = [base, base * 2, base * 4, base * 8]
        self.desce = nn.ModuleList([bloco(3, c[0]), bloco(c[0], c[1]), bloco(c[1], c[2]), bloco(c[2], c[3])])
        self.sobe = nn.ModuleList([nn.ConvTranspose2d(c[i], c[i - 1], 2, stride=2) for i in (3, 2, 1)])
        self.junta = nn.ModuleList([bloco(c[i - 1] * 2, c[i - 1]) for i in (3, 2, 1)])
        self.saida = nn.Conv2d(c[0], classes, 1)

    def forward(self, x):
        atalhos = []
        for i, b in enumerate(self.desce):
            x = b(x if i == 0 else F.max_pool2d(x, 2))
            atalhos.append(x)
        for sobe, junta, atalho in zip(self.sobe, self.junta, atalhos[-2::-1]):
            x = junta(torch.cat([sobe(x), atalho], dim=1))
        return self.saida(x)


def criar_modelo(classes: int, pretreino: bool):
    try:
        import segmentation_models_pytorch as smp
        return smp.Unet("resnet34", encoder_weights="imagenet" if pretreino else None, classes=classes), "smp.Unet(resnet34)"
    except ImportError:
        return UNetMinima(classes), "UNetMinima"


def miou(pred: torch.Tensor, alvo: torch.Tensor, classes: int) -> float:
    ious = []
    for c in range(classes):
        p, a = pred == c, alvo == c
        uniao = (p | a).sum().item()
        if uniao:
            ious.append((p & a).sum().item() / uniao)
    return float(np.mean(ious)) if ious else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=".")
    ap.add_argument("--iteracoes", type=int, default=500)
    ap.add_argument("--lote", type=int, default=4)
    ap.add_argument("--tamanho", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--sem-pretreino", action="store_true", help="não baixa os pesos ImageNet do encoder")
    a = ap.parse_args()
    disp = ("cuda" if torch.cuda.is_available() else "cpu") if a.dispositivo == "auto" else a.dispositivo

    with open(os.path.join(a.raiz, "classes.json"), encoding="utf-8") as f:
        classes = len(json.load(f))  # inclui o fundo (índice 0)
    treino = Semantica(a.raiz, "train", a.tamanho)
    carregador = torch.utils.data.DataLoader(treino, batch_size=a.lote, shuffle=True)
    modelo, nome = criar_modelo(classes, not a.sem_pretreino)
    modelo = modelo.to(disp)
    otim = torch.optim.AdamW(modelo.parameters(), lr=a.lr)
    modelo.train()
    it, t0, perdas = 0, time.perf_counter(), []
    while it < a.iteracoes:
        for img, alvo in carregador:
            img, alvo = img.to(disp), alvo.to(disp)
            perda = F.cross_entropy(modelo(img), alvo)
            otim.zero_grad()
            perda.backward()
            otim.step()
            perdas.append(float(perda))
            it += 1
            print(f"iteração {it}/{a.iteracoes}  perda {float(perda):.3f}", flush=True)
            if it >= a.iteracoes:
                break

    modelo.eval()
    val = Semantica(a.raiz, "val", a.tamanho)
    valores = []
    with torch.no_grad():
        for i in range(len(val)):
            img, alvo = val[i]
            pred = modelo(img[None].to(disp)).argmax(1)[0].cpu()
            valores.append(miou(pred, alvo, classes))
    torch.save(modelo.state_dict(), "unet_treinada.pt")
    print(json.dumps({"modelo": nome, "dispositivo": disp, "iteracoes": it,
                      "tempo_s": round(time.perf_counter() - t0, 1), "perda_inicial": round(perdas[0], 4),
                      "perda_final": round(perdas[-1], 4),
                      "miou_val": round(float(np.nanmean(valores)), 4) if valores else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
