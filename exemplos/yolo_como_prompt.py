"""Exemplo autocontido: YOLO como gerador de prompts do SAM 2.1.

AVISO DE LICENCA: a biblioteca ultralytics (YOLO) e AGPL-3.0. Ela NAO e dependencia do
sam-anota (que e MIT) e nao deve virar uma. Este exemplo fica fora do aplicativo, e a
ultralytics e uma dependencia opcional, instalada a parte, de preferencia num venv separado:

    python -m pip install ultralytics

Ideia: o detector diz O QUE e ONDE (classe e caixa); cada caixa vira prompt de caixa do
SAM 2.1, que devolve o CONTORNO. Em video, o SAM roda quadro a quadro (set_image a cada
quadro), sem memoria entre quadros; a classe vem do YOLO em cada quadro.

Uso:
    python exemplos/yolo_como_prompt.py imagem.jpg
    python exemplos/yolo_como_prompt.py video.mp4 --max-quadros 100 --passo 2
    python exemplos/yolo_como_prompt.py foto.jpg --classes 0 1 --conf 0.35 --modelo-sam tiny

Saida: <entrada>_yolo_sam.png (imagem) ou <entrada>_yolo_sam.mp4 (video), ao lado da
entrada, ou no caminho dado em --saida. Os pesos do YOLO sao baixados pela ultralytics na
primeira execucao; os do SAM 2.1 sao lidos de checkpoints/ (baixe pelo aplicativo).
"""
from __future__ import annotations

import argparse
import colorsys
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
CONFIGS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogv"}


def cor(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR, para o OpenCV


def carregar(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("A ultralytics nao esta instalada. Ela e opcional e AGPL-3.0: "
                 "python -m pip install ultralytics (de preferencia num venv separado).")
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    disp = args.dispositivo
    if disp == "auto":
        disp = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, arq = CONFIGS[args.modelo_sam]
    ck = Path(args.checkpoint) if args.checkpoint else RAIZ / "checkpoints" / arq
    if not ck.exists():
        sys.exit(f"Checkpoint do SAM nao encontrado: {ck}")
    yolo = YOLO(args.yolo)
    sam = SAM2ImagePredictor(build_sam2(cfg, str(ck), device=disp))
    return yolo, sam, disp


def segmentar(yolo, sam, disp, bgr, args):
    """Devolve lista de (classe, conf, caixa xyxy, mascara bool) e os tempos (yolo, sam)."""
    import torch

    t0 = time.perf_counter()
    r = yolo.predict(bgr, conf=args.conf, classes=args.classes, verbose=False,
                     device=0 if disp == "cuda" else "cpu")[0]
    caixas = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    t1 = time.perf_counter()
    if len(caixas) == 0:
        return [], r.names, (t1 - t0, 0.0)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if disp == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), ctx:
        sam.set_image(rgb)
        m, _, _ = sam.predict(box=caixas, multimask_output=False)
    if m.ndim == 3:  # uma caixa so: (1, H, W)
        m = m[None]
    mascaras = m[:, 0] > 0
    t2 = time.perf_counter()
    return list(zip(cls, conf, caixas, mascaras)), r.names, (t1 - t0, t2 - t1)


def desenhar(bgr, itens, nomes):
    import cv2

    out = bgr.astype(np.float32)
    for k, (_, _, _, m) in enumerate(itens):
        out[m] = out[m] * 0.45 + np.array(cor(k + 1)) * 0.55
    out = out.clip(0, 255).astype(np.uint8)
    for k, (c, p, cx, m) in enumerate(itens):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, cor(k + 1), 2)
        x0, y0, x1, y1 = (int(v) for v in cx)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 1)
        txt = f"{nomes[int(c)]} {p:.2f}"
        cv2.putText(out, txt, (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor(k + 1), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="YOLO (caixas) como prompt do SAM 2.1, em imagem ou video.")
    ap.add_argument("entrada", type=Path)
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--yolo", default="yolo11x.pt", help="pesos de deteccao da ultralytics (padrao: yolo11x.pt)")
    ap.add_argument("--modelo-sam", choices=list(CONFIGS), default="large")
    ap.add_argument("--checkpoint", default=None, help="caminho do checkpoint do SAM 2.1 (padrao: checkpoints/)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="ids COCO (0 pessoa, 1 bicicleta, 2 carro...)")
    ap.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--max-quadros", type=int, default=None, help="video: para depois de N quadros processados")
    ap.add_argument("--passo", type=int, default=1, help="video: processa 1 a cada PASSO quadros")
    args = ap.parse_args()

    import cv2

    yolo, sam, disp = carregar(args)
    video = args.entrada.suffix.lower() in EXT_VIDEO
    saida = args.saida or args.entrada.with_name(args.entrada.stem + ("_yolo_sam.mp4" if video else "_yolo_sam.png"))

    if not video:
        bgr = cv2.imread(str(args.entrada))
        if bgr is None:
            sys.exit(f"Nao consegui ler a imagem: {args.entrada}")
        itens, nomes, (ty, ts) = segmentar(yolo, sam, disp, bgr, args)
        cv2.imwrite(str(saida), desenhar(bgr, itens, nomes))
        print(f"{len(itens)} objetos; YOLO {ty * 1000:.0f} ms, SAM {ts * 1000:.0f} ms ({disp}). Salvo em {saida}")
        return

    cap = cv2.VideoCapture(str(args.entrada))
    if not cap.isOpened():
        sys.exit(f"Nao consegui abrir o video: {args.entrada}")
    fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) / args.passo
    escritor = None
    i = n = 0
    tempos = []
    while True:
        ok, bgr = cap.read()
        if not ok or (args.max_quadros and n >= args.max_quadros):
            break
        if i % args.passo == 0:
            itens, nomes, t = segmentar(yolo, sam, disp, bgr, args)
            tempos.append(sum(t))
            quadro = desenhar(bgr, itens, nomes)
            if escritor is None:
                h, w = quadro.shape[:2]
                escritor = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            escritor.write(quadro)
            n += 1
            if n % 20 == 0:
                print(f"  {n} quadros, {np.median(tempos) * 1000:.0f} ms por quadro (mediana)", flush=True)
        i += 1
    cap.release()
    if escritor is not None:
        escritor.release()
    if not tempos:
        sys.exit("Nenhum quadro lido.")
    print(f"{n} quadros; mediana {np.median(tempos) * 1000:.0f} ms por quadro ({disp}). Salvo em {saida}")


if __name__ == "__main__":
    main()
