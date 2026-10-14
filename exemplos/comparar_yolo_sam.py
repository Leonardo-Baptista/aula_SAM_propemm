"""YOLO-seg contra SAM, lado a lado: a mesma caixa, duas máscaras.

AVISO DE LICENÇA: a ultralytics (YOLO) é AGPL-3.0. Ela NÃO é dependência do aplicativo
(MIT); este exemplo fica fora dele e a ultralytics é instalada à parte, de preferência
num venv separado:  python -m pip install -r requirements-exemplos.txt

Para cada imagem ou quadro de vídeo:
  esquerda  (A): a máscara que o próprio YOLO-seg devolve (retina_masks=True);
  direita   (B): a caixa do mesmo YOLO-seg vira prompt de caixa do SAM escolhido
                 (multimask; fica a candidata de maior escore previsto).
Com --pos, o mesmo pós-processamento é aplicado em A e em B: preenche buracos e remove
respingos menores que max(64 px, 1% da área da caixa), com componentes conexas do OpenCV.
Sem referência humana, a IoU entre A e B mede concordância entre métodos, não acurácia.

Uso:
    python exemplos/comparar_yolo_sam.py dados/fora_do_coco/canguru/imagem.jpg
    python exemplos/comparar_yolo_sam.py video.webm --inicio 5 --duracao 4 --qps 5 --largura 1280
    python exemplos/comparar_yolo_sam.py foto.jpg --yolo yolo11x-seg.pt --sam sam2.1_large --dispositivo cuda --pos

Saída: <entrada>_yolo_sam.png (imagem) ou .mp4 (vídeo), mais um .json com tempos e IoU A contra B
por objeto, ao lado da entrada ou em --saida. Os pesos do YOLO são baixados pela ultralytics
na primeira execução; os do SAM vêm de checkpoints/ (baixados na hora se faltarem).
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
EXT_VIDEO = {".mp4", ".webm", ".ogv", ".avi", ".mov", ".mkv", ".m4v"}
PIX_MIN, FRAC_CAIXA = 64, 0.01


def cor(i: int) -> np.ndarray:
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.8, 1.0)
    return np.array([r * 255, g * 255, b * 255])  # RGB


def pos_processar(m: np.ndarray, caixa) -> np.ndarray:
    """Preenche buracos e remove respingos menores que o limiar (igual em A e B)."""
    import cv2

    x0, y0, x1, y1 = caixa
    lim = max(PIX_MIN, FRAC_CAIXA * max(0.0, (x1 - x0) * (y1 - y0)))
    m = m.astype(np.uint8)
    _, rot, st, _ = cv2.connectedComponentsWithStats(1 - m, connectivity=8)
    peq = [k for k in np.where(st[:, cv2.CC_STAT_AREA] < lim)[0] if k > 0]
    if peq:
        m[np.isin(rot, peq) & (m == 0)] = 1
    _, rot, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    peq = [k for k in np.where(st[:, cv2.CC_STAT_AREA] < lim)[0] if k > 0]
    if peq:
        m[np.isin(rot, peq)] = 0
    return m.astype(bool)


def iou(a, b) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


class Comparador:
    def __init__(self, a):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("A ultralytics não está instalada. Ela é opcional e AGPL-3.0: "
                     "python -m pip install -r requirements-exemplos.txt (de preferência num venv separado).")
        from sam_anota.modelo import CATALOGO, ModeloSAM, baixar_checkpoint, resolver_chave

        self.a = a
        info = CATALOGO[resolver_chave(a.sam)]
        if not info.baixado:
            print(f"Baixando {info.rotulo} ({info.tamanho_bytes / 1e6:.0f} MB)...")
            baixar_checkpoint(info)
        self.sam = ModeloSAM(info.chave, dispositivo=a.dispositivo, cache_disco=False, cache_memoria=1)
        self.sam.carregar()
        self.disp = self.sam.dispositivo
        self.yolo = YOLO(a.yolo)
        self.rotulo_sam = info.rotulo
        print(f"YOLO {a.yolo}; {self.sam.descricao()}")

    def processar(self, rgb: np.ndarray, chave: str) -> dict:
        a = self.a
        t0 = time.perf_counter()
        r = self.yolo.predict(rgb[:, :, ::-1], conf=a.conf, classes=a.classes, retina_masks=True, verbose=False,
                              device=0 if self.disp == "cuda" else "cpu")[0]
        caixas = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)
        conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        h, w = rgb.shape[:2]
        if r.masks is not None and len(caixas):
            ma = r.masks.data.cpu().numpy() > 0.5
            if ma.shape[1:] != (h, w):  # garante a resolução da imagem
                import cv2
                ma = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for m in ma])
        else:
            ma = np.zeros((len(caixas), h, w), bool)
        t_yolo = time.perf_counter() - t0

        t0 = time.perf_counter()
        mb, escores = [], []
        if len(caixas):
            self.sam.codificar(chave, rgb)
            for cx in caixas:
                p = self.sam.prever(chave, None, tuple(float(v) for v in cx))
                k = int(np.argmax(p.escores))
                mb.append(p.mascaras[k])
                escores.append(float(p.escores[k]))
        mb = np.array(mb).reshape(-1, h, w)
        t_sam = time.perf_counter() - t0

        t_pos = 0.0
        if a.pos and len(caixas):
            t0 = time.perf_counter()
            ma = np.array([pos_processar(m, c) for m, c in zip(ma, caixas)])
            mb = np.array([pos_processar(m, c) for m, c in zip(mb, caixas)])
            t_pos = time.perf_counter() - t0
        objetos = [{"classe": r.names[int(c)], "conf_yolo": round(float(p), 3),
                    "caixa": [round(float(v), 1) for v in cx], "area_yolo": int(m1.sum()), "area_sam": int(m2.sum()),
                    "escore_sam": round(s, 3), "iou_yolo_sam": round(iou(m1, m2), 3)}
                   for c, p, cx, m1, m2, s in zip(cls, conf, caixas, ma, mb, escores)]
        return {"objetos": objetos, "ma": ma, "mb": mb, "caixas": caixas, "nomes": [r.names[int(c)] for c in cls],
                "tempos_s": {"yolo": round(t_yolo, 4), "sam": round(t_sam, 4), "pos": round(t_pos, 4)}}

    def desenhar(self, rgb, res, titulo_extra=""):
        from PIL import Image, ImageDraw, ImageFont

        lados = []
        for chave, titulo in (("ma", f"A) YOLO-seg ({Path(self.a.yolo).name})"),
                              ("mb", f"B) {self.rotulo_sam} a partir da caixa do YOLO")):
            out = rgb.astype(np.float32).copy()
            for k, m in enumerate(res[chave]):
                out[m] = out[m] * 0.45 + cor(k + 1) * 0.55
            im = Image.fromarray(out.clip(0, 255).astype(np.uint8))
            d = ImageDraw.Draw(im)
            tam = max(14, rgb.shape[1] // 60)
            fonte = _fonte(tam)
            usados = []
            for k, (cx, nome, ob) in enumerate(zip(res["caixas"], res["nomes"], res["objetos"])):
                d.rectangle([float(v) for v in cx], outline=tuple(int(v) for v in cor(k + 1)), width=2)
                x, y = float(cx[0]) + 3, float(cx[1]) + 2
                while any(abs(x - ux) < tam * 4 and abs(y - uy) < tam * 1.2 for ux, uy in usados):
                    y += tam * 1.25  # caixas quase iguais: empilha os rótulos
                usados.append((x, y))
                d.text((x, y), f"{nome} {ob['conf_yolo']:.2f}", fill=tuple(int(v) for v in cor(k + 1)), font=fonte,
                       stroke_width=2, stroke_fill=(0, 0, 0))
            faixa = Image.new("RGB", (im.width, int(tam * 2.2)), (20, 20, 20))
            ImageDraw.Draw(faixa).text((10, int(tam * 0.5)), titulo, fill=(255, 255, 255), font=_fonte(tam))
            junto = Image.new("RGB", (im.width, im.height + faixa.height))
            junto.paste(faixa, (0, 0))
            junto.paste(im, (0, faixa.height))
            lados.append(junto)
        largura = lados[0].width + lados[1].width + 8
        tela = Image.new("RGB", (largura, lados[0].height), (0, 0, 0))
        tela.paste(lados[0], (0, 0))
        tela.paste(lados[1], (lados[0].width + 8, 0))
        rodape = "  |  ".join(t for t in (titulo_extra, self.a.credito) if t)
        if rodape:
            tam = max(12, tela.width // 120)
            faixa = Image.new("RGB", (tela.width, int(tam * 2)), (20, 20, 20))
            ImageDraw.Draw(faixa).text((10, int(tam * 0.4)), rodape, fill=(220, 220, 220), font=_fonte(tam))
            final = Image.new("RGB", (tela.width, tela.height + faixa.height))
            final.paste(tela, (0, 0))
            final.paste(faixa, (0, tela.height))
            tela = final
        return np.asarray(tela)


def _fonte(tamanho: int):
    from PIL import ImageFont

    for nome in ("segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    return ImageFont.load_default(size=tamanho)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="YOLO-seg contra SAM a partir da caixa do YOLO, lado a lado.")
    ap.add_argument("entrada", type=Path, help="imagem ou vídeo")
    ap.add_argument("--yolo", default="yolo11n-seg.pt", help="pesos de segmentação da ultralytics (yolo11n-seg.pt, yolo11x-seg.pt, yolo26n-seg.pt)")
    ap.add_argument("--sam", default="sam2.1_tiny", help="modelo do catálogo: sam2.1_tiny, sam2.1_large, sam1_vit_b...")
    ap.add_argument("--dispositivo", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="ids COCO (0 pessoa, 1 bicicleta, 2 carro, 3 moto...)")
    ap.add_argument("--pos", action="store_true", help="mesmo pós-processamento (buracos e respingos) em A e B")
    ap.add_argument("--inicio", type=float, default=0.0, help="vídeo: segundo inicial")
    ap.add_argument("--duracao", type=float, default=5.0, help="vídeo: segundos a processar")
    ap.add_argument("--qps", type=float, default=5.0, help="vídeo: quadros por segundo processados")
    ap.add_argument("--largura", type=int, default=None, help="redimensiona a entrada para esta largura")
    ap.add_argument("--credito", default="", help="texto de crédito no rodapé (autor, licença do vídeo ou foto)")
    ap.add_argument("--saida", type=Path, default=None)
    a = ap.parse_args()

    import cv2

    if not a.entrada.exists():
        sys.exit(f"Entrada não encontrada: {a.entrada}")
    video = a.entrada.suffix.lower() in EXT_VIDEO
    saida = a.saida or a.entrada.with_name(a.entrada.stem + ("_yolo_sam.mp4" if video else "_yolo_sam.png"))
    comp = Comparador(a)

    def preparar(bgr):
        if a.largura and bgr.shape[1] != a.largura:
            h = int(round(bgr.shape[0] * a.largura / bgr.shape[1]))
            bgr = cv2.resize(bgr, (a.largura, h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(bgr[:, :, ::-1])

    registros = []
    if not video:
        from PIL import Image, ImageOps
        with Image.open(a.entrada) as im:
            rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"))
        rgb = preparar(rgb[:, :, ::-1])
        res = comp.processar(rgb, "imagem")
        Image.fromarray(comp.desenhar(rgb, res)).save(saida)
        registros.append({"quadro": 0, "objetos": res["objetos"], "tempos_s": res["tempos_s"]})
        t = res["tempos_s"]
        print(f"{len(res['objetos'])} objetos; YOLO {t['yolo']:.3f} s, SAM {t['sam']:.3f} s ({comp.disp}). Salvo em {saida}")
    else:
        cap = cv2.VideoCapture(str(a.entrada))
        if not cap.isOpened():
            sys.exit(f"Não consegui abrir o vídeo: {a.entrada} (confira se o OpenCV lê este formato)")
        qps_orig = cap.get(cv2.CAP_PROP_FPS) or 25.0
        passo = max(1, int(round(qps_orig / a.qps)))
        cap.set(cv2.CAP_PROP_POS_MSEC, a.inicio * 1000)
        ultimo = int(a.duracao * qps_orig)
        escritor = None
        for i in range(ultimo):
            ok, bgr = cap.read()
            if not ok:
                break
            if i % passo:
                continue
            rgb = preparar(bgr)
            res = comp.processar(rgb, f"q{i}")
            instante = a.inicio + i / qps_orig
            quadro = comp.desenhar(rgb, res, f"t = {instante:.1f} s")
            if escritor is None:
                hq, wq = quadro.shape[:2]
                escritor = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"mp4v"), qps_orig / passo, (wq, hq))
            escritor.write(np.ascontiguousarray(quadro[:, :, ::-1]))
            registros.append({"quadro": i, "t_s": round(instante, 2), "objetos": res["objetos"], "tempos_s": res["tempos_s"]})
            print(f"  t = {instante:5.1f} s: {len(res['objetos'])} objetos, YOLO {res['tempos_s']['yolo']:.3f} s, "
                  f"SAM {res['tempos_s']['sam']:.3f} s", flush=True)
        cap.release()
        if escritor is None:
            sys.exit("Nenhum quadro lido no trecho pedido.")
        escritor.release()
        print(f"{len(registros)} quadros ({comp.disp}). Salvo em {saida}")

    ious = [o["iou_yolo_sam"] for r in registros for o in r["objetos"]]
    resumo = {"entrada": str(a.entrada), "yolo": a.yolo, "sam": comp.rotulo_sam, "dispositivo": comp.disp,
              "pos_processamento": a.pos, "quadros": len(registros), "objetos": len(ious),
              "iou_yolo_sam_mediana": round(float(np.median(ious)), 3) if ious else None,
              "tempo_mediano_s": {k: round(float(np.median([r["tempos_s"][k] for r in registros])), 4)
                                  for k in ("yolo", "sam", "pos")},
              "observacao": "IoU entre métodos (concordância), sem referência humana",
              "registros": registros}
    destino = saida.with_suffix(".json")
    destino.write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"IoU YOLO contra SAM, mediana {resumo['iou_yolo_sam_mediana']} em {len(ious)} objetos. Registro em {destino}")


if __name__ == "__main__":
    main()
