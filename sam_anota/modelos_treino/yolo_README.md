# Treino YOLO-seg (Ultralytics YOLO11 e YOLO26)

Estrutura (padrão Ultralytics):

```
data.yaml            path: . | train: images/train | val: images/val | names (na ordem das classes)
images/train/ ...    images/val/ ...
labels/train/*.txt   labels/val/*.txt   (um .txt por imagem, mesmo nome)
```

Cada linha de `labels/*.txt`: `classe x1 y1 x2 y2 ...`, coordenadas normalizadas de 0 a 1, um objeto por linha. Imagem sem objeto tem `.txt` vazio (exemplo negativo). {{N}} classes.

Objeto em vários pedaços: os pedaços são costurados num polígono só, ligando os pontos mais próximos por pontes de largura zero (uma linha por objeto, que é o que a Ultralytics espera para uma instância). Furos internos não existem neste formato.

## Treinar

Rode **de dentro desta pasta** (o `path: .` do `data.yaml` é a pasta atual):

```bash
pip install ultralytics
yolo segment train data=data.yaml model=yolo11n-seg.pt epochs=50 imgsz=640
# YOLO26 (ultralytics 8.4 ou mais nova):
yolo segment train data=data.yaml model=yolo26n-seg.pt epochs=50 imgsz=640
```

Em Python:

```python
from ultralytics import YOLO
YOLO("yolo11n-seg.pt").train(data="data.yaml", epochs=50, imgsz=640)
```

Os pesos treinados ficam em `runs/segment/train*/weights/best.pt`.

Licença: a Ultralytics é AGPL 3.0. Por isso ela não é dependência do sam-anota (que é MIT); instale no seu ambiente de treino.
