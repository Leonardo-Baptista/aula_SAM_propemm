# Treino de segmentação de instâncias (formato COCO)

Estrutura:

```
images/train/ ...   images/val/ ...
annotations/instances_train.json   annotations/instances_val.json
treinar_torchvision.py             (Mask R-CNN do torchvision, pronto para rodar)
```

`categories` com id de 1 a {{N}}, na ordem das classes do projeto. Cada anotação tem `segmentation`, `area` (pixels da máscara), `bbox` em XYWH e `iscrowd` 0. Objeto em vários pedaços: um polígono por pedaço, na mesma anotação. {{AVISO_FURO}}

Campos extras (ignorados pelos frameworks): `sam_escore_previsto`, `sam_origem`, `iou_referencia`.

## torchvision (Mask R-CNN v2)

```bash
pip install torch torchvision pycocotools
python treinar_torchvision.py --iteracoes 300
```

O script usa `maskrcnn_resnet50_fpn_v2` com pesos COCO (baixa cerca de 180 MB; `--sem-pretreino` para não baixar), troca as cabeças para {{N}} classes mais o fundo e treina. Salva `maskrcnn_treinado.pt`.

## Detectron2 (mask_rcnn_R_50_FPN_3x)

```python
import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

register_coco_instances("anota_train", {}, "annotations/instances_train.json", "images/train")
register_coco_instances("anota_val", {}, "annotations/instances_val.json", "images/val")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN, cfg.DATASETS.TEST = ("anota_train",), ("anota_val",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = {{N}}
cfg.INPUT.MASK_FORMAT = "bitmask"  # aceita polígonos e RLE (objetos com furo)
cfg.SOLVER.IMS_PER_BATCH, cfg.SOLVER.BASE_LR, cfg.SOLVER.MAX_ITER = 2, 0.00025, 1000
cfg.OUTPUT_DIR = "saida_detectron2"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
treinador = DefaultTrainer(cfg)
treinador.resume_or_load(resume=False)
treinador.train()
```

O Detectron2 não tem pacote oficial para Windows; use Linux ou Colab.

## MMDetection

No config do `mask-rcnn_r50_fpn_1x_coco.py`, aponte `data_root` para esta pasta, `ann_file='annotations/instances_train.json'`, `data_prefix=dict(img='images/train/')` (e o mesmo para val), defina `metainfo=dict(classes=(...))` com os nomes na ordem de `categories` e `num_classes={{N}}` nas cabeças `bbox_head` e `mask_head`.
