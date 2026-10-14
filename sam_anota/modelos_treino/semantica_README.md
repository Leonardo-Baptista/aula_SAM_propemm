# Treino de segmentação semântica (U-Net, DeepLabV3, SegFormer)

Estrutura:

```
images/train/ ...        images/val/ ...
masks/train/*.png        masks/val/*.png      (PNG de 8 bits, mesmo nome da imagem, extensão .png)
visualizacao/train|val/  só para olhar (cores de classes.json); não use no treino
classes.json             [{"indice": 0, "nome": "fundo", "cor": [0, 0, 0]}, ...]
treinar_unet.py          U-Net pronta para rodar
```

Valor do pixel na máscara = índice da classe: 0 é fundo, 1 em diante seguem a ordem de `classes.json` ({{N}} classes contando o fundo).

Sobreposição: os objetos são pintados na ordem em que foram aceitos; onde dois se sobrepõem, **o último aceito vence**.

## U-Net (segmentation_models_pytorch)

```bash
pip install torch segmentation-models-pytorch opencv-python-headless
python treinar_unet.py --iteracoes 500
```

O script usa `smp.Unet("resnet34", encoder_weights="imagenet", classes={{N}})` (`--sem-pretreino` para não baixar os pesos do encoder). Sem o `segmentation_models_pytorch`, cai numa U-Net mínima definida no próprio arquivo. Redimensiona para `--tamanho` (padrão 512, múltiplo de 32), treina com entropia cruzada e mede a mIoU em val.

Trecho mínimo:

```python
import segmentation_models_pytorch as smp, torch.nn.functional as F
modelo = smp.Unet("resnet34", encoder_weights="imagenet", classes={{N}})
perda = F.cross_entropy(modelo(imagens), mascaras)   # imagens (B,3,H,W), mascaras (B,H,W) inteiras
```

DeepLabV3 (torchvision) e SegFormer (transformers) leem o mesmo par imagem e máscara; só troque o modelo e use `num_labels={{N}}`.
