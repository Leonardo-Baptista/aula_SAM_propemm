# Exemplos: YOLO e SAM

Dois scripts independentes do aplicativo. Eles usam a **ultralytics** (YOLO), que é **AGPL-3.0**: ela não é dependência do aplicativo (MIT) e é instalada à parte, só se você aceitar os termos da AGPL-3.0 para o seu uso. Recomendo um venv separado:

```powershell
py -3.12 -m venv .venv_yolo
.venv_yolo\Scripts\python scripts\instalar.py --sem-opcionais
.venv_yolo\Scripts\python -m pip install -r requirements-exemplos.txt
```

Os pesos do YOLO (`yolo11n-seg.pt`, `yolo11x-seg.pt`, `yolo26n-seg.pt`...) são baixados pela ultralytics na primeira execução, na pasta atual. Os do SAM vêm de `checkpoints/` e, se faltarem, `comparar_yolo_sam.py` os baixa com conferência de SHA-256.

| Script | O que faz |
| --- | --- |
| `comparar_yolo_sam.py` | Para cada imagem ou quadro de vídeo, mostra lado a lado **A)** a máscara do próprio YOLO-seg e **B)** a máscara do SAM a partir da caixa do mesmo YOLO (três candidatas, fica a de maior escore). `--pos` aplica o mesmo pós-processamento em A e em B (preenche buracos e remove respingos menores que max(64 px, 1% da caixa)). Grava PNG ou MP4 e um JSON com tempos e a IoU entre A e B por objeto (concordância entre métodos, não acurácia). Aceita qualquer SAM do catálogo do aplicativo (`--sam sam2.1_tiny`, `sam2.1_large`, `sam1_vit_b`...) |
| `yolo_como_prompt.py` | YOLO como gerador de prompts do SAM 2.1: cada caixa do detector vira prompt de caixa; desenha as máscaras com classe e confiança, em imagem ou vídeo |

Exemplos:

```powershell
# imagem fora do COCO: o YOLO erra o nome, o SAM não precisa dele
.venv_yolo\Scripts\python exemplos\comparar_yolo_sam.py dados\fora_do_coco\canguru\imagem.jpg --pos

# 3 s de vídeo, 5 quadros por segundo, só veículos (1 bicicleta, 2 carro, 3 moto, 5 ônibus, 7 caminhão)
.venv_yolo\Scripts\python exemplos\comparar_yolo_sam.py traffic_in_hyderabad.webm --inicio 10 --duracao 3 --qps 5 --largura 1280 --classes 1 2 3 5 7 --pos --credito "Traffic in Hyderabad, Oleg Yunakov, Wikimedia Commons, CC BY-SA 4.0"

.venv_yolo\Scripts\python exemplos\yolo_como_prompt.py dados\exemplos\rua_moto_carro_pessoas.jpg --yolo yolo11n.pt --modelo-sam tiny
```

Testado em 14/10/2026 (RTX 3060, ultralytics 8.4.177, torch 2.6.0+cu124):

| Teste | Resultado |
| --- | --- |
| canguru, YOLO11n-seg e SAM 2.1 tiny, GPU, `--pos` | 2 detecções (ovelha e cão, para o mesmo canguru); IoU A contra B mediana 0,90 |
| lingote, YOLO11n-seg e SAM 2.1 tiny, CPU | 2 objetos; YOLO 0,52 s e SAM 2,1 s na imagem de 1800x1870 |
| Traffic in Hyderabad, 3 s a 5 quadros/s, 1280 px, YOLO11n-seg e SAM 2.1 large, GPU, `--pos` | 15 quadros, 45 objetos; IoU A contra B mediana 0,74; por quadro, YOLO 0,03 s e SAM 0,25 s |
| `yolo_como_prompt.py`, rua, YOLO11n-seg e SAM 2.1 tiny, GPU | 11 objetos |

## Vídeos usados na aula

Os vídeos **não** estão no repositório (tamanho). Baixe pela página do Wikimedia Commons (link "Download" ou "Original file") e respeite a licença: CC BY-SA 4.0 exige crédito ao autor, link para a licença e a mesma licença em obras derivadas (o `--credito` grava o crédito no rodapé do vídeo gerado).

| Vídeo | Autor | Licença | Página |
| --- | --- | --- | --- |
| Traffic in Hyderabad (27/01/2025, 3840x2160, 56 s) | Oleg Yunakov | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | https://commons.wikimedia.org/wiki/File:Traffic_in_Hyderabad.webm |
| Coureurs cyclistes (Colmar) (10/07/2019, 1920x1080, 10,6 s) | Gzen92 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | https://commons.wikimedia.org/wiki/File:Coureurs_cyclistes_(Colmar).ogv |

O OpenCV do pip leu os dois arquivos no Windows (`.webm` de Hyderabad e `.ogv` de Colmar, testados). Se o seu não ler, converta com ffmpeg: `ffmpeg -i entrada.webm -vf scale=1280:-2 saida.mp4`.
