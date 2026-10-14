# Imagens fora do COCO

Duas imagens cujo objeto principal não está entre as 80 classes do COCO, usadas na aula para mostrar a diferença entre um segmentador de classes fechadas (YOLO-seg treinado no COCO) e o SAM, que segmenta sem saber o nome do objeto. Teste com `exemplos/comparar_yolo_sam.py`.

| Pasta | Imagem | Autor e acervo | Licença | Página original |
| --- | --- | --- | --- | --- |
| `canguru/` | 1600x1071, reduzida do original de 3872x2592 (PIL LANCZOS, JPEG 95, sem recorte) | Mark Wagner (User:Carnildo), retoque sobre foto de User:Lilly M | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [File:Kangaroo Australia 01 11 2008 - retouch.JPG](https://commons.wikimedia.org/wiki/File:Kangaroo_Australia_01_11_2008_-_retouch.JPG) |
| `lingote/` | 1800x1870, sem alteração | Underwood & Underwood, c. 1895; Science History Institute, via DPLA | domínio público nos Estados Unidos | [File:Steel ingot entering first rolling mill...](https://commons.wikimedia.org/wiki/File:Steel_ingot_entering_first_rolling_mill_to_be_rolled_into_plates_or_rails_-_DPLA_-_4fda4165573a72d02aff04d564f68e6b.jpg) |

Crédito sugerido:

* canguru: "Foto: Mark Wagner, a partir de imagem de Lilly M, Wikimedia Commons, CC BY-SA 3.0". Obras derivadas devem manter a CC BY-SA 3.0.
* lingote: "Underwood & Underwood, c. 1895, Science History Institute, Wikimedia Commons, domínio público".

Detalhes (hashes, tamanhos, data de download) em `fonte.json` de cada pasta.

O que se viu na aula (YOLO11 treinado no COCO): o canguru sai com o contorno certo e o nome errado (ovelha, cão); no lingote, o YOLO11x-seg chamou o lingote de `suitcase` e o YOLO11n-seg não o achou, enquanto o SAM 2.1 o segmentou com um clique (escore previsto 0,96).
