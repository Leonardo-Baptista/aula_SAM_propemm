# Fontes das imagens difíceis

Dez imagens, cada uma com a máscara de referência feita por humanos
(`NN_nome_ref.png`, binária 0/255, mesmo tamanho da imagem). Todas vêm de
conjuntos de dados públicos com licença que permite redistribuição (CC BY 4.0
ou CC0). Coletadas em 13/10/2026.

Regras gerais de processamento, válidas para todas as fichas:

* Imagem com mais de 1600 px no maior lado foi reduzida para 1600 px
  (reamostragem Lanczos); a máscara foi reduzida para o mesmo tamanho com
  vizinho mais próximo, para continuar binária.
* Imagem salva em JPEG qualidade 95; máscara salva em PNG de 8 bits com
  valores 0 (fundo) e 255 (alvo).
* Nenhum outro recorte, filtro ou ajuste de cor foi aplicado.

## Licenças usadas

* CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ (exige crédito,
  link para a licença e indicação das alterações; as alterações estão em cada
  ficha abaixo).
* CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/ (domínio
  público; o crédito abaixo é cortesia, não obrigação).

---

## Fumaça e emissões: `fumaca/`

### Conjunto IJmond Industrial Smoke Segmentation Dataset (fichas 01 a 04)

* Página do conjunto: https://uvaauas.figshare.com/articles/dataset/IJmond_Industrial_Smoke_Segmentation_Dataset/31847188
  (DOI 10.21942/uva.31847188)
* Autores: Yen-Chia Hsu e Despoina Touska (Universidade de Amsterdã).
  Relatório técnico: "IJmond Industrial Smoke Segmentation Dataset",
  arXiv 2603.23754, https://arxiv.org/abs/2603.23754
* Origem das fotos: câmeras fixas da organização de qualidade do ar
  FrisseWind.nu, apontadas para as coquerias ("Kooks Fabriek 1" e
  "Kooks Fabriek 2") do polo siderúrgico de IJmond, Holanda. A marca d'água
  "Frisse Wind.Nu" e o carimbo de data das câmeras fazem parte das imagens
  originais e foram mantidos.
* Licença: CC BY 4.0 (declarada no registro do figshare e no README do
  pacote), https://creativecommons.org/licenses/by/4.0/
* Como a referência foi feita: polígonos traçados na plataforma Roboflow; o
  SAM gerou a proposta inicial a partir de cliques e os autores refinaram o
  contorno à mão; o primeiro autor tem treinamento EPA Method 9 de leitura de
  opacidade de fumaça e revisou todas as máscaras.
* Binarização: a máscara original tem três valores: 0 (fundo), 155 (fumaça de
  baixa opacidade) e 255 (fumaça de alta opacidade). Aqui, tudo que é maior
  que 0 virou 255, ou seja, a referência inclui a fumaça tênue. Atenção: vapor
  d'água branco NÃO é rotulado como fumaça neste conjunto.
* Arquivo bruto: `ijmond_seg.tar.gz` (254 MB), pasta `test/images` e
  `test/masks`.

| Arquivo | Arquivo original | Alterações | Dificuldade |
| --- | --- | --- | --- |
| `01_ijmond_coqueria1_pluma_ceu_nublado` | `kooks_1__2024-10-31T06-51-46Z_frame_2817_jpg.rf.3f217a002c93c2802ab0fc88f4a1e2f9` | 1920x1080 para 1600x900; máscara 0/155/255 binarizada | pluma cinza longa que se dissolve num céu nublado também cinza; a turbina eólica atravessa a pluma |
| `02_ijmond_coqueria2_pluma_chamine` | `kooks_2__2024-07-18T18-15-03Z_frame_1230_jpg.rf.26af85ca41961697207aeef86ae2d383` | idem | pluma quase transparente entre duas chaminés; vapor branco no solo não faz parte do alvo |
| `03_ijmond_coqueria2_fumaca_no_vapor` | `kooks_2__2024-10-29T10-19-43Z_frame_1926_jpg.rf.3afa4772085a9700f9b69be0398af40e` | idem | fumaça escura misturada a nuvens de vapor branco, que NÃO são alvo; fumaça e vapor se tocam |
| `04_ijmond_coqueria2_fumaca_rasteira` | `kooks_2__2024-08-17T08-32-34Z_frame_3196_jpg.rf.dbe1ecd6fab24cc432f980470b51c8cc` | idem (só havia o tom 155: toda a fumaça é de baixa opacidade) | fumaça tênue, quase invisível, na frente de chaminé e prédios |

### Conjunto Boreal Forest Fire: UAV-collected Wildfire Detection and Smoke Segmentation Dataset (fichas 05 e 06)

* Página do conjunto: https://etsin.fairdata.fi/dataset/1dce1023-493a-4d63-a906-f2a44f831898
  (DOI 10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9)
* Autores: J. Pesonen, A.-M. Raita-Hakola, J. Joutsalainen, T. Hakala,
  W. Akhtar, N. Koivumäki, L. Markelin, J. Suomalainen, R. Alves de Oliveira,
  I. Pölönen, E. Honkavaara. Artigo: Scientific Data 12, 1419 (2025),
  https://doi.org/10.1038/s41597-025-05634-0
* Licença: CC BY 4.0 (campo de licença do registro Fairdata),
  https://creativecommons.org/licenses/by/4.0/
* Como a referência foi feita: foram usadas apenas máscaras do subconjunto
  `Subset-C/manual_masks/test`, desenhadas à mão pixel a pixel. As máscaras
  de treino e validação do mesmo conjunto foram geradas pelo SAM e NÃO foram
  usadas.
* Binarização: a máscara original já é 0/255; nada a fazer.

| Arquivo | Arquivo original | Alterações | Dificuldade |
| --- | --- | --- | --- |
| `05_boreal_fumaca_nuvens` | `Subset-C/images/test/evoDJI_0009_frame132.jpg` | 2048x1080 para 1600x844 (máscara com vizinho mais próximo) | a fumaça do incêndio sobe e se funde com nuvens cúmulo de mesma cor; a borda superior não existe visualmente |
| `06_boreal_fumaca_lago` | `Subset-C/images/test/heinola_DJI_0033_frame157.jpg` | idem | fumaça tênue contra céu claro e horizonte enevoado; a parte alta da pluma esmaece até sumir |

---

## Estruturas finas: `trinca/`

### Conjunto Complex Tunnel Crack Dataset (CTCD) (fichas 07 e 08)

* Página do conjunto: https://huggingface.co/datasets/shiweiluo99/tunnel-crack-segmentation-dataset
* Autores: Shiwei Luo, Xiongyao Xie, Biao Zhou, Kun Zeng, Jun Guo. Artigo:
  "Multiscenario Generalization Crack Detection Network Based on the Visual
  Foundation Model", Structural Control and Health Monitoring (2025),
  https://doi.org/10.1155/stc/6269747
* Origem: revestimento de 52 túneis rodoviários (Yunnan e Zhejiang, China),
  veículo de inspeção de alta velocidade.
* Licença: CC BY 4.0 ("Images and annotations by the authors", README do
  repositório), https://creativecommons.org/licenses/by/4.0/
* Como a referência foi feita: pixels de trinca pintados à mão no Photoshop
  com pincel de 1 a 2 px ajustado à largura da trinca.
* Binarização: a máscara original já é 0/255; nada a fazer.

| Arquivo | Arquivo original | Alterações | Dificuldade |
| --- | --- | --- | --- |
| `07_ctcd_trinca_mancha` | `val/170-3.bmp` e `valannot/170-3.bmp` | BMP em tons de cinza convertido para JPEG; tamanho mantido (256x256) | trinca de 1 a 2 px ao lado de uma mancha escura radial que chama mais atenção que a própria trinca |
| `08_ctcd_trinca_textura` | `val/58-5.bmp` e `valannot/58-5.bmp` | idem | trinca fina e sinuosa passando por uma região com bolhas e microfissuras que não são alvo |

---

## Baixo contraste em materiais: `microestrutura/`

### Conjunto Aachen-Heerlen Annotated Steel Microstructure Dataset (fichas 09 e 10)

* Página do conjunto: https://doi.org/10.6084/m9.figshare.c.5185004
  * Imagens PNG: https://springernature.figshare.com/articles/dataset/Steel_microscopy_images_PNG_/13135625
  * Polígonos de especialistas: https://springernature.figshare.com/articles/dataset/Expert_annotated_polygons/13135610
* Autores (como no registro do figshare): Deniz Iren, Marc Ackerman, Julian
  Gorfer, Sebastian Wesselmecking, Ulrich Krupp, Stefano Bromuri, Gaurav
  Pujar. Artigo: Iren et al., "Aachen-Heerlen annotated steel microstructure
  dataset", Scientific Data (2021), https://doi.org/10.1038/s41597-021-00926-7
* Origem: MEV (4000x) de aço bainítico; o alvo são as ilhas de
  martensita-austenita (M-A).
* Licença: CC0 1.0 (imagens e polígonos, conforme os registros do figshare),
  https://creativecommons.org/publicdomain/zero/1.0/
* Como a referência foi feita: polígonos traçados por especialistas a partir
  de pontos de interesse. A máscara foi rasterizada aqui a partir do CSV
  `nature_scidata_heerlen_aachen_steel_annotations_polygon.csv` (preenchimento
  de polígono do Pillow, coordenadas na resolução 1024x768 da imagem PNG).
* ATENÇÃO, referência PARCIAL: os especialistas contornaram apenas as ilhas
  M-A em torno dos pontos de interesse, não todas as ilhas da imagem. Use estas
  imagens para clicar numa ilha marcada e comparar o contorno, e não para
  medir IoU da imagem inteira.
* Download: as 2 imagens foram extraídas individualmente do arquivo
  `PNG.zip` (1,2 GB) por requisições HTTP de intervalo (cerca de 1,5 MB
  transferidos), sem baixar o arquivo inteiro.

| Arquivo | Arquivo original | Alterações | Dificuldade |
| --- | --- | --- | --- |
| `09_aachen_ilhas_ma_bainita` | `PNG/IMG_02289.png` | PNG com paleta convertido para JPEG em tons de cinza; 1024x768 mantido; máscara rasterizada dos polígonos | ilhas M-A com o mesmo cinza e o mesmo relevo da matriz bainítica; a borda só aparece como um leve degrau de sombra |
| `10_aachen_ilhas_ma_baixo_contraste` | `PNG/IMG_02431.png` | idem | ainda menos contraste: a textura de ripas da bainita domina a imagem e as ilhas quase não se destacam |
