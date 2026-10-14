# Por que estas imagens são difíceis para o SAM

O SAM aprendeu, com 1 bilhão de máscaras, que "objeto" é uma região com borda
nítida, textura coerente e forma compacta. Cada pasta aqui quebra uma dessas
premissas.

| Tipo | Por que o SAM sofre (uma linha) |
| --- | --- |
| `fumaca/` | Fumaça é semitransparente e não tem borda: a opacidade cai aos poucos até virar céu, nuvem ou vapor, então não existe um contorno "verdadeiro" para o SAM encaixar, e o modelo tende a pegar só o miolo denso, ou a nuvem e o vapor junto. |
| `trinca/` | Trinca é um objeto de 1 a 2 px de largura e centenas de px de comprimento; o SAM trabalha numa grade de 64x64 tokens (cada um vê 16x16 px), então a trinca cabe dentro de um único token e some, ou a máscara "engorda" e vaza para manchas vizinhas. |
| `microestrutura/` | As ilhas M-A têm o mesmo cinza e a mesma textura da matriz bainítica; sem diferença de textura entre alvo e fundo, o SAM ou devolve a imagem inteira, ou recorta pedaços de relevo que não são fase nenhuma. |

Pontos para discutir em aula:

* Na fumaça do IJmond, o vapor d'água branco NÃO é alvo. Um clique na fumaça
  que "vaza" para o vapor é um erro de conceito (o que é emissão?), não só de
  pixel. O SAM não sabe a diferença; o especialista sabe.
* A própria referência do IJmond e do Boreal Forest Fire foi feita com ajuda do
  SAM e depois corrigida à mão; no Boreal, os autores só confiaram em máscaras
  desenhadas à mão para o conjunto de teste. Ou seja: quem montou os dados já
  tratou a saída do SAM como rascunho.
* Na microestrutura, a referência é parcial (só algumas ilhas foram
  contornadas). Bom gancho para discutir o custo de anotar e o que é
  "verdade de referência" em metalografia.

## Referências da literatura

* Zhang, Konz, Kramer, Mazurowski. "Quantifying the Limits of Segmentation
  Foundation Models: Modeling Challenges in Segmenting Tree-Like and
  Low-Contrast Objects", WACV 2026 (arquivo
  `aula_sam_doutorado/sam_pacote/01_artigos/SAM_limites_WACV_2026.pdf`).
  Mostra que o desempenho de SAM, SAM 2 e HQ-SAM cai com a "arborescência" do
  objeto e com a baixa separabilidade de textura entre objeto e fundo, e que
  ajuste fino não resolve. Base direta para `trinca/` e `microestrutura/`.
* Ji, G.-P. et al. "SAM Struggles in Concealed Scenes: Empirical Study on
  Segment Anything", 2023, https://arxiv.org/abs/2304.06022. Avalia o SAM sem
  prompt em três cenas "escondidas" (animais camuflados, defeitos industriais e
  lesões médicas) e conclui que o SAM "parece inábil" nelas.
* Ji, W. et al. "Segment Anything Is Not Always Perfect: An Investigation of SAM
  on Different Real-world Applications", 2023, https://arxiv.org/abs/2304.05750.
  Testa o SAM em várias aplicações reais (imagens naturais, agricultura,
  manufatura, sensoriamento remoto, saúde) e discute, caso a caso, onde ele
  funciona e onde falha; bom panorama de que o "anything" tem limites.
* Tang, Xiao, Li. "Can SAM Segment Anything? When SAM Meets Camouflaged Object
  Detection", 2023, https://arxiv.org/abs/2304.04709. Mostra que o SAM, sem
  ajuste, fica atrás dos modelos de detecção de camuflagem nos benchmarks CAMO,
  COD10K e NC4K.
* Han, Zhang, Qiao et al. "Segment Anything Model (SAM) Meets Glass: Mirror and
  Transparent Objects Cannot Be Easily Detected", 2023,
  https://arxiv.org/abs/2305.00278. Mostra que o SAM frequentemente não
  detecta vidro e espelhos; a fumaça compartilha o problema da transparência.
* Yan, Zhang, Barnes. "Transmission-Guided Bayesian Generative Model for Smoke
  Segmentation", AAAI 2023, https://arxiv.org/abs/2303.00900. Argumenta que a
  fumaça é não rígida e transparente, parecida com objetos transparentes, e que
  redes profundas ficam superconfiantes nela (não avalia o SAM).
* Hsu, Touska. "IJmond Industrial Smoke Segmentation Dataset", 2026,
  https://arxiv.org/abs/2603.23754. Descreve o fluxo de anotação: proposta do
  SAM por cliques e refinamento manual; discute que a opacidade percebida
  depende da iluminação.
* Pesonen et al. "Boreal Forest Fire: UAV-collected Wildfire Detection and Smoke
  Segmentation Dataset", Scientific Data, 2025,
  https://doi.org/10.1038/s41597-025-05634-0. Treino e validação com máscaras
  geradas pelo SAM a partir de caixas; teste com máscaras manuais.
* Revisão sobre detecção de trincas, 2025, https://arxiv.org/abs/2508.10256.
  Relata que a segmentação de trincas sem treino ainda é limitada, com
  "undershoot" (perde a trinca fina) e "overshoot" (marca regiões parecidas que
  não são trinca).
