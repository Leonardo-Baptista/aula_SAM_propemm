# Instruções para assistentes de código

Este arquivo é lido automaticamente pelos assistentes de IA dos editores que seguem a convenção AGENTS.md (ver `docs/EDITORES.md`). Vale também como resumo técnico para quem chega ao repositório.

## O repositório

Material da aula de SAM (Segment Anything Model), PMM.051, PROPEMM-IFES. O centro é um aplicativo desktop de anotação (`sam_anota/`, CustomTkinter) usado por alunos em notebooks comuns, quase sempre Windows e sem GPU. Ponto de entrada: `python -m sam_anota`.

| Caminho | Papel |
| --- | --- |
| `sam_anota/app.py` | interface. Toda chamada a modelo vai para uma thread única (`ThreadPoolExecutor(max_workers=1)`) e volta por uma fila lida a cada 30 ms. Download de pesos e exportação rodam em threads próprias e também respondem pela fila. Nunca toque em widget fora da thread principal. Tarefas carregam o objeto do modelo e o resultado é descartado se o modelo mudou |
| `sam_anota/modelo.py` | `CATALOGO` (SAM 2.1 tiny, small, base+, large; SAM 1 ViT-B, L, H: URL, SHA-256, tamanho, parâmetros), `baixar_checkpoint` (SHA conferido no download), `escolher_dispositivo` (auto, cpu, cuda, mps; GPU indisponível volta para CPU com `diagnosticar_cuda`), cache de embeddings (memória e disco, chave com o modelo), `prever` com 3 candidatas |
| `sam_anota/texto.py` | modo texto: Grounding DINO tiny + SAM (caixas viram prompts) e SAM 3 via transformers (pesos com acesso controlado; nunca contornar) |
| `sam_anota/projeto.py` | estado em `saidas/<pasta>/projeto.json` (máscaras em RLE próprio, ordem C; `origem` diz de qual modelo veio cada máscara); `PALETA` de cores |
| `sam_anota/exportar.py`, `sam_anota/modelos_treino/` | exportação simples e para treino (YOLO-seg, COCO de instâncias, semântica PNG, divisão por imagem com semente), README_TREINO e scripts de treino copiados para a exportação |
| `sam_anota/dialogos.py`, `sam_anota/metricas.py` | diálogo de exportação; IoU, leitura de `<nome>_ref.png`, relatório por sessão |
| `scripts/` | `instalar.py`, `obter_checkpoint.py`, `verificar.py`, `teste_interface.py` (ponta a ponta com a janela), `medir_modelos.py`, `prova_treino.py` |
| `exemplos/` | YOLO e SAM, fora do aplicativo (ultralytics é AGPL-3.0, opcional) |
| `notebooks/` | `01_sam1_guiado.ipynb`, `02_treinar_no_colab.ipynb` (badges do Colab) |
| `dados/` | imagens da aula, cada pasta com `FONTES.md`; só leitura |
| `docs/` | `INSTALACAO.md`, `EDITORES.md`, captura de tela |

## Convenções

* Português do Brasil em nomes, comentários, interface e documentação.
* Proibido travessão, meia-risca e sinal de menos tipográfico em qualquer arquivo. Use dois pontos, vírgula, parênteses ou hífen comum.
* Nada é escrito em `dados/`. Saídas vão para `saidas/` (no `.gitignore`), configurável por `SAM_ANOTA_SAIDAS`.
* Pesos ficam em `checkpoints/` (fora do git). Modelo novo no catálogo exige URL oficial e SHA-256 conferido.
* O sam2 é instalado do GitHub oficial, commit fixo `2b90b9f`, com `SAM2_BUILD_CUDA=0` e `--no-build-isolation`. A ultralytics (AGPL-3.0) não pode ser dependência do aplicativo; só aparece em `exemplos/`, em `requirements-exemplos.txt` e nas instruções de treino.
* SAM 1 (`segment-anything`) e modo texto (`transformers`) são opcionais: o app tem de abrir e explicar o que falta quando não estão instalados.
* Versões fixadas em `requirements*.txt` e em `scripts/instalar.py` (torch); mude só depois de testar a instalação num venv novo.
* Toda imagem nova precisa de fonte e licença registradas no `FONTES.md` da pasta. Nada de imagem de cliente ou de arquivo maior que 20 MB.
* Números mostrados ao usuário usam vírgula decimal.

## Como rodar e testar

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python scripts\instalar.py --cpu --opcionais
.venv\Scripts\python -m pytest -q                 # rápido, sem o SAM
.venv\Scripts\python scripts\verificar.py         # carrega o SAM e mede tempos, sem janela
.venv\Scripts\python scripts\teste_interface.py   # abre a janela, simula cliques, troca modelo e dispositivo, modo texto, exporta
```

Em Linux e macOS, troque `.venv\Scripts\python` por `.venv/bin/python`. Ao mudar a interface, rode `scripts/teste_interface.py` e, se o visual mudar, refaça `docs/tela.png` com `--captura docs\tela.png`.
