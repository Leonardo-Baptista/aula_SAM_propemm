# Instalação

Guia completo. O caminho curto está no README; aqui estão os requisitos, as variações por sistema, a instalação manual e os erros comuns.

## Requisitos

| Item | Detalhe |
| --- | --- |
| Python | **3.12**, de [python.org](https://www.python.org/downloads/). Testado com 3.12.10. Outras versões (3.10, 3.11, 3.13) não foram testadas |
| git | [git-scm.com](https://git-scm.com/downloads). Necessário para clonar e para instalar o sam2 do GitHub |
| Disco | cerca de 2,5 GB na instalação CPU (venv com os opcionais 1,3 GB, pesos do SAM 2.1 tiny e do SAM 1 ViT-B 0,5 GB, Grounding DINO 0,7 GB no cache do Hugging Face); cerca de 6,5 GB com GPU (o venv com torch CUDA ocupa 5,2 GB). Os pesos maiores somam mais 5 GB se você baixar todos |
| Internet | na instalação e na primeira execução de cada recurso: pesos do SAM (156 MB o tiny), Grounding DINO na primeira busca por texto (690 MB), pesos do Mask R-CNN e do YOLO nos treinos |
| GPU | opcional. Tudo roda em CPU. Com placa NVIDIA e driver instalado, o instalador usa CUDA 12.4. No Mac com Apple Silicon, o app oferece MPS (não testado) |
| Memória | 8 GB de RAM bastam para o SAM 2.1 tiny; o SAM 1 ViT-H em CPU passa de 6 GB |

## Windows

1. Instale o Python 3.12 de python.org. Na primeira tela do instalador marque **"Add python.exe to PATH"**; deixe marcado o **"py launcher"**. No fim, se aparecer **"Disable path length limit"**, clique (evita erro de caminho longo no pip).
2. Instale o git (opções padrão).
3. Abra o **PowerShell** (ou o Prompt de Comando) e rode:

```powershell
git clone https://github.com/Leonardo-Baptista/aula_SAM_propemm.git
cd aula_SAM_propemm
py -3.12 -m venv .venv
.venv\Scripts\python scripts\instalar.py --opcionais
.venv\Scripts\python -m sam_anota
```

Não é preciso ativar o venv: chamar `.venv\Scripts\python` já usa o Python certo. Se quiser ativar (`.venv\Scripts\Activate.ps1`) e o PowerShell recusar com "a execução de scripts foi desabilitada", libere só para o seu usuário:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Caminhos longos: se o `git clone` ou o `pip` reclamarem de caminho longo, clone numa pasta curta (por exemplo `C:\aula`) e rode `git config --global core.longpaths true`.

## Linux (Ubuntu, Debian e derivados)

```bash
sudo apt install python3.12 python3.12-venv python3.12-tk git
git clone https://github.com/Leonardo-Baptista/aula_SAM_propemm.git
cd aula_SAM_propemm
python3.12 -m venv .venv
.venv/bin/python scripts/instalar.py --opcionais
.venv/bin/python -m sam_anota
```

Se a sua distribuição não tiver `python3.12` nos repositórios, use o Python que ela oferece (3.10 ou mais novo) e o pacote `python3-tk` correspondente; o instalador avisa que não é a versão testada. Não testado em Linux.

## macOS

Use o instalador do python.org (ele já traz o Tk). Com Homebrew, instale também o Tk: `brew install python@3.12 python-tk@3.12`.

```bash
git clone https://github.com/Leonardo-Baptista/aula_SAM_propemm.git
cd aula_SAM_propemm
python3.12 -m venv .venv
.venv/bin/python scripts/instalar.py --opcionais
.venv/bin/python -m sam_anota
```

No Mac o torch do PyPI já traz o MPS (GPU da Apple Silicon); escolha "GPU (MPS, Mac)" no app. Não testado em macOS.

## CPU ou GPU

O `instalar.py` decide sozinho: se o comando `nvidia-smi` responder, instala o torch com CUDA 12.4 (cerca de 2,5 GB de download); senão, o torch só CPU (cerca de 200 MB). Para forçar: `--cpu` ou `--gpu`.

Para usar a GPU NVIDIA:

* o driver da NVIDIA precisa estar instalado ([nvidia.com/Download](https://www.nvidia.com/Download/index.aspx)); teste com `nvidia-smi` no terminal, que deve listar a placa;
* não é preciso instalar o CUDA Toolkit: o torch traz as bibliotecas de CUDA;
* versões fixadas: `torch==2.6.0` e `torchvision==0.21.0` do índice `https://download.pytorch.org/whl/cu124` (testado na RTX 3060 Laptop, driver 551.76). Em CPU: `torch==2.14.1` e `torchvision==0.29.1` do índice `https://download.pytorch.org/whl/cpu`.

Trocar um venv só CPU para GPU:

```powershell
.venv\Scripts\python -m pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

No app, o combo **Dispositivo** tem Automático, CPU, GPU (CUDA) e, no Mac, GPU (MPS). Se você escolher GPU e ela não puder ser usada, o app não quebra: mostra o motivo (torch instalado sem CUDA, driver ausente, nenhuma placa NVIDIA) com o comando acima e segue em CPU.

## Opções do instalador

| Opção | O que faz |
| --- | --- |
| `--cpu`, `--gpu` | força o torch só CPU ou com CUDA 12.4 |
| `--sam1` | `requirements-sam1.txt` (segment-anything) e baixa o peso ViT-B (375 MB) |
| `--texto` | `requirements-texto.txt` (transformers) para o modo texto |
| `--notebooks` | `requirements-notebooks.txt` (jupyter, matplotlib, segment-anything) para rodar os notebooks no computador |
| `--opcionais` | os três acima |
| `--sem-opcionais` | não pergunta pelos opcionais |
| `--sem-checkpoint` | não baixa pesos |

Sem nenhuma opção, num terminal interativo, ele pergunta pelos opcionais. Os exemplos com YOLO (`requirements-exemplos.txt`, ultralytics, AGPL-3.0) ficam de fora de propósito: veja `exemplos/LEIA.md`.

## Instalação manual (os mesmos passos, um a um)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
$env:SAM2_BUILD_CUDA = "0"
.venv\Scripts\python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
.venv\Scripts\python -m pip install -r requirements-sam1.txt -r requirements-texto.txt -r requirements-notebooks.txt
.venv\Scripts\python scripts\obter_checkpoint.py --modelo sam2.1_tiny,sam1_vit_b
.venv\Scripts\python scripts\verificar.py
```

No Prompt de Comando (cmd), troque a linha do `$env:` por `set SAM2_BUILD_CUDA=0`. No Git Bash, Linux e macOS, use `SAM2_BUILD_CUDA=0 .venv/bin/python -m pip install ...` na mesma linha.

Por que assim:

* `SAM2_BUILD_CUDA=0`: sem isso o pip tenta compilar a extensão CUDA do sam2 (`sam2._C`), o que no Windows falha sem Visual Studio e CUDA Toolkit. A extensão só serve para preencher furos pequenos em máscaras de vídeo; no SAM para imagens ela não é usada.
* `--no-build-isolation`: o `pyproject.toml` do sam2 pede o torch para o build; sem essa opção o pip baixaria um segundo torch (no Linux, a versão CUDA de vários GB) só para instalar. Por isso `setuptools` e `wheel` estão no `requirements.txt` (o venv do Python 3.12 não os traz).
* O sam2 é instalado do GitHub oficial num commit fixo, para todos terem o mesmo código.

Outros pesos, quando quiser (o app também baixa sob demanda, com barra de progresso):

```powershell
.venv\Scripts\python scripts\obter_checkpoint.py --listar
.venv\Scripts\python scripts\obter_checkpoint.py --modelo sam2.1_large,sam1_vit_h
```

## Verificação

```powershell
.venv\Scripts\python scripts\verificar.py
.venv\Scripts\python -m pytest -q
```

O `verificar.py` lista os pacotes (obrigatórios e opcionais), carrega o SAM 2.1 tiny, codifica uma imagem e mede os tempos; a saída termina em `Ambiente ok.` Os testes não precisam do SAM e levam poucos segundos.

## Erros comuns

| Sintoma | Causa e solução |
| --- | --- |
| `'py' não é reconhecido` | Python instalado sem o launcher. Reinstale marcando "py launcher", ou use o caminho completo do `python.exe` 3.12 |
| `git não está no PATH` | instale o git e abra um terminal novo |
| `No module named 'tkinter'` ou `_tkinter` | Linux: `sudo apt install python3.12-tk`; macOS com Homebrew: `brew install python-tk@3.12`; Windows: reinstale o Python de python.org com "tcl/tk and IDLE" marcado |
| pip ou git falham com erro de conexão, SSL ou certificado | rede com proxy: `set HTTPS_PROXY=http://proxy:porta` (cmd) ou `$env:HTTPS_PROXY="http://proxy:porta"` (PowerShell) antes de rodar o instalador; no git, `git config --global http.proxy http://proxy:porta`. O download dos pesos usa a mesma variável |
| instalação muito lenta, arquivos "em uso" ou apagados | antivírus examinando cada arquivo do venv. Espere, ou adicione a pasta do repositório às exclusões do antivírus |
| "GPU (CUDA) indisponível" ao escolher GPU | o app explica o motivo e segue em CPU. Torch sem CUDA: rode o comando de troca acima. Sem `nvidia-smi`: instale o driver. Notebook com GPU híbrida: confira o modo de energia |
| `UserWarning` sobre `sam2._C` ou a extensão C | esperado com `SAM2_BUILD_CUDA=0`; não afeta o SAM para imagens |
| `FutureWarning: torch.jit.script is deprecated` | vem do sam2, pode ignorar |
| primeira busca por texto demora | o Grounding DINO tiny (690 MB) é baixado do Hugging Face na primeira vez, para `%USERPROFILE%\.cache\huggingface` (Linux e macOS: `~/.cache/huggingface`) |
| "SAM 3 indisponível" | os pesos exigem pedido de acesso no Hugging Face; o app explica os passos e oferece o Grounding DINO |
| `No space left on device` ou disco cheio | apague `saidas/cache_embeddings/` (cache, pode apagar), pesos que não usa em `checkpoints/`, e o cache do pip (`python -m pip cache purge`) |
| janela cortada | telas com escala de 125% ou 150%: maximize; o layout foi pensado para 1366x768 |

## O que foi testado

Em 14/10/2026, num clone novo do repositório, venv novo e sem cache do pip (Windows 11 Home, Python 3.12.10, i7-12700H, RTX 3060 Laptop, rede de cerca de 30 MB/s):

| Caminho | Tempo de instalação | Verificação |
| --- | --- | --- |
| `instalar.py --cpu --opcionais` | 4 min 53 s | `Ambiente ok.`, 45 testes, teste da interface (com o fallback de GPU para CPU), notebook 01 executado em 43 s |
| `instalar.py --opcionais` (GPU detectada) | 5 min 48 s | `Ambiente ok.` na GPU, 45 testes, teste da interface na GPU |

Linux e macOS não foram testados.
