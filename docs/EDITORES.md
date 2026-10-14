# VS Code, Cursor e Antigravity

Os três editores vêm da mesma base (o VS Code) e abrem este repositório do mesmo jeito. A pasta `.vscode/` versionada traz o interpretador padrão, o pytest ligado, a configuração de execução "SAM Anota" (F5) e as tarefas de instalação e teste. Informações conferidas em 14/10/2026, com o link de cada fonte.

## Diferenças que importam

| | VS Code | Cursor | Antigravity |
| --- | --- | --- | --- |
| Base | o próprio | baseado no código do VS Code, com rebase regular ([cursor.com](https://cursor.com/docs/configuration/migrations/vscode)) | IDE agêntica do Google ([antigravity.google](https://antigravity.google/blog/introducing-google-antigravity)); terceiros a descrevem como fork do VS Code, o que o Google não confirmou publicamente ([Visual Studio Magazine](https://visualstudiomagazine.com/Articles/2025/11/20/Google-Joins-AI-IDE-Race-to-Compete-with-VS-Code-Apparently-Forking-VS-Code.aspx)) |
| Extensões | Visual Studio Marketplace | Open VSX, por um proxy próprio; extensões exclusivas do marketplace da Microsoft podem faltar ([cursor.com](https://cursor.com/docs/configuration/extensions)) | segundo terceiros, Open VSX (não confirmado em página oficial) |
| Python | extensão Python (ms-python.python) com Pylance | Pylance não é suportado; a extensão "Python" da Anysphere (`anysphere.cursorpyright`) substitui o Pylance e usa a ms-python.python ([fórum oficial](https://forum.cursor.com/t/new-in-house-extensions-c-c-ssh-devcontainers-wsl-python/94531)) | instale a extensão Python que aparecer na busca de extensões |
| Instruções para o assistente de IA | `AGENTS.md` na raiz, lido pelo chat desde a versão 1.104 (`chat.useAgentsMdFile`, ligado por padrão) ([code.visualstudio.com](https://code.visualstudio.com/docs/copilot/customization/custom-instructions)) | `AGENTS.md` na raiz e em subpastas, além de `.cursor/rules` ([cursor.com](https://cursor.com/docs/context/rules)) | `AGENTS.md` ou `GEMINI.md` na pasta, além de `.agents/rules/*.md` ([antigravity.google](https://antigravity.google/docs/rules?tab=ide)) |
| Download | [code.visualstudio.com](https://code.visualstudio.com/) | [cursor.com](https://cursor.com/) | [antigravity.google/download](https://antigravity.google/download) (há o "app" e a "IDE standalone"; a IDE não é oferecida para clientes enterprise) |

Este repositório tem um `AGENTS.md` na raiz: os três assistentes o leem sozinhos e passam a conhecer a estrutura, as convenções e os comandos de teste. Não é preciso configurar nada.

## Passo a passo (igual nos três; os nomes dos comandos são os do VS Code)

### 1. Clonar e abrir

* Pelo editor: paleta de comandos (Ctrl+Shift+P), **Git: Clone**, cole `https://github.com/Leonardo-Baptista/aula_SAM_propemm.git`, escolha a pasta e aceite abrir ([VS Code](https://code.visualstudio.com/docs/sourcecontrol/repos-remotes)).
* Pelo terminal: `git clone ...` e depois **File > Open Folder** na pasta clonada.

Na primeira abertura o editor sugere as extensões de `.vscode/extensions.json` (Python, Python Debugger, Jupyter). Aceite. No Cursor, a extensão Python da Anysphere entra no lugar do Pylance.

### 2. Criar o ambiente e instalar

Use o **terminal integrado** (Ctrl+'), que já abre na pasta do repositório:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python scripts\instalar.py --opcionais
```

Ou pelas tarefas: menu **Terminal > Run Task** ([VS Code](https://code.visualstudio.com/docs/debugtest/tasks)), "Criar ambiente (.venv)" e depois "Instalar (CPU)" ou "Instalar (GPU NVIDIA, CUDA 12.4)".

O comando **Python: Create Environment** também cria um venv, mas instala só o `requirements.txt`; o torch e o sam2 precisam do `scripts/instalar.py`. Se usar esse comando, aceite o nome `.venv` e rode o instalador depois.

### 3. Selecionar o interpretador

O `.vscode/settings.json` já aponta para `.venv\Scripts\python.exe` (Windows). Se o editor não o pegar, rode **Python: Select Interpreter** e escolha o `.venv` ([VS Code](https://code.visualstudio.com/docs/python/environments)). Linux e macOS: escolha `.venv/bin/python`.

### 4. Rodar o aplicativo

**F5** (ou Run and Debug, "SAM Anota"): roda `python -m sam_anota` com o interpretador do `.venv` ([VS Code](https://code.visualstudio.com/docs/debugtest/debugging-configuration)). Há também "SAM Anota (forçar CPU)". Sem depurador, use a tarefa "Abrir o aplicativo".

### 5. Testes

O pytest já está ligado (`python.testing.pytestEnabled`). Abra o painel **Testing** (ícone de béquer) e rode tudo ou um teste ([VS Code](https://code.visualstudio.com/docs/python/testing)). Pelo terminal: `.venv\Scripts\python -m pytest -q`, ou a tarefa "Rodar testes".

### 6. Notebooks

Abra `notebooks/01_sam1_guiado.ipynb`, clique no seletor de kernel no canto superior direito (ou **Notebook: Select Notebook Kernel**) e escolha o `.venv` ([VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)). O kernel precisa do jupyter instalado no venv: `scripts/instalar.py --notebooks` (já incluído em `--opcionais`). No Colab, use o badge no topo de cada notebook.

### 7. Assistente de IA

Peça ao assistente do editor coisas como "rode os testes", "explique como o cache de embeddings funciona" ou "adicione uma classe padrão". Ele lê o `AGENTS.md` e segue as regras do projeto (português, nada escrito em `dados/`, versões fixadas, ultralytics fora do aplicativo). Revise o que ele propuser antes de aceitar.
