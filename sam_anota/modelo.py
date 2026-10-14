"""Catálogo de modelos SAM, escolha de dispositivo, download de pesos, cache e predição.

Modelos:
  * SAM 2.1 (pacote sam2, sempre instalado): tiny, small, base+, large;
  * SAM 1 (pacote segment-anything, opcional): ViT-B, ViT-L, ViT-H.

Dispositivo: "auto" (CUDA, depois MPS, depois CPU), "cpu", "cuda" ou "mps". Pedir GPU
sem GPU utilizável não quebra: volta para CPU com uma mensagem que diz por quê e como
resolver (ver diagnosticar_cuda).

Cache de embeddings em dois níveis, com a chave incluindo o modelo:
  * memória: os últimos N embeddings (cerca de 16 MB cada no SAM 2.1);
  * disco: saidas/cache_embeddings/<sha1 da imagem>_<modelo>.pt.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import projeto as _projeto
from .projeto import RAIZ

PASTA_CHECKPOINTS = RAIZ / "checkpoints"


def pasta_cache() -> Path:
    return _projeto.PASTA_SAIDAS / "cache_embeddings"


# ================================================================ catálogo

@dataclass(frozen=True)
class InfoModelo:
    chave: str
    familia: str  # "sam2" ou "sam1"
    rotulo: str
    arquivo: str
    url: str
    sha256: str  # conferido no download (13/10/2026)
    tamanho_bytes: int
    parametros_m: float  # milhões de parâmetros (README oficial e artigo)
    config: str  # sam2: config hydra; sam1: tipo do registro (vit_b...)

    @property
    def rotulo_combo(self) -> str:
        mb = self.tamanho_bytes / 1e6
        tam = f"{mb / 1000:.1f} GB".replace(".", ",") if mb >= 1000 else f"{mb:.0f} MB"
        return f"{self.rotulo} ({tam}, {_virgula(self.parametros_m)} M)"

    @property
    def caminho(self) -> Path:
        return PASTA_CHECKPOINTS / self.arquivo

    @property
    def baixado(self) -> bool:
        p = self.caminho
        return p.exists() and p.stat().st_size == self.tamanho_bytes


def _virgula(x: float) -> str:
    return (f"{x:.1f}" if x % 1 else f"{x:.0f}").replace(".", ",")


_U2 = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
_U1 = "https://dl.fbaipublicfiles.com/segment_anything/"
CATALOGO: dict[str, InfoModelo] = {m.chave: m for m in [
    InfoModelo("sam2.1_tiny", "sam2", "SAM 2.1 tiny", "sam2.1_hiera_tiny.pt", _U2 + "sam2.1_hiera_tiny.pt",
               "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69", 156008466, 38.9,
               "configs/sam2.1/sam2.1_hiera_t.yaml"),
    InfoModelo("sam2.1_small", "sam2", "SAM 2.1 small", "sam2.1_hiera_small.pt", _U2 + "sam2.1_hiera_small.pt",
               "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38", 184416285, 46,
               "configs/sam2.1/sam2.1_hiera_s.yaml"),
    InfoModelo("sam2.1_base_plus", "sam2", "SAM 2.1 base+", "sam2.1_hiera_base_plus.pt",
               _U2 + "sam2.1_hiera_base_plus.pt",
               "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5", 323606802, 80.8,
               "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    InfoModelo("sam2.1_large", "sam2", "SAM 2.1 large", "sam2.1_hiera_large.pt", _U2 + "sam2.1_hiera_large.pt",
               "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318", 898083611, 224.4,
               "configs/sam2.1/sam2.1_hiera_l.yaml"),
    # SAM 1: o sufixo do nome do arquivo é o início do MD5 (conferido); o SHA-256 foi registrado aqui
    InfoModelo("sam1_vit_b", "sam1", "SAM 1 ViT-B", "sam_vit_b_01ec64.pth", _U1 + "sam_vit_b_01ec64.pth",
               "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912", 375042383, 91, "vit_b"),
    InfoModelo("sam1_vit_l", "sam1", "SAM 1 ViT-L", "sam_vit_l_0b3195.pth", _U1 + "sam_vit_l_0b3195.pth",
               "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622", 1249524607, 308, "vit_l"),
    InfoModelo("sam1_vit_h", "sam1", "SAM 1 ViT-H", "sam_vit_h_4b8939.pth", _U1 + "sam_vit_h_4b8939.pth",
               "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e", 2564550879, 636, "vit_h"),
]}
MODELO_PADRAO = "sam2.1_tiny"
# nomes curtos aceitos na linha de comando
APELIDOS = {"tiny": "sam2.1_tiny", "small": "sam2.1_small", "base_plus": "sam2.1_base_plus",
            "large": "sam2.1_large", "vit_b": "sam1_vit_b", "vit_l": "sam1_vit_l", "vit_h": "sam1_vit_h"}


def familia_instalada(familia: str) -> bool:
    pacote = {"sam2": "sam2", "sam1": "segment_anything"}[familia]
    return importlib.util.find_spec(pacote) is not None


def modelos_disponiveis() -> list[InfoModelo]:
    """Modelos cujo pacote está instalado (baixados ou não)."""
    return [m for m in CATALOGO.values() if familia_instalada(m.familia)]


def resolver_chave(nome: str) -> str:
    chave = APELIDOS.get(nome, nome)
    if chave not in CATALOGO:
        raise ErroModelo(f"Modelo desconhecido: {nome}. Opções: {', '.join(CATALOGO)}")
    return chave


# ================================================================ download

def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar_checkpoint(info: InfoModelo, progresso=None, cancelado=None) -> Path:
    """Baixa o peso oficial para checkpoints/, calculando o SHA-256 durante o download.

    progresso(feito_bytes, total_bytes) é chamado a cada bloco; cancelado() -> True interrompe.
    Hash diferente do registrado: apaga o arquivo e levanta ErroModelo.
    """
    destino = info.caminho
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    h = hashlib.sha256()
    req = urllib.request.Request(info.url, headers={"User-Agent": "sam-anota/0.2"})
    with urllib.request.urlopen(req, timeout=60) as r, open(parcial, "wb") as f:
        total = int(r.headers.get("Content-Length") or info.tamanho_bytes)
        feito = 0
        while True:
            if cancelado and cancelado():
                f.close()
                parcial.unlink(missing_ok=True)
                raise ErroModelo("Download cancelado.")
            bloco = r.read(1 << 20)
            if not bloco:
                break
            f.write(bloco)
            h.update(bloco)
            feito += len(bloco)
            if progresso:
                progresso(feito, total)
    if h.hexdigest() != info.sha256:
        parcial.unlink(missing_ok=True)
        raise ErroModelo(f"SHA-256 de {info.arquivo} não confere (esperado {info.sha256[:12]}..., "
                         f"obtido {h.hexdigest()[:12]}...). Arquivo apagado; tente de novo.")
    os.replace(parcial, destino)
    return destino


# ================================================================ dispositivo

class ErroModelo(RuntimeError):
    """Erro com mensagem pronta para mostrar ao aluno."""


DISPOSITIVOS_ROTULO = {"auto": "Automático", "cpu": "CPU", "cuda": "GPU (CUDA)", "mps": "GPU (MPS, Mac)"}


def mps_disponivel() -> bool:
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def dispositivos_disponiveis() -> list[str]:
    """Opções para o combo. CUDA aparece sempre (para explicar quando não funciona); MPS só no Mac."""
    opcoes = ["auto", "cpu", "cuda"]
    if mps_disponivel():
        opcoes.append("mps")
    return opcoes


def _nvidia_smi() -> str | None:
    """Saída de nvidia-smi -L, ou None se o comando não existir ou falhar."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def comando_torch_cuda() -> str:
    py = sys.executable if " " not in sys.executable else f'"{sys.executable}"'
    return (f"{py} -m pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 "
            "--index-url https://download.pytorch.org/whl/cu124")


def diagnosticar_cuda() -> str:
    """Explica por que torch.cuda.is_available() é falso e como resolver."""
    import torch

    smi = _nvidia_smi()
    linhas = []
    if torch.version.cuda is None:
        linhas.append(f"O torch instalado ({torch.__version__}) é a versão só CPU, sem CUDA.")
        if smi:
            linhas.append(f"A placa foi detectada pelo driver: {smi.splitlines()[0]}.")
            linhas.append("Para usar a GPU, reinstale o torch com CUDA (download de cerca de 2,5 GB):")
            linhas.append(comando_torch_cuda())
        elif smi is None:
            linhas.append("Além disso, o comando nvidia-smi não existe: não há placa NVIDIA ou o driver "
                          "não está instalado. Sem placa NVIDIA, use CPU.")
        else:
            linhas.append("O nvidia-smi existe, mas não listou nenhuma placa: verifique o driver NVIDIA.")
    else:
        linhas.append(f"O torch tem CUDA {torch.version.cuda} ({torch.__version__}), mas não achou GPU utilizável.")
        if smi is None:
            linhas.append("O comando nvidia-smi não existe: driver NVIDIA ausente ou nenhuma placa NVIDIA. "
                          "Instale o driver em https://www.nvidia.com/Download/index.aspx.")
        elif not smi:
            linhas.append("O nvidia-smi não listou placas: driver com problema ou GPU desativada "
                          "(em notebooks híbridos, confira o modo de energia).")
        else:
            linhas.append(f"Placa vista pelo driver: {smi.splitlines()[0]}. O driver pode ser antigo demais "
                          f"para CUDA {torch.version.cuda}: atualize o driver NVIDIA.")
    linhas.append("Seguindo em CPU.")
    return "\n".join(linhas)


def escolher_dispositivo(preferido: str | None = None) -> tuple[str, str | None]:
    """Devolve (dispositivo efetivo, aviso ou None). Nunca levanta por falta de GPU."""
    import torch

    preferido = (preferido or os.environ.get("SAM_ANOTA_DISPOSITIVO", "") or "auto").strip().lower()
    if preferido == "cpu":
        return "cpu", None
    if preferido == "cuda":
        if torch.cuda.is_available():
            return "cuda", None
        return "cpu", "GPU (CUDA) indisponível.\n" + diagnosticar_cuda()
    if preferido == "mps":
        if mps_disponivel():
            return "mps", None
        return "cpu", "GPU MPS indisponível (só existe em Mac com Apple Silicon e torch recente). Seguindo em CPU."
    if torch.cuda.is_available():
        return "cuda", None
    if mps_disponivel():
        return "mps", None
    return "cpu", None


def descrever_dispositivo(dispositivo: str) -> str:
    """Texto para a barra: dispositivo efetivo, placa e versões."""
    import torch

    if dispositivo == "cuda":
        try:
            nome = torch.cuda.get_device_name(0)
        except Exception:
            nome = "NVIDIA"
        nome = nome.replace("NVIDIA ", "").replace("GeForce ", "")
        return f"GPU {nome} | torch {torch.__version__} | CUDA {torch.version.cuda}"
    if dispositivo == "mps":
        return f"GPU Apple (MPS) | torch {torch.__version__}"
    extra = "" if torch.version.cuda is None else f" (CUDA {torch.version.cuda} sem uso)"
    return f"CPU | torch {torch.__version__}{extra}"


# ================================================================ imagem

def carregar_imagem_rgb(caminho: Path) -> np.ndarray:
    """Abre a imagem respeitando a orientação EXIF e devolve RGB uint8 (H, W, 3)."""
    with Image.open(caminho) as im:
        im = ImageOps.exif_transpose(im)
        return np.array(im.convert("RGB"))


def sha1_arquivo(caminho: Path) -> str:
    h = hashlib.sha1()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


# ================================================================ modelo

@dataclass
class Embedding:
    caracteristicas: dict
    altura: int
    largura: int
    tempo_s: float
    origem: str  # "calculado", "cache em memória" ou "cache em disco"


@dataclass
class Predicao:
    mascaras: np.ndarray  # (N, H, W) bool, N = 3 no SAM com multimask
    escores: np.ndarray  # (N,) IoU prevista pelo modelo
    tempo_s: float


class ModeloSAM:
    def __init__(self, nome: str = MODELO_PADRAO, checkpoint: Path | None = None, dispositivo: str | None = None,
                 cache_memoria: int = 12, cache_disco: bool = True):
        self.info = CATALOGO[resolver_chave(nome)]
        self.nome = self.info.chave
        self.checkpoint = Path(checkpoint) if checkpoint else self.info.caminho
        self.dispositivo_pedido = dispositivo
        self.dispositivo = "?"
        self.aviso_dispositivo: str | None = None
        self.cache_memoria_max = cache_memoria
        self.cache_disco = cache_disco
        self._cache: OrderedDict[str, Embedding] = OrderedDict()
        self._preditor = None
        self.tempo_carga_s: float | None = None

    # ------------------------------------------------ carga

    def carregar(self) -> float:
        """Constrói o modelo. Lento (1 a 30 s, conforme o modelo): fora da thread da interface."""
        if not self.checkpoint.exists():
            raise ErroModelo(f"Checkpoint não encontrado em {self.checkpoint}.\n"
                             f"Rode: python scripts/obter_checkpoint.py --modelo {self.nome}")
        if not familia_instalada(self.info.familia):
            extra = ("pip install segment-anything" if self.info.familia == "sam1"
                     else "a seção Instalação do README")
            raise ErroModelo(f"O pacote de {self.info.rotulo} não está instalado. Veja {extra}.")
        import torch
        self._torch = torch
        t0 = time.perf_counter()
        self.dispositivo, self.aviso_dispositivo = escolher_dispositivo(self.dispositivo_pedido)
        if self.info.familia == "sam2":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            modelo = build_sam2(self.info.config, str(self.checkpoint), device=self.dispositivo)
            modelo.eval()
            self._preditor = SAM2ImagePredictor(modelo)
        else:
            from segment_anything import SamPredictor, sam_model_registry

            modelo = sam_model_registry[self.info.config](checkpoint=str(self.checkpoint))
            modelo.to(self.dispositivo).eval()
            self._preditor = SamPredictor(modelo)
        self.tempo_carga_s = time.perf_counter() - t0
        return self.tempo_carga_s

    @property
    def carregado(self) -> bool:
        return self._preditor is not None

    def descricao(self) -> str:
        return f"{self.info.rotulo}, {descrever_dispositivo(self.dispositivo)}"

    def liberar(self) -> None:
        """Solta o modelo e o cache (troca de modelo)."""
        self._preditor = None
        self._cache.clear()
        try:
            if self.dispositivo == "cuda":
                self._torch.cuda.empty_cache()
        except Exception:
            pass

    def _contexto(self):
        torch = self._torch
        if self.dispositivo == "cuda" and self.info.familia == "sam2":
            return torch.autocast("cuda", dtype=torch.bfloat16)  # padrão dos exemplos oficiais
        return nullcontext()

    def _sincronizar(self):
        if self.dispositivo == "cuda":
            self._torch.cuda.synchronize()

    # ------------------------------------------------ cache e codificação

    def _arquivo_cache(self, chave: str) -> Path:
        return pasta_cache() / f"{chave}_{self.nome}.pt"

    def tem_no_cache(self, chave: str) -> bool:
        return chave in self._cache or self._arquivo_cache(chave).exists()

    def chave_de(self, caminho: Path) -> str:
        return sha1_arquivo(caminho)

    def _capturar(self) -> dict:
        p = self._preditor
        if self.info.familia == "sam2":
            f = p._features
            return {"image_embed": f["image_embed"], "high_res_feats": list(f["high_res_feats"])}
        return {"features": p.features, "input_size": tuple(int(v) for v in p.input_size)}

    def _restaurar(self, e: Embedding) -> None:
        p = self._preditor
        c = e.caracteristicas
        if self.info.familia == "sam2":
            p._features = {"image_embed": c["image_embed"], "high_res_feats": c["high_res_feats"]}
            p._orig_hw = [(e.altura, e.largura)]
            p._is_image_set = True
            p._is_batch = False
        else:
            p.features = c["features"]
            p.original_size = (e.altura, e.largura)
            p.input_size = tuple(c["input_size"])
            p.is_image_set = True

    def codificar(self, chave: str, imagem_rgb: np.ndarray) -> Embedding:
        """Devolve o embedding da imagem, do cache se possível."""
        if chave in self._cache:
            self._cache.move_to_end(chave)
            e = self._cache[chave]
            return Embedding(e.caracteristicas, e.altura, e.largura, 0.0, "cache em memória")
        torch = self._torch
        arq = self._arquivo_cache(chave)
        if self.cache_disco and arq.exists():
            t0 = time.perf_counter()
            try:
                d = torch.load(arq, map_location=self.dispositivo, weights_only=True)
                c = dict(d["caracteristicas"])
                if "high_res_feats" in c:
                    c["high_res_feats"] = list(c["high_res_feats"])
                e = Embedding(c, int(d["altura"]), int(d["largura"]), time.perf_counter() - t0, "cache em disco")
                self._guardar(chave, e)
                return e
            except Exception:
                arq.unlink(missing_ok=True)  # cache corrompido ou de outra versão: recalcula
        t0 = time.perf_counter()
        with torch.inference_mode(), self._contexto():
            self._preditor.set_image(imagem_rgb)
        self._sincronizar()
        tempo = time.perf_counter() - t0
        e = Embedding(self._capturar(), int(imagem_rgb.shape[0]), int(imagem_rgb.shape[1]), tempo, "calculado")
        self._guardar(chave, e)
        if self.cache_disco:
            self._salvar_cache(arq, e)
        return e

    def _salvar_cache(self, arq: Path, e: Embedding) -> None:
        torch = self._torch
        c = {}
        for k, v in e.caracteristicas.items():
            if isinstance(v, list):
                c[k] = [t.float().cpu() for t in v]
            elif hasattr(v, "cpu"):
                c[k] = v.float().cpu()
            else:
                c[k] = list(v)
        try:
            arq.parent.mkdir(parents=True, exist_ok=True)
            tmp = arq.with_suffix(".parcial")
            torch.save({"caracteristicas": c, "altura": e.altura, "largura": e.largura}, tmp)
            os.replace(tmp, arq)
        except OSError:
            pass  # sem espaço ou sem permissão: segue só com o cache em memória

    def _guardar(self, chave: str, e: Embedding) -> None:
        self._cache[chave] = e
        self._cache.move_to_end(chave)
        while len(self._cache) > self.cache_memoria_max:
            self._cache.popitem(last=False)

    # ------------------------------------------------ predição

    def prever(self, chave: str, pontos: list[tuple[float, float, int]] | None = None,
               caixa: tuple[float, float, float, float] | None = None) -> Predicao:
        """Três máscaras candidatas para os prompts (coordenadas em pixels da imagem).

        pontos: [(x, y, rotulo)], rotulo 1 = positivo, 0 = negativo. caixa: (x0, y0, x1, y1).
        """
        if chave not in self._cache:
            raise ErroModelo("Imagem ainda não codificada.")
        if not pontos and caixa is None:
            raise ErroModelo("Nenhum prompt.")
        self._restaurar(self._cache[chave])
        coords = rotulos = None
        if pontos:
            coords = np.array([[x, y] for x, y, _ in pontos], dtype=np.float32)
            rotulos = np.array([r for _, _, r in pontos], dtype=np.int32)
        box = None if caixa is None else np.array(caixa, dtype=np.float32)
        torch = self._torch
        t0 = time.perf_counter()
        with torch.inference_mode(), self._contexto():
            mascaras, escores, _ = self._preditor.predict(point_coords=coords, point_labels=rotulos, box=box,
                                                          multimask_output=True)
        self._sincronizar()
        tempo = time.perf_counter() - t0
        return Predicao(np.asarray(mascaras) > 0, np.asarray(escores, dtype=np.float32), tempo)
