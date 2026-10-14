"""Modo texto: um prompt em inglês ("smoke", "car") vira máscaras por instância.

Dois motores, ambos opcionais (instalados com scripts/instalar.py --texto):

  * "gdino": Grounding DINO tiny (IDEA-Research/grounding-dino-tiny, Apache 2.0, pesos
    abertos) acha caixas para o texto; cada caixa vira prompt de caixa do SAM escolhido.
    Roda em CPU. São dois modelos encadeados, não é o SAM 3.
  * "sam3": SAM 3 da Meta via transformers (Sam3Model, repositório facebook/sam3). Os pesos
    exigem pedido de acesso no Hugging Face e login com token. O SAM 3.1 (facebook/sam3.1)
    é publicado só para o pacote oficial github.com/facebookresearch/sam3, que pede GPU
    CUDA 12.6 ou mais; ele não é integrado aqui.

Nada aqui tenta contornar o controle de acesso: sem acesso aprovado, o motor sam3
responde com a explicação e o passo para resolver.
"""
from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field

import numpy as np

ID_GDINO = "IDEA-Research/grounding-dino-tiny"
ID_SAM3 = "facebook/sam3"
MOTORES = {"gdino": "Grounding DINO tiny + SAM", "sam3": "SAM 3 (transformers)"}
LIMIAR_CAIXA = 0.35
LIMIAR_TEXTO = 0.25
LIMIAR_SAM3 = 0.5

# tradução simples para casar o texto em inglês com classes em português
TRADUCOES = {"smoke": "fumaca", "plume": "fumaca", "car": "carro", "person": "pessoa", "people": "pessoa",
             "motorcycle": "moto", "motorbike": "moto", "dog": "cao", "cat": "gato", "crack": "trinca",
             "truck": "caminhao", "bus": "onibus", "object": "objeto"}


class ErroTexto(RuntimeError):
    """Motivo pronto para mostrar ao aluno, com o passo para resolver."""


@dataclass
class Instancia:
    caixa: tuple[float, float, float, float]  # x0, y0, x1, y1 em pixels
    escore_caixa: float  # confiança do detector (ou do SAM 3)
    rotulo: str  # trecho do texto casado
    mascaras: np.ndarray | None = None  # (N, H, W) bool
    escores: np.ndarray | None = None  # (N,)
    origem: str = ""


@dataclass
class ResultadoTexto:
    texto: str
    motor: str
    instancias: list[Instancia] = field(default_factory=list)
    tempo_deteccao_s: float = 0.0
    tempo_sam_s: float = 0.0


def normalizar_texto(texto: str) -> str:
    return " ".join(texto.strip().lower().replace(".", " ").split())


def classe_para_texto(texto: str, classes: list[str]) -> str | None:
    """Classe de mesmo nome do texto (ou tradução conhecida), se existir."""
    t = normalizar_texto(texto)
    for candidato in (t, TRADUCOES.get(t), t.rstrip("s"), TRADUCOES.get(t.rstrip("s"))):
        if candidato and candidato in classes:
            return candidato
    return None


def _tem(pacote: str) -> bool:
    return importlib.util.find_spec(pacote) is not None


def requisitos_gdino() -> str | None:
    """None se o motor pode rodar; senão o motivo com o passo para resolver."""
    if not _tem("transformers"):
        return ("O modo texto precisa do pacote transformers.\n"
                "Instale com: python scripts/instalar.py --texto  (ou pip install transformers)")
    return None


def requisitos_sam3(verificar_acesso: bool = True) -> str | None:
    """Confere pacote, classe, token e acesso aprovado ao facebook/sam3."""
    if not _tem("transformers"):
        return ("O SAM 3 usa o pacote transformers, que não está instalado.\n"
                "Instale com: python scripts/instalar.py --texto")
    import transformers

    if not hasattr(transformers, "Sam3Model"):
        return (f"O transformers instalado ({transformers.__version__}) não tem Sam3Model.\n"
                "Atualize com: pip install -U transformers")
    try:
        from huggingface_hub import get_token
        token = get_token()
    except Exception:
        token = None
    if not token:
        return ("Os pesos do SAM 3 (facebook/sam3) exigem pedido de acesso e login no Hugging Face.\n"
                "1. Crie conta e peça acesso em https://huggingface.co/facebook/sam3 (a Meta aprova manualmente).\n"
                "2. Crie um token de leitura em https://huggingface.co/settings/tokens.\n"
                "3. No terminal do venv: hf auth login\n"
                "Enquanto isso, use o motor Grounding DINO tiny + SAM.")
    if verificar_acesso:
        try:
            from huggingface_hub import auth_check
            auth_check(ID_SAM3, token=token)
        except Exception as e:
            nome = type(e).__name__
            if "Gated" in nome:
                return ("Há token, mas o acesso ao facebook/sam3 ainda não foi aprovado para esta conta.\n"
                        "Confira o pedido em https://huggingface.co/facebook/sam3.\n"
                        "Enquanto isso, use o motor Grounding DINO tiny + SAM.")
            return f"Não consegui verificar o acesso ao facebook/sam3 ({nome}: {e}). Sem internet?"
    return None


def _dispositivo_torch(dispositivo: str) -> str:
    return dispositivo if dispositivo in ("cpu", "cuda", "mps") else "cpu"


class DetectorGdino:
    """Grounding DINO tiny: texto -> caixas."""

    def __init__(self, dispositivo: str = "cpu"):
        self.dispositivo = _dispositivo_torch(dispositivo)
        self._proc = self._modelo = None
        self.tempo_carga_s: float | None = None

    def carregar(self) -> float:
        motivo = requisitos_gdino()
        if motivo:
            raise ErroTexto(motivo)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        t0 = time.perf_counter()
        try:
            self._proc = AutoProcessor.from_pretrained(ID_GDINO)
            self._modelo = AutoModelForZeroShotObjectDetection.from_pretrained(ID_GDINO).to(self.dispositivo).eval()
        except OSError as e:
            raise ErroTexto(f"Não consegui baixar {ID_GDINO} (cerca de 690 MB na primeira vez): {e}") from e
        self.tempo_carga_s = time.perf_counter() - t0
        return self.tempo_carga_s

    def detectar(self, imagem_rgb: np.ndarray, texto: str, limiar_caixa: float = LIMIAR_CAIXA,
                 limiar_texto: float = LIMIAR_TEXTO) -> tuple[list[Instancia], float]:
        import torch
        from PIL import Image

        if self._modelo is None:
            self.carregar()
        im = Image.fromarray(imagem_rgb)
        consulta = normalizar_texto(texto) + "."  # o Grounding DINO espera frases terminadas em ponto
        entrada = self._proc(images=im, text=consulta, return_tensors="pt").to(self.dispositivo)
        t0 = time.perf_counter()
        with torch.inference_mode():
            saida = self._modelo(**entrada)
        r = self._proc.post_process_grounded_object_detection(
            saida, entrada["input_ids"], threshold=limiar_caixa, text_threshold=limiar_texto,
            target_sizes=[im.size[::-1]])[0]
        tempo = time.perf_counter() - t0
        rotulos = r.get("text_labels", r.get("labels"))
        inst = [Instancia(tuple(float(v) for v in b.tolist()), float(s), str(l))
                for b, s, l in zip(r["boxes"], r["scores"], rotulos)]
        inst.sort(key=lambda i: -i.escore_caixa)
        return inst, tempo


class SegmentadorSam3:
    """SAM 3 via transformers: texto -> máscaras por instância. Exige acesso aos pesos."""

    def __init__(self, dispositivo: str = "cpu"):
        self.dispositivo = _dispositivo_torch(dispositivo)
        self._proc = self._modelo = None
        self.tempo_carga_s: float | None = None

    def carregar(self) -> float:
        motivo = requisitos_sam3()
        if motivo:
            raise ErroTexto(motivo)
        from transformers import Sam3Model, Sam3Processor

        t0 = time.perf_counter()
        self._proc = Sam3Processor.from_pretrained(ID_SAM3)
        self._modelo = Sam3Model.from_pretrained(ID_SAM3).to(self.dispositivo).eval()
        self.tempo_carga_s = time.perf_counter() - t0
        return self.tempo_carga_s

    def segmentar(self, imagem_rgb: np.ndarray, texto: str, limiar: float = LIMIAR_SAM3) -> tuple[list[Instancia], float]:
        import torch
        from PIL import Image

        if self._modelo is None:
            self.carregar()
        im = Image.fromarray(imagem_rgb)
        entrada = self._proc(images=im, text=normalizar_texto(texto), return_tensors="pt").to(self.dispositivo)
        t0 = time.perf_counter()
        with torch.inference_mode():
            saida = self._modelo(**entrada)
        r = self._proc.post_process_instance_segmentation(
            saida, threshold=limiar, mask_threshold=0.5,
            target_sizes=entrada.get("original_sizes").tolist())[0]
        tempo = time.perf_counter() - t0
        inst = []
        for m, b, s in zip(r["masks"], r["boxes"], r["scores"]):
            m = np.asarray(m.cpu().numpy() if hasattr(m, "cpu") else m) > 0
            inst.append(Instancia(tuple(float(v) for v in b.tolist()), float(s), normalizar_texto(texto),
                                  mascaras=m[None], escores=np.array([float(s)], np.float32), origem="SAM 3"))
        inst.sort(key=lambda i: -i.escore_caixa)
        return inst, tempo


def texto_com_gdino(detector: DetectorGdino, modelo_sam, chave: str, imagem_rgb: np.ndarray, texto: str,
                    limiar_caixa: float = LIMIAR_CAIXA, limiar_texto: float = LIMIAR_TEXTO,
                    max_instancias: int = 30) -> ResultadoTexto:
    """Texto -> caixas (Grounding DINO) -> 3 candidatas do SAM por caixa.

    O embedding do SAM para `chave` precisa estar pronto (modelo_sam.codificar).
    """
    inst, t_det = detector.detectar(imagem_rgb, texto, limiar_caixa, limiar_texto)
    inst = inst[:max_instancias]
    t0 = time.perf_counter()
    for i in inst:
        p = modelo_sam.prever(chave, None, i.caixa)
        i.mascaras, i.escores = p.mascaras, p.escores
        i.origem = f"Grounding DINO tiny + {modelo_sam.info.rotulo}"
    return ResultadoTexto(texto, "gdino", inst, t_det, time.perf_counter() - t0)
