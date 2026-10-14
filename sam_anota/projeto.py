"""Estado do projeto de anotação: imagens, classes e objetos aceitos.

Tudo é salvo num único JSON (projeto.json) dentro de saidas/<nome_da_pasta>/,
nunca dentro da pasta de imagens. As máscaras vão em RLE simples (linha a linha),
para o arquivo ficar pequeno e legível por qualquer linguagem.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
# SAM_ANOTA_SAIDAS permite mandar tudo para outro lugar (útil em testes)
PASTA_SAIDAS = Path(os.environ.get("SAM_ANOTA_SAIDAS") or RAIZ / "saidas")
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUFIXO_REFERENCIA = "_ref"
CLASSES_PADRAO = ["objeto", "pessoa", "carro", "moto", "cao", "gato"]
# cores por classe (RGB), da paleta de Sasha Trubetskoy; a mesma na tela e na exportação semântica
PALETA = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
          (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
          (0, 128, 128), (170, 110, 40)]


def cor_da_classe(indice: int) -> tuple[int, int, int]:
    return PALETA[indice % len(PALETA)]
VERSAO_FORMATO = 1


# ---------------------------------------------------------------- RLE

def mascara_para_rle(mascara: np.ndarray) -> dict:
    """Codifica uma máscara booleana em RLE, percorrendo as linhas (ordem C).

    contagens alterna corridas de 0 e de 1, começando sempre por 0
    (a primeira corrida pode ter tamanho zero).
    """
    m = np.asarray(mascara, dtype=bool)
    plano = m.ravel(order="C").astype(np.uint8)
    if plano.size == 0:
        return {"tamanho": list(m.shape), "contagens": []}
    mudancas = np.flatnonzero(np.diff(plano)) + 1
    limites = np.concatenate(([0], mudancas, [plano.size]))
    contagens = np.diff(limites).tolist()
    if plano[0] == 1:
        contagens = [0] + contagens
    return {"tamanho": [int(m.shape[0]), int(m.shape[1])], "contagens": [int(c) for c in contagens]}


def rle_para_mascara(rle: dict) -> np.ndarray:
    altura, largura = rle["tamanho"]
    plano = np.zeros(altura * largura, dtype=bool)
    pos = 0
    valor = False
    for c in rle["contagens"]:
        if valor:
            plano[pos:pos + c] = True
        pos += c
        valor = not valor
    return plano.reshape(altura, largura)


# ---------------------------------------------------------------- modelos de dados

@dataclass
class Objeto:
    id: int
    classe: str
    rle: dict
    escore_previsto: float
    indice_candidata: int
    cliques: int
    tempo_s: float
    pontos: list = field(default_factory=list)  # [[x, y, rotulo], ...] em pixels da imagem
    caixa: list | None = None  # [x0, y0, x1, y1]
    iou_real: float | None = None
    criado_em: str = ""
    origem: str = ""  # de qual modelo veio a máscara, por exemplo "SAM 2.1 tiny (cliques)"

    def mascara(self) -> np.ndarray:
        return rle_para_mascara(self.rle)

    @property
    def area(self) -> int:
        # soma das corridas de 1 (posições ímpares)
        return int(sum(self.rle["contagens"][1::2]))


@dataclass
class Imagem:
    caminho_relativo: str  # relativo à pasta de imagens, sempre com "/"
    largura: int = 0
    altura: int = 0
    concluida: bool = False
    objetos: list[Objeto] = field(default_factory=list)
    tempo_codificacao_s: float | None = None


class Projeto:
    """Pasta de imagens + classes + anotações. Salva com escrita atômica."""

    def __init__(self, pasta_imagens: Path, classes: list[str] | None = None):
        self.pasta_imagens = Path(pasta_imagens).resolve()
        self.classes: list[str] = list(classes) if classes else list(CLASSES_PADRAO)
        self.imagens: dict[str, Imagem] = {}
        self.proximo_id = 1
        self.criado_em = _agora()
        self.caminho_arquivo: Path | None = None

    # ------------------------------------------------ caminhos

    @staticmethod
    def pasta_saida_para(pasta_imagens: Path) -> Path:
        """saidas/<nome_da_pasta>. Pastas com o mesmo nome em lugares diferentes
        ganham sufixo com o nome da pasta-mãe."""
        p = Path(pasta_imagens).resolve()
        nome = _nome_seguro(p.name or "imagens")
        try:
            rel = p.relative_to(RAIZ)
            nome = _nome_seguro("_".join(rel.parts)) or nome
        except ValueError:
            pass
        return PASTA_SAIDAS / nome

    @property
    def pasta_saida(self) -> Path:
        return self.pasta_saida_para(self.pasta_imagens)

    def caminho_absoluto(self, rel: str) -> Path:
        return self.pasta_imagens / rel

    # ------------------------------------------------ imagens

    @staticmethod
    def listar_imagens(pasta: Path) -> list[str]:
        """Imagens da pasta e subpastas, sem as referências <nome>_ref.png."""
        pasta = Path(pasta)
        achadas = []
        # exemplos/ primeiro: a aula começa pelas imagens fáceis
        for raiz, dirs, arquivos in os.walk(pasta):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for a in sorted(arquivos):
                p = Path(raiz) / a
                if p.suffix.lower() not in EXTENSOES_IMAGEM:
                    continue
                if p.stem.endswith(SUFIXO_REFERENCIA):
                    continue
                achadas.append(p.relative_to(pasta).as_posix())
        return sorted(achadas, key=lambda r: (not r.startswith("exemplos/"), r))

    def sincronizar(self) -> list[str]:
        """Inclui imagens novas encontradas na pasta. Devolve a lista ordenada."""
        nomes = self.listar_imagens(self.pasta_imagens)
        for n in nomes:
            if n not in self.imagens:
                self.imagens[n] = Imagem(caminho_relativo=n)
        return nomes

    # ------------------------------------------------ objetos

    def adicionar_objeto(self, rel: str, mascara: np.ndarray, classe: str, escore: float,
                         indice: int, cliques: int, tempo_s: float, pontos=None, caixa=None,
                         iou_real: float | None = None, origem: str = "") -> Objeto:
        img = self.imagens.setdefault(rel, Imagem(caminho_relativo=rel))
        if classe not in self.classes:
            self.classes.append(classe)
        obj = Objeto(
            id=self.proximo_id, classe=classe, rle=mascara_para_rle(mascara),
            escore_previsto=float(escore), indice_candidata=int(indice), cliques=int(cliques),
            tempo_s=float(tempo_s), pontos=[list(map(float, p)) for p in (pontos or [])],
            caixa=list(map(float, caixa)) if caixa is not None else None,
            iou_real=None if iou_real is None else float(iou_real), criado_em=_agora(), origem=origem,
        )
        self.proximo_id += 1
        img.objetos.append(obj)
        img.altura, img.largura = mascara.shape[:2]
        return obj

    def remover_objeto(self, rel: str, id_objeto: int) -> bool:
        img = self.imagens.get(rel)
        if img is None:
            return False
        antes = len(img.objetos)
        img.objetos = [o for o in img.objetos if o.id != id_objeto]
        return len(img.objetos) != antes

    def classe_em_uso(self, classe: str) -> bool:
        return any(o.classe == classe for img in self.imagens.values() for o in img.objetos)

    def total_objetos(self) -> int:
        return sum(len(i.objetos) for i in self.imagens.values())

    # ------------------------------------------------ persistência

    def para_dict(self) -> dict:
        return {
            "formato": "sam-anota",
            "versao": VERSAO_FORMATO,
            "pasta_imagens": str(self.pasta_imagens),
            "criado_em": self.criado_em,
            "salvo_em": _agora(),
            "classes": self.classes,
            "proximo_id": self.proximo_id,
            "imagens": {k: asdict(v) for k, v in self.imagens.items()},
        }

    def salvar(self, caminho: Path | None = None) -> Path:
        caminho = Path(caminho or self.caminho_arquivo or self.pasta_saida / "projeto.json")
        caminho.parent.mkdir(parents=True, exist_ok=True)
        dados = json.dumps(self.para_dict(), ensure_ascii=False, indent=1)
        # escrita atômica: grava num temporário e troca, para não corromper se cair a luz
        fd, tmp = tempfile.mkstemp(dir=caminho.parent, prefix=".projeto_", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(dados)
        os.replace(tmp, caminho)
        self.caminho_arquivo = caminho
        return caminho

    @classmethod
    def carregar(cls, caminho: Path) -> "Projeto":
        caminho = Path(caminho)
        d = json.loads(caminho.read_text(encoding="utf-8"))
        proj = cls(Path(d["pasta_imagens"]), d.get("classes"))
        proj.proximo_id = int(d.get("proximo_id", 1))
        proj.criado_em = d.get("criado_em", proj.criado_em)
        for rel, di in d.get("imagens", {}).items():
            objs = [Objeto(**o) for o in di.pop("objetos", [])]
            proj.imagens[rel] = Imagem(objetos=objs, **di)
        proj.caminho_arquivo = caminho
        return proj

    @classmethod
    def abrir_ou_criar(cls, pasta_imagens: Path) -> "Projeto":
        caminho = cls.pasta_saida_para(pasta_imagens) / "projeto.json"
        if caminho.exists():
            proj = cls.carregar(caminho)
            proj.pasta_imagens = Path(pasta_imagens).resolve()
        else:
            proj = cls(pasta_imagens)
            proj.caminho_arquivo = caminho
        proj.sincronizar()
        return proj


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _nome_seguro(texto: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", texto).strip("_")
