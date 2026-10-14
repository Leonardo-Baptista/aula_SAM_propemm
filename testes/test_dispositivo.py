import pytest

torch = pytest.importorskip("torch")

from sam_anota import modelo as m  # noqa: E402
from sam_anota import texto  # noqa: E402


def sem_gpu(monkeypatch, versao_cuda=None, smi=None):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", versao_cuda)
    monkeypatch.setattr(m, "_nvidia_smi", lambda: smi)
    monkeypatch.setattr(m, "mps_disponivel", lambda: False)


def test_cpu_explicito():
    assert m.escolher_dispositivo("cpu") == ("cpu", None)


def test_gpu_pedida_com_torch_so_cpu_volta_para_cpu(monkeypatch):
    sem_gpu(monkeypatch, None, "GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU")
    disp, aviso = m.escolher_dispositivo("cuda")
    assert disp == "cpu"
    assert "só CPU" in aviso and "RTX 3060" in aviso
    assert "download.pytorch.org/whl/cu124" in aviso and "Seguindo em CPU" in aviso


def test_gpu_pedida_sem_driver(monkeypatch):
    sem_gpu(monkeypatch, None, None)
    disp, aviso = m.escolher_dispositivo("cuda")
    assert disp == "cpu" and "nvidia-smi não existe" in aviso


def test_gpu_pedida_torch_cuda_sem_placa(monkeypatch):
    sem_gpu(monkeypatch, "12.4", None)
    disp, aviso = m.escolher_dispositivo("cuda")
    assert disp == "cpu" and "CUDA 12.4" in aviso and "driver" in aviso


def test_automatico_sem_gpu_nao_avisa(monkeypatch):
    sem_gpu(monkeypatch)
    assert m.escolher_dispositivo("auto") == ("cpu", None)


def test_mps_pedido_fora_do_mac(monkeypatch):
    sem_gpu(monkeypatch)
    disp, aviso = m.escolher_dispositivo("mps")
    assert disp == "cpu" and "MPS" in aviso


def test_descricao_cpu():
    assert m.descrever_dispositivo("cpu").startswith("CPU | torch ")


def test_catalogo():
    assert set(m.CATALOGO) == {"sam2.1_tiny", "sam2.1_small", "sam2.1_base_plus", "sam2.1_large",
                               "sam1_vit_b", "sam1_vit_l", "sam1_vit_h"}
    for info in m.CATALOGO.values():
        assert len(info.sha256) == 64 and info.url.startswith("https://dl.fbaipublicfiles.com/")
        assert info.url.endswith(info.arquivo) and info.tamanho_bytes > 1e8
    assert m.CATALOGO["sam2.1_tiny"].rotulo_combo == "SAM 2.1 tiny (156 MB, 38,9 M)"
    assert m.CATALOGO["sam1_vit_h"].rotulo_combo == "SAM 1 ViT-H (2,6 GB, 636 M)"
    assert m.resolver_chave("tiny") == "sam2.1_tiny" and m.resolver_chave("vit_b") == "sam1_vit_b"
    with pytest.raises(m.ErroModelo):
        m.resolver_chave("sam9")


def test_cache_em_disco_inclui_o_modelo(tmp_path, monkeypatch):
    a = m.ModeloSAM("sam2.1_tiny")
    b = m.ModeloSAM("sam1_vit_b")
    assert a._arquivo_cache("abc").name == "abc_sam2.1_tiny.pt"
    assert a._arquivo_cache("abc") != b._arquivo_cache("abc")


def test_download_confere_sha(tmp_path, monkeypatch):
    """Download simulado: hash errado apaga o arquivo e avisa."""
    import io

    class Resposta(io.BytesIO):
        headers = {"Content-Length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(m, "PASTA_CHECKPOINTS", tmp_path)
    monkeypatch.setattr(m.urllib.request, "urlopen", lambda *a, **k: Resposta(b"abcd"))
    info = m.InfoModelo("x", "sam2", "X", "x.pt", "https://exemplo/x.pt", "0" * 64, 4, 1, "")
    with pytest.raises(m.ErroModelo, match="SHA-256"):
        m.baixar_checkpoint(info)
    assert not list(tmp_path.iterdir())
    import hashlib
    ok = m.InfoModelo("y", "sam2", "Y", "y.pt", "https://exemplo/y.pt", hashlib.sha256(b"abcd").hexdigest(), 4, 1, "")
    progresso = []
    assert m.baixar_checkpoint(ok, lambda f, t: progresso.append(f)).read_bytes() == b"abcd"
    assert progresso[-1] == 4


# ---------------------------------------------------------------- modo texto

def test_classe_para_texto():
    classes = ["objeto", "fumaca", "carro", "gato"]
    assert texto.classe_para_texto("smoke", classes) == "fumaca"
    assert texto.classe_para_texto("Cars.", classes) == "carro"
    assert texto.classe_para_texto("gato", classes) == "gato"
    assert texto.classe_para_texto("airplane", classes) is None


def test_sam3_sem_token_explica(monkeypatch):
    pytest.importorskip("transformers")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    motivo = texto.requisitos_sam3()
    assert motivo and "huggingface.co/facebook/sam3" in motivo and "hf auth login" in motivo


def test_sam3_sem_transformers(monkeypatch):
    monkeypatch.setattr(texto, "_tem", lambda pacote: False)
    assert "transformers" in texto.requisitos_sam3()
    assert "transformers" in texto.requisitos_gdino()
