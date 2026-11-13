"""Fixtures dos testes: um modelo TFLite minúsculo, falsos do classificador e do Gemini."""
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Config
from app.main import criar_app

CLASSES = ["cao", "gato", "nenhum"]
NOMES = {"cao": "Cão", "gato": "Gato", "nenhum": "Nem cão nem gato"}


@pytest.fixture(scope="session")
def pasta_modelos(tmp_path_factory):
    """Cria modelos/v0 com um modelo TFLite pequeno (não treinado) e seus metadados."""
    import tensorflow as tf
    from tensorflow import keras

    raiz = tmp_path_factory.mktemp("modelos")
    pasta = raiz / "v0"
    pasta.mkdir()
    entrada = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Rescaling(1 / 255)(entrada)
    x = keras.layers.GlobalAveragePooling2D()(x)
    saida = keras.layers.Dense(3, activation="softmax")(x)
    modelo = keras.Model(entrada, saida)
    with tempfile.TemporaryDirectory() as tmp:
        modelo.export(tmp, verbose=False)
        (pasta / "modelo.tflite").write_bytes(tf.lite.TFLiteConverter.from_saved_model(tmp).convert())
    (pasta / "metadados.json").write_text(json.dumps({
        "versao": "v0", "classes": CLASSES, "nomes": NOMES, "tamanho_entrada": [32, 32]}))
    return raiz


class ClassificadorFalso:
    """Devolve sempre as probabilidades definidas no teste."""
    versao = "teste"
    classes = CLASSES
    nomes = NOMES

    def __init__(self, probabilidades):
        self.probabilidades = probabilidades

    def prever(self, img):
        return dict(self.probabilidades)


class GeminiFalso:
    def __init__(self, resposta):
        self.resposta = resposta
        self.chamadas = 0

    def classificar(self, jpeg_bytes):
        self.chamadas += 1
        assert jpeg_bytes[:2] == b"\xff\xd8"  # recebeu um JPEG
        return self.resposta


@pytest.fixture
def config(tmp_path, pasta_modelos):
    return Config(
        pasta_modelos=pasta_modelos, modelo_ativo="v0", limiar=0.8, gemini_api_key=None,
        modelo_gemini="gemini-teste", timeout_gemini_s=5, arquivo_padroes=tmp_path / "padroes.md",
        banco=tmp_path / "registros.db", pasta_imagens=tmp_path / "imagens",
        salvar_imagens="duvidosas", max_mb=1, limite_por_minuto=100)


@pytest.fixture
def cliente(config):
    """Fábrica: cliente(classificador=..., gemini=..., **mudancas_na_config)."""
    def _criar(**kwargs):
        mudancas = {k: v for k, v in kwargs.items() if k not in ("classificador", "gemini")}
        cfg = Config(**{**config.__dict__, **mudancas})
        app = criar_app(cfg, classificador=kwargs.get("classificador"), gemini=kwargs.get("gemini"))
        return TestClient(app)
    return _criar


def imagem_bytes(formato="JPEG", tamanho=(64, 48), cor=(200, 120, 40)):
    saida = io.BytesIO()
    Image.new("RGB", tamanho, cor).save(saida, formato)
    return saida.getvalue()


def enviar(cliente_http, conteudo, nome="foto.jpg", tipo="image/jpeg"):
    return cliente_http.post("/api/classificar", files={"foto": (nome, conteudo, tipo)})
