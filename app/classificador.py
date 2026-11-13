"""Carrega um modelo TensorFlow Lite treinado e faz previsões.

O mesmo código é usado pela aplicação e pelo script treino/avaliar.py, para que a avaliação
meça exatamente o que roda em produção (mesmo arquivo de modelo, mesmo pré-processamento).
"""
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

try:  # runtime leve do TensorFlow Lite (Linux e macOS)
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # no Windows, usa o TensorFlow completo
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


def preparar_imagem(img: Image.Image, tamanho) -> np.ndarray:
    """Converte para RGB, redimensiona e devolve um array float32 com pixels de 0 a 255.

    A normalização para a faixa esperada pela rede fica dentro do próprio modelo.
    """
    altura, largura = tamanho
    img = img.convert("RGB").resize((largura, altura), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


class Classificador:
    def __init__(self, pasta_versao: Path):
        pasta_versao = Path(pasta_versao)
        meta = json.loads((pasta_versao / "metadados.json").read_text(encoding="utf-8"))
        self.versao = meta["versao"]
        self.classes = meta["classes"]
        self.nomes = meta["nomes"]
        self.tamanho = meta["tamanho_entrada"]

        self._interpreter = Interpreter(model_path=str(pasta_versao / "modelo.tflite"))
        self._interpreter.allocate_tensors()
        self._entrada = self._interpreter.get_input_details()[0]["index"]
        self._saida = self._interpreter.get_output_details()[0]["index"]
        # O interpretador não é seguro para várias threads ao mesmo tempo
        self._trava = threading.Lock()

    def prever_array(self, x: np.ndarray) -> np.ndarray:
        """Recebe um array (altura, largura, 3) já preparado e devolve as probabilidades."""
        with self._trava:
            self._interpreter.set_tensor(self._entrada, x[None].astype(np.float32))
            self._interpreter.invoke()
            return self._interpreter.get_tensor(self._saida)[0].copy()

    def prever(self, img: Image.Image) -> dict:
        """Devolve {classe: probabilidade} para uma imagem PIL."""
        probs = self.prever_array(preparar_imagem(img, self.tamanho))
        return {classe: float(p) for classe, p in zip(self.classes, probs)}
