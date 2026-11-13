"""Consulta ao Gemini, usada só quando o modelo próprio fica com confiança baixa.

Este é o único arquivo que conhece o Gemini. Para trocar de provedor, basta outro módulo
com o mesmo método classificar(jpeg_bytes) -> dict | None.
"""
import json
import logging
from pathlib import Path

log = logging.getLogger("classificador.gemini")

INSTRUCOES_BASE = """Você classifica fotos para um sistema que reconhece apenas cães e gatos.
Responda com uma única classe:
- "cao": a foto mostra um cão (cachorro) real;
- "gato": a foto mostra um gato real;
- "nenhum": não há cão nem gato real na foto (outro animal, pessoa, objeto, paisagem, desenho).
Em "confianca", informe de 0 a 1 o quanto você tem certeza da resposta."""


class ConsultorGemini:
    def __init__(self, api_key: str, modelo: str, timeout_s: float, classes: list[str], arquivo_padroes: Path):
        from google import genai
        from google.genai import types

        self._types = types
        self._modelo = modelo
        self._classes = classes
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

        instrucoes = INSTRUCOES_BASE
        if arquivo_padroes.exists():
            instrucoes += "\n\n" + arquivo_padroes.read_text(encoding="utf-8").strip()
        self._config = types.GenerateContentConfig(
            system_instruction=instrucoes,
            temperature=0,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "classe": {"type": "STRING", "enum": classes},
                    "confianca": {"type": "NUMBER"},
                },
                "required": ["classe", "confianca"],
            },
        )

    def classificar(self, jpeg_bytes: bytes) -> dict | None:
        """Devolve {"classe": ..., "confianca": ...} ou None se a consulta falhar."""
        try:
            resposta = self._client.models.generate_content(
                model=self._modelo,
                contents=[
                    self._types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    "Qual é a classe desta foto?",
                ],
                config=self._config,
            )
            dados = json.loads(resposta.text)
            classe = dados["classe"]
            confianca = min(max(float(dados["confianca"]), 0.0), 1.0)
            if classe not in self._classes:
                raise ValueError(f"classe inesperada: {classe}")
            return {"classe": classe, "confianca": confianca}
        except Exception as erro:  # rede, cota, timeout ou resposta fora do formato
            log.warning("Falha ao consultar o Gemini: %s", erro)
            return None
