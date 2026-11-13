"""Configuração da aplicação, lida de variáveis de ambiente (arquivo .env na raiz do projeto)."""
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")


def _caminho(nome, padrao):
    p = Path(os.getenv(nome) or padrao)
    return p if p.is_absolute() else RAIZ / p


@dataclass(frozen=True)
class Config:
    pasta_modelos: Path
    modelo_ativo: str
    limiar: float | None          # None = usa o limiar sugerido no metricas.json do modelo
    gemini_api_key: str | None    # None = fallback para o Gemini desativado
    modelo_gemini: str
    timeout_gemini_s: float
    arquivo_padroes: Path
    banco: Path
    pasta_imagens: Path
    salvar_imagens: str           # "duvidosas", "todas" ou "nenhuma"
    max_mb: float
    limite_por_minuto: int


def carregar_config() -> Config:
    limiar = os.getenv("LIMIAR")
    salvar = os.getenv("SALVAR_IMAGENS", "duvidosas")
    if salvar not in {"duvidosas", "todas", "nenhuma"}:
        raise ValueError("SALVAR_IMAGENS deve ser 'duvidosas', 'todas' ou 'nenhuma'")
    return Config(
        pasta_modelos=_caminho("PASTA_MODELOS", "modelos"),
        modelo_ativo=os.getenv("MODELO_ATIVO", "v1"),
        limiar=float(limiar) if limiar else None,
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        modelo_gemini=os.getenv("MODELO_GEMINI", "gemini-3.5-flash-lite"),
        timeout_gemini_s=float(os.getenv("TIMEOUT_GEMINI_S", "15")),
        arquivo_padroes=_caminho("ARQUIVO_PADROES", "app/padroes.md"),
        banco=_caminho("BANCO", "dados_app/registros.db"),
        pasta_imagens=_caminho("PASTA_IMAGENS", "dados_app/imagens"),
        salvar_imagens=salvar,
        max_mb=float(os.getenv("MAX_MB", "10")),
        limite_por_minuto=int(os.getenv("LIMITE_POR_MINUTO", "30")),
    )


def resolver_limiar(config: Config, pasta_versao: Path) -> float:
    """LIMIAR do .env tem prioridade; senão, o limiar calibrado pelo avaliar.py; senão, 0.85."""
    if config.limiar is not None:
        return config.limiar
    metricas = pasta_versao / "metricas.json"
    if metricas.exists():
        return float(json.loads(metricas.read_text(encoding="utf-8"))["limiar"]["sugerido"])
    return 0.85
