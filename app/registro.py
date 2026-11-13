"""Registro de cada classificação em um banco SQLite.

Esses registros alimentam a revisão humana e o retreino: só casos com rotulo_confirmado
preenchido por uma pessoa podem virar dados de treino.
"""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

ESQUEMA = """
CREATE TABLE IF NOT EXISTS casos (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em         TEXT NOT NULL,
    versao_modelo     TEXT NOT NULL,
    classe_modelo     TEXT NOT NULL,
    confianca_modelo  REAL NOT NULL,
    probabilidades    TEXT NOT NULL,   -- JSON {classe: probabilidade}
    classe_gemini     TEXT,
    confianca_gemini  REAL,
    classe_final      TEXT NOT NULL,
    fonte             TEXT NOT NULL,   -- modelo | gemini | modelo_sem_gemini
    imagem            TEXT,            -- arquivo guardado para revisão, ou NULL
    rotulo_confirmado TEXT,            -- preenchido na revisão humana
    revisado_em       TEXT
)
"""


class Registro:
    def __init__(self, banco: Path, pasta_imagens: Path):
        self._banco = Path(banco)
        self._pasta_imagens = Path(pasta_imagens)
        self._banco.parent.mkdir(parents=True, exist_ok=True)
        self._pasta_imagens.mkdir(parents=True, exist_ok=True)
        with self._conectar() as con:
            con.execute(ESQUEMA)

    def _conectar(self):
        return sqlite3.connect(self._banco, timeout=10)

    def registrar(self, *, versao_modelo, classe_modelo, confianca_modelo, probabilidades,
                  classe_final, fonte, classe_gemini=None, confianca_gemini=None,
                  imagem_jpeg: bytes | None = None) -> int:
        nome_imagem = None
        if imagem_jpeg is not None:
            nome_imagem = f"{uuid.uuid4().hex}.jpg"
            (self._pasta_imagens / nome_imagem).write_bytes(imagem_jpeg)
        with self._conectar() as con:
            cursor = con.execute(
                """INSERT INTO casos (criado_em, versao_modelo, classe_modelo, confianca_modelo,
                       probabilidades, classe_gemini, confianca_gemini, classe_final, fonte, imagem)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(timespec="seconds"), versao_modelo, classe_modelo,
                 confianca_modelo, json.dumps(probabilidades), classe_gemini, confianca_gemini,
                 classe_final, fonte, nome_imagem))
            return cursor.lastrowid

    def estatisticas(self) -> dict:
        with self._conectar() as con:
            total = con.execute("SELECT COUNT(*) FROM casos").fetchone()[0]
            por_fonte = dict(con.execute("SELECT fonte, COUNT(*) FROM casos GROUP BY fonte").fetchall())
            pendentes = con.execute(
                "SELECT COUNT(*) FROM casos WHERE imagem IS NOT NULL AND rotulo_confirmado IS NULL"
            ).fetchone()[0]
        return {"total": total, "por_fonte": por_fonte, "aguardando_revisao": pendentes}
