import sqlite3

import pytest

from conftest import ClassificadorFalso, GeminiFalso, enviar, imagem_bytes

ALTA = {"cao": 0.95, "gato": 0.04, "nenhum": 0.01}
BAIXA = {"cao": 0.50, "gato": 0.45, "nenhum": 0.05}


def ler_casos(config):
    con = sqlite3.connect(config.banco)
    con.row_factory = sqlite3.Row
    return [dict(l) for l in con.execute("SELECT * FROM casos ORDER BY id")]


def test_pagina_e_saude(cliente):
    c = cliente()
    assert c.get("/").status_code == 200
    info = c.get("/api/saude").json()
    assert info["versao_modelo"] == "v0"
    assert info["limiar"] == 0.8
    assert info["gemini_ativo"] is False


def test_modelo_tflite_real_responde(cliente):
    resposta = enviar(cliente(), imagem_bytes())
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["classe"] in ("cao", "gato", "nenhum")
    assert sum(corpo["modelo"]["probabilidades"].values()) == pytest.approx(1, abs=1e-3)


def test_aceita_png_e_webp(cliente):
    c = cliente()
    assert enviar(c, imagem_bytes("PNG"), "f.png", "image/png").status_code == 200
    assert enviar(c, imagem_bytes("WEBP"), "f.webp", "image/webp").status_code == 200


def test_confianca_alta_nao_chama_gemini(cliente, config):
    gemini = GeminiFalso({"classe": "gato", "confianca": 0.9})
    corpo = enviar(cliente(classificador=ClassificadorFalso(ALTA), gemini=gemini), imagem_bytes()).json()
    assert corpo["fonte"] == "modelo"
    assert corpo["classe"] == "cao" and corpo["rotulo"] == "Cão"
    assert gemini.chamadas == 0
    caso = ler_casos(config)[0]
    assert caso["fonte"] == "modelo" and caso["imagem"] is None  # foto certeira não é guardada


def test_confianca_baixa_consulta_gemini(cliente, config):
    gemini = GeminiFalso({"classe": "gato", "confianca": 0.9})
    corpo = enviar(cliente(classificador=ClassificadorFalso(BAIXA), gemini=gemini), imagem_bytes()).json()
    assert gemini.chamadas == 1
    assert corpo["fonte"] == "gemini"
    assert corpo["classe"] == "gato" and corpo["confianca"] == 0.9
    assert corpo["modelo"]["classe"] == "cao"  # a opinião do modelo continua na resposta
    caso = ler_casos(config)[0]
    assert caso["classe_gemini"] == "gato" and caso["classe_modelo"] == "cao"
    assert (config.pasta_imagens / caso["imagem"]).exists()  # foto duvidosa fica para revisão


def test_falha_do_gemini_mantem_resposta_do_modelo(cliente):
    corpo = enviar(cliente(classificador=ClassificadorFalso(BAIXA), gemini=GeminiFalso(None)),
                   imagem_bytes()).json()
    assert corpo["fonte"] == "modelo_sem_gemini"
    assert corpo["classe"] == "cao"
    assert corpo["aviso"]


def test_sem_gemini_configurado(cliente):
    corpo = enviar(cliente(classificador=ClassificadorFalso(BAIXA)), imagem_bytes()).json()
    assert corpo["fonte"] == "modelo_sem_gemini" and corpo["aviso"]


def test_salvar_imagens_nenhuma(cliente, config):
    enviar(cliente(classificador=ClassificadorFalso(BAIXA), salvar_imagens="nenhuma"), imagem_bytes())
    assert ler_casos(config)[0]["imagem"] is None


def test_arquivo_que_nao_e_imagem(cliente):
    resposta = enviar(cliente(), b"isto nao e uma imagem", "x.jpg")
    assert resposta.status_code == 400


def test_formato_nao_aceito(cliente):
    resposta = enviar(cliente(), imagem_bytes("GIF"), "x.gif", "image/gif")
    assert resposta.status_code == 415


def test_arquivo_grande_demais(cliente):
    grande = imagem_bytes("PNG", tamanho=(1200, 1200), cor=(1, 2, 3)) + b"0" * 1_200_000
    assert enviar(cliente(), grande, "g.png", "image/png").status_code == 413


def test_limite_de_requisicoes(cliente):
    c = cliente(limite_por_minuto=2, classificador=ClassificadorFalso(ALTA))
    codigos = [enviar(c, imagem_bytes()).status_code for _ in range(3)]
    assert codigos == [200, 200, 429]


def test_estatisticas_na_saude(cliente):
    c = cliente(classificador=ClassificadorFalso(BAIXA))
    enviar(c, imagem_bytes())
    casos = c.get("/api/saude").json()["casos"]
    assert casos["total"] == 1 and casos["aguardando_revisao"] == 1
