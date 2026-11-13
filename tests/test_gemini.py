"""Testa a interpretação das respostas do Gemini sem chamar a API (sem gastar crédito)."""
from types import SimpleNamespace

from app.gemini import ConsultorGemini


def consultor_com_resposta(tmp_path, texto=None, erro=None):
    padroes = tmp_path / "padroes.md"
    padroes.write_text("- regra de teste")
    consultor = ConsultorGemini("chave-falsa", "gemini-teste", 5, ["cao", "gato", "nenhum"], padroes)

    def gerar(**kwargs):
        if erro:
            raise erro
        return SimpleNamespace(text=texto)

    consultor._client = SimpleNamespace(models=SimpleNamespace(generate_content=gerar))
    return consultor


def test_resposta_valida(tmp_path):
    c = consultor_com_resposta(tmp_path, '{"classe": "nenhum", "confianca": 0.87}')
    assert c.classificar(b"\xff\xd8") == {"classe": "nenhum", "confianca": 0.87}


def test_instrucoes_incluem_padroes(tmp_path):
    c = consultor_com_resposta(tmp_path, "{}")
    assert "regra de teste" in c._config.system_instruction


def test_confianca_e_limitada_entre_0_e_1(tmp_path):
    c = consultor_com_resposta(tmp_path, '{"classe": "gato", "confianca": 7}')
    assert c.classificar(b"")["confianca"] == 1.0


def test_classe_fora_da_lista(tmp_path):
    assert consultor_com_resposta(tmp_path, '{"classe": "coelho", "confianca": 0.9}').classificar(b"") is None


def test_json_invalido(tmp_path):
    assert consultor_com_resposta(tmp_path, "não é json").classificar(b"") is None


def test_erro_de_rede(tmp_path):
    assert consultor_com_resposta(tmp_path, erro=TimeoutError("demorou")).classificar(b"") is None
