"""API e página web do classificador de cães e gatos.

Fluxo de cada foto:
  1. valida o arquivo (tamanho, formato, integridade) e corrige a rotação de fotos de celular;
  2. o modelo próprio (TensorFlow Lite) classifica em cão, gato ou nenhum;
  3. se a confiança ficar abaixo do limiar, consulta o Gemini (quando configurado);
  4. registra o caso no banco e guarda a imagem se ela precisar de revisão.

Execução local (a partir da raiz do projeto):
  uvicorn app.main:criar_app --factory --reload
"""
import io
import logging
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from app.classificador import Classificador
from app.config import RAIZ, Config, carregar_config, resolver_limiar
from app.registro import Registro

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("classificador")

FORMATOS_ACEITOS = {"JPEG", "PNG", "WEBP"}
LADO_MAXIMO_GEMINI = 768  # reduz a imagem antes de enviar: menos tokens, menor custo, mesma decisão
PASTA_STATIC = RAIZ / "app" / "static"


class LimitadorDeTaxa:
    """Limita requisições por IP numa janela de 60 segundos (em memória, por processo)."""

    def __init__(self, por_minuto: int):
        self._limite = por_minuto
        self._acessos = defaultdict(deque)
        self._trava = threading.Lock()

    def permitir(self, chave: str) -> bool:
        agora = time.monotonic()
        with self._trava:
            fila = self._acessos[chave]
            while fila and agora - fila[0] > 60:
                fila.popleft()
            if len(fila) >= self._limite:
                return False
            fila.append(agora)
            return True


def abrir_imagem(conteudo: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(conteudo)) as img:
            img.verify()
        img = Image.open(io.BytesIO(conteudo))
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(400, "O arquivo enviado não é uma imagem válida ou está corrompido.")
    if img.format not in FORMATOS_ACEITOS:
        raise HTTPException(415, "Formato não aceito. Envie uma foto em JPEG, PNG ou WebP.")
    img = ImageOps.exif_transpose(img)  # fotos de celular vêm com a rotação só nos metadados
    return img.convert("RGB")


def para_jpeg(img: Image.Image, lado_maximo: int | None = None) -> bytes:
    if lado_maximo:
        img = img.copy()
        img.thumbnail((lado_maximo, lado_maximo))
    saida = io.BytesIO()
    img.save(saida, "JPEG", quality=90)
    return saida.getvalue()


def criar_app(config: Config | None = None, classificador=None, gemini=None, registro=None) -> FastAPI:
    """Monta a aplicação. Os parâmetros opcionais permitem injetar substitutos nos testes."""
    config = config or carregar_config()
    pasta_versao = config.pasta_modelos / config.modelo_ativo
    if classificador is None:
        if not (pasta_versao / "modelo.tflite").exists():
            raise RuntimeError(
                f"Modelo não encontrado em {pasta_versao}. Treine com treino/treinar.py "
                f"ou ajuste MODELO_ATIVO no .env.")
        classificador = Classificador(pasta_versao)
    limiar = resolver_limiar(config, pasta_versao)
    if gemini is None and config.gemini_api_key:
        from app.gemini import ConsultorGemini
        gemini = ConsultorGemini(config.gemini_api_key, config.modelo_gemini, config.timeout_gemini_s,
                                 classificador.classes, config.arquivo_padroes)
    registro = registro or Registro(config.banco, config.pasta_imagens)
    limitador = LimitadorDeTaxa(config.limite_por_minuto)

    log.info("Modelo %s carregado | limiar %.2f | Gemini %s",
             classificador.versao, limiar, f"ativo ({config.modelo_gemini})" if gemini else "desativado")

    app = FastAPI(title="Classificador de cães e gatos", version=classificador.versao)
    app.mount("/static", StaticFiles(directory=PASTA_STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def pagina_inicial():
        return FileResponse(PASTA_STATIC / "index.html")

    @app.get("/api/saude")
    def saude():
        return {
            "status": "ok",
            "versao_modelo": classificador.versao,
            "classes": classificador.nomes,
            "limiar": limiar,
            "gemini_ativo": gemini is not None,
            "casos": registro.estatisticas(),
        }

    @app.post("/api/classificar")
    def classificar(request: Request, foto: UploadFile = File(...)):
        ip = request.client.host if request.client else "desconhecido"
        if not limitador.permitir(ip):
            raise HTTPException(429, "Muitas fotos em pouco tempo. Aguarde um minuto e tente de novo.")

        conteudo = foto.file.read(int(config.max_mb * 1024 * 1024) + 1)
        if len(conteudo) > config.max_mb * 1024 * 1024:
            raise HTTPException(413, f"A foto passa do limite de {config.max_mb:g} MB.")
        if not conteudo:
            raise HTTPException(400, "Nenhum arquivo foi enviado.")
        img = abrir_imagem(conteudo)

        probabilidades = classificador.prever(img)
        classe_modelo = max(probabilidades, key=probabilidades.get)
        confianca_modelo = probabilidades[classe_modelo]
        duvidosa = confianca_modelo < limiar

        classe_final, confianca_final, fonte, aviso, resposta_gemini = (
            classe_modelo, confianca_modelo, "modelo", None, None)
        if duvidosa:
            if gemini is not None:
                resposta_gemini = gemini.classificar(para_jpeg(img, LADO_MAXIMO_GEMINI))
            if resposta_gemini:
                classe_final, confianca_final, fonte = (
                    resposta_gemini["classe"], resposta_gemini["confianca"], "gemini")
            else:
                fonte = "modelo_sem_gemini"
                aviso = "O modelo ficou em dúvida nesta foto. Confira o resultado."

        guardar = config.salvar_imagens == "todas" or (config.salvar_imagens == "duvidosas" and duvidosa)
        id_caso = registro.registrar(
            versao_modelo=classificador.versao,
            classe_modelo=classe_modelo,
            confianca_modelo=confianca_modelo,
            probabilidades=probabilidades,
            classe_gemini=resposta_gemini["classe"] if resposta_gemini else None,
            confianca_gemini=resposta_gemini["confianca"] if resposta_gemini else None,
            classe_final=classe_final,
            fonte=fonte,
            imagem_jpeg=para_jpeg(img) if guardar else None,
        )

        return {
            "id": id_caso,
            "classe": classe_final,
            "rotulo": classificador.nomes[classe_final],
            "confianca": round(confianca_final, 4),
            "fonte": fonte,
            "aviso": aviso,
            "modelo": {
                "versao": classificador.versao,
                "classe": classe_modelo,
                "confianca": round(confianca_modelo, 4),
                "probabilidades": {c: round(p, 4) for c, p in probabilidades.items()},
            },
            "gemini": resposta_gemini,
            "limiar": limiar,
        }

    return app
