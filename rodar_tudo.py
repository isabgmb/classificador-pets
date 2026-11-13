"""Faz tudo localmente, sem Colab: instala, prepara os dados, treina, avalia e sobe a aplicação.

Cada etapa é pulada se já estiver pronta, então nas próximas vezes o script vai direto
para a aplicação.

Uso (a partir da pasta raiz do projeto; no Windows, prefira o rodar_tudo.bat):
  python rodar_tudo.py                 # fluxo completo (1500 imagens por classe)
  python rodar_tudo.py --rapido        # primeiro teste: 500 imagens por classe e menos épocas
  python rodar_tudo.py --novo-treino   # treina uma nova versão (v2, v3...) e compara com a anterior
  python rodar_tudo.py --sem-app       # para depois da avaliação, sem subir o servidor
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PY = sys.executable
AMBIENTE = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "TF_CPP_MIN_LOG_LEVEL": "2",  # esconde os avisos de inicialização do TensorFlow
}

# módulo para importar -> pacote no requirements
MODULOS = {
    "tensorflow": "tensorflow",
    "tensorflow_datasets": "tensorflow-datasets",
    "importlib_resources": "importlib_resources",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "multipart": "python-multipart",
    "dotenv": "python-dotenv",
    "PIL": "pillow",
    "numpy": "numpy",
    "google.genai": "google-genai",
    "pytest": "pytest",
    "httpx": "httpx",
}


def titulo(texto):
    print(f"\n{'=' * 60}\n {texto}\n{'=' * 60}", flush=True)


def rodar(*args):
    """Executa um comando com o mesmo Python deste script e para tudo se ele falhar."""
    print(">", " ".join(str(a) for a in args), flush=True)
    resultado = subprocess.run([PY, *map(str, args)], cwd=RAIZ, env=AMBIENTE)
    if resultado.returncode != 0:
        raise SystemExit(f"\nA etapa falhou (código {resultado.returncode}). Veja a mensagem acima.")


def versoes_prontas():
    """Versões em modelos/ que já têm o modelo.tflite, em ordem numérica."""
    versoes = []
    for p in (RAIZ / "modelos").glob("v*"):
        m = re.fullmatch(r"v(\d+)", p.name)
        if m and (p / "modelo.tflite").exists():
            versoes.append((int(m.group(1)), p.name))
    return [nome for _, nome in sorted(versoes)]


def etapa_ambiente():
    titulo("1/6 Conferindo o Python e as dependências")
    print("Python em uso:", PY)
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        print(f"  aviso: o projeto foi testado com Python 3.11 ou 3.12 (este é {sys.version.split()[0]})")
    if "envs" not in Path(PY).parts and ".venv" not in Path(PY).parts:
        print("  aviso: este parece ser o Python base do Anaconda, não o ambiente 'pets'.")
        print("  Use o rodar_tudo.bat, que escolhe o Python certo sozinho.")

    faltando = [pacote for modulo, pacote in MODULOS.items() if not _tem_modulo(modulo)]
    if not faltando:
        print("Todas as dependências já estão instaladas.")
        return
    print("Faltando:", ", ".join(faltando))
    rodar("-m", "pip", "install", "--upgrade", "pip")
    rodar("-m", "pip", "install", "-r", "requirements-treino.txt")


def _tem_modulo(nome):
    try:
        return importlib.util.find_spec(nome) is not None
    except ModuleNotFoundError:
        return False


def etapa_dados(args):
    titulo("2/6 Preparando as imagens")
    if (RAIZ / "dados" / "teste").exists():
        print("A pasta dados/ já está pronta. Pulando (apague a pasta dados para refazer).")
        return
    print("Na primeira vez, baixa cerca de 1 GB. Pode demorar.")
    comando = ["treino/preparar_dados.py"]
    if args.rapido:
        comando += ["--n-por-classe", "500"]
    if args.extra_nenhum:
        comando += ["--extra-nenhum", args.extra_nenhum]
    rodar(*comando)


def etapa_treino(args):
    titulo("3/6 Treinando o modelo")
    prontas = versoes_prontas()
    if prontas and not args.novo_treino:
        print(f"Já existe um modelo treinado ({prontas[-1]}). Pulando (use --novo-treino para outra versão).")
        return prontas[-1], None

    # Remove versões que ficaram pela metade (sem modelo.tflite), para não bloquear o treino
    for p in (RAIZ / "modelos").glob("v*"):
        if p.is_dir() and re.fullmatch(r"v\d+", p.name) and not (p / "modelo.tflite").exists():
            print(f"Removendo versão incompleta: {p.name}")
            shutil.rmtree(p)

    print("Sem GPU, o treino leva de 30 minutos a mais de 1 hora.")
    comando = ["treino/treinar.py"]
    if args.rapido:
        comando += ["--epocas", "4", "--epocas-fino", "3"]
    rodar(*comando)
    nova = versoes_prontas()[-1]
    anterior = prontas[-1] if prontas else None
    return nova, anterior


def etapa_avaliacao(versao, anterior):
    titulo(f"4/6 Avaliando o modelo {versao}")
    if (RAIZ / "modelos" / versao / "metricas.json").exists():
        print("Já avaliado. Resultados em", f"modelos/{versao}/metricas.json")
        return
    comando = ["treino/avaliar.py", "--versao", versao]
    if anterior and (RAIZ / "modelos" / anterior / "metricas.json").exists():
        comando += ["--comparar-com", anterior]
    rodar(*comando)
    print("\nGuarde a saída acima (acurácia, matriz de confusão e limiar) para o documento do trabalho.")


def etapa_env(versao, anterior):
    titulo("5/6 Conferindo o arquivo .env")
    env = RAIZ / ".env"
    if not env.exists():
        texto = (RAIZ / ".env.example").read_text(encoding="utf-8")
        texto = re.sub(r"(?m)^MODELO_ATIVO=.*$", f"MODELO_ATIVO={versao}", texto)
        env.write_text(texto, encoding="utf-8")
        print(f"Criado o .env com MODELO_ATIVO={versao}.")
        print("Para ativar o Gemini, coloque a chave em GEMINI_API_KEY no .env (opcional).")
        return

    texto = env.read_text(encoding="utf-8")
    m = re.search(r"(?m)^MODELO_ATIVO=(.*)$", texto)
    ativo = m.group(1).strip() if m else "v1"
    if not (RAIZ / "modelos" / ativo / "modelo.tflite").exists():
        texto = (re.sub(r"(?m)^MODELO_ATIVO=.*$", f"MODELO_ATIVO={versao}", texto) if m
                 else texto + f"\nMODELO_ATIVO={versao}\n")
        env.write_text(texto, encoding="utf-8")
        print(f"O .env apontava para {ativo}, que não existe. Trocado para MODELO_ATIVO={versao}.")
    else:
        print(f"Modelo ativo: {ativo}")
        if anterior and ativo != versao:
            print(f"A nova versão {versao} foi treinada. Se ela for melhor na comparação acima,")
            print(f"troque para MODELO_ATIVO={versao} no .env e rode de novo.")


def etapa_app():
    titulo("6/6 Subindo a aplicação")
    url = "http://localhost:8000"
    print(f"Abrindo {url} no navegador. Para encerrar, pressione Ctrl+C nesta janela.")
    threading.Thread(target=lambda: (time.sleep(6), webbrowser.open(url)), daemon=True).start()
    try:
        subprocess.run([PY, "-m", "uvicorn", "app.main:criar_app", "--factory",
                        "--host", "127.0.0.1", "--port", "8000"], cwd=RAIZ, env=AMBIENTE)
    except KeyboardInterrupt:
        pass
    print("\nAplicação encerrada.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rapido", action="store_true", help="500 imagens por classe e menos épocas")
    parser.add_argument("--novo-treino", action="store_true", help="treina uma nova versão do modelo")
    parser.add_argument("--extra-nenhum", help="pasta raw-img da Animals-10 para a classe 'nenhum'")
    parser.add_argument("--sem-app", action="store_true", help="não sobe o servidor no final")
    args = parser.parse_args()

    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    os.chdir(RAIZ)
    etapa_ambiente()
    etapa_dados(args)
    versao, anterior = etapa_treino(args)
    etapa_avaliacao(versao, anterior)
    etapa_env(versao, anterior)
    if not args.sem_app:
        etapa_app()


if __name__ == "__main__":
    main()
