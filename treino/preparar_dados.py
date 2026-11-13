"""Etapa 1 do treino: monta a base com três classes e divide em treino, validação e teste.

Fontes das imagens:
  cão e gato -> zip original da base Asirra/cats_vs_dogs (Microsoft Research), lido direto
                (o mesmo arquivo do TFDS "cats_vs_dogs", cujo gerador falha no Windows)
  nenhum     -> TensorFlow Datasets "imagenette" (objetos e cenas; a classe de cachorro é excluída)
                + opcionalmente uma pasta local com outros animais (--extra-nenhum),
                  por exemplo a base Animals-10 do Kaggle

Uso (a partir da pasta raiz do projeto):
  python treino/preparar_dados.py
  python treino/preparar_dados.py --extra-nenhum caminho/para/animals10 --n-por-classe 1500
"""
import argparse
import io
import json
import random
import re
import shutil
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from comum import CLASSES, CONJUNTOS, SEED

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp"}

# Nomes de pasta que indicam cão ou gato e nunca podem entrar na classe "nenhum".
# Inclui os nomes em italiano usados pela base Animals-10 ("cane" e "gatto").
PASTAS_PROIBIDAS = {"cane", "gatto", "dog", "dogs", "cat", "cats", "cao", "gato", "cachorro"}

# Classe da Imagenette que é uma raça de cachorro (English springer)
IMAGENETTE_CACHORRO = "n02102040"

# Zip original da base Asirra (o mesmo que o TFDS "cats_vs_dogs" baixa)
URL_PETS = ("https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/"
            "kagglecatsanddogs_5340.zip")


def salvar_jpeg(origem, destino):
    """Salva uma imagem (array NumPy ou caminho) como JPEG RGB."""
    if isinstance(origem, np.ndarray):
        if origem.ndim == 3 and origem.shape[-1] == 1:
            origem = origem[..., 0]
        img = Image.fromarray(origem)
    else:
        img = Image.open(origem)
    img.convert("RGB").save(destino, "JPEG", quality=95)


def baixar_zip_pets():
    """Devolve o caminho do zip original do cats_vs_dogs, baixando só se ainda não existir.

    Reaproveita o zip que o TFDS já tenha baixado em ~/tensorflow_datasets/downloads.
    """
    cache_tfds = Path.home() / "tensorflow_datasets" / "downloads" / "cats_vs_dogs"
    candidatos = [p for p in cache_tfds.glob("*.zip") if p.stat().st_size > 800_000_000]
    if candidatos:
        return candidatos[0]
    destino = Path.home() / ".cache" / "classificador-pets" / "kagglecatsanddogs_5340.zip"
    if destino.exists() and destino.stat().st_size > 800_000_000:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"  baixando {URL_PETS} (~790 MB)...")
    parcial = destino.with_suffix(".parcial")
    urllib.request.urlretrieve(URL_PETS, parcial)
    parcial.replace(destino)
    return destino


def coletar_pets(metas, pasta_bruta, prefixo):
    """Lê cães e gatos direto do zip original (PetImages/Cat e PetImages/Dog).

    Não usa o TFDS aqui porque o gerador do cats_vs_dogs falha no Windows
    ("There is no item named 'PetImages\\\\Cat\\\\0.jpg' in the archive").
    """
    padrao = re.compile(r"^PetImages/(Cat|Dog)/\d+\.jpg$")
    mapa = {"Cat": "gato", "Dog": "cao"}
    contagem = {c: 0 for c in metas}
    with zipfile.ZipFile(baixar_zip_pets()) as zf:
        nomes = sorted(n for n in zf.namelist() if padrao.match(n.replace("\\", "/")))
        random.Random(SEED).shuffle(nomes)
        for nome in nomes:
            if all(contagem[c] >= metas[c] for c in metas):
                break
            classe = mapa[padrao.match(nome.replace("\\", "/")).group(1)]
            if contagem[classe] >= metas[classe]:
                continue
            try:
                with Image.open(io.BytesIO(zf.read(nome))) as img:
                    img.load()
                    if min(img.size) < 32:
                        continue
                    img.convert("RGB").save(
                        pasta_bruta / classe / f"{prefixo}_{contagem[classe]:05d}.jpg", "JPEG", quality=95)
            except Exception:
                continue  # a base original tem alguns arquivos corrompidos
            contagem[classe] += 1
    for c in metas:
        if contagem[c] < metas[c]:
            print(f"  aviso: só {contagem[c]} imagens para '{c}' (meta {metas[c]})")
    return contagem


def gerar_nenhum_sintetico(n, pasta_bruta):
    """Gera n imagens artificiais (degradês, formas e ruído) para a classe 'nenhum', sem download."""
    from PIL import ImageDraw

    rng = np.random.default_rng(SEED)
    lado = 224
    for i in range(n):
        tipo = i % 3
        if tipo == 0:  # degradê entre duas cores
            c1, c2 = rng.integers(0, 256, (2, 3))
            t = np.linspace(0, 1, lado)[:, None, None]
            arr = (c1 * (1 - t) + c2 * t).repeat(lado, axis=1)
        elif tipo == 1:  # ruído colorido suavizado
            pequeno = rng.integers(0, 256, (8, 8, 3)).astype(np.uint8)
            arr = np.asarray(Image.fromarray(pequeno).resize((lado, lado), Image.BICUBIC))
        else:  # fundo liso
            arr = np.full((lado, lado, 3), rng.integers(0, 256, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        desenho = ImageDraw.Draw(img)
        for _ in range(int(rng.integers(1, 6))):  # formas geométricas por cima
            x0, y0 = rng.integers(0, lado - 20, 2)
            x1, y1 = x0 + rng.integers(20, 120), y0 + rng.integers(20, 120)
            cor = tuple(int(v) for v in rng.integers(0, 256, 3))
            forma = desenho.ellipse if rng.random() < 0.5 else desenho.rectangle
            forma([int(x0), int(y0), int(x1), int(y1)], fill=cor)
        img.save(pasta_bruta / "nenhum" / f"sint_{i:05d}.jpg", "JPEG", quality=90)


def coletar_tfds(nome_ds, mapear, metas, pasta_bruta, prefixo):
    """Lê um dataset do TFDS embaralhado e salva imagens até atingir a meta de cada classe."""
    import tensorflow_datasets as tfds

    ds, info = tfds.load(nome_ds, split="train", as_supervised=True, with_info=True)
    nomes_rotulos = info.features["label"].names
    contagem = {c: 0 for c in metas}

    def completo():
        return all(contagem[c] >= metas[c] for c in metas)

    for imagem, rotulo in ds.shuffle(5000, seed=SEED).as_numpy_iterator():
        if completo():
            break
        classe = mapear(nomes_rotulos[int(rotulo)])
        if classe is None or contagem[classe] >= metas[classe]:
            continue
        destino = pasta_bruta / classe / f"{prefixo}_{contagem[classe]:05d}.jpg"
        salvar_jpeg(imagem, destino)
        contagem[classe] += 1

    for c in metas:
        if contagem[c] < metas[c]:
            print(f"  aviso: {nome_ds} tinha só {contagem[c]} imagens para '{c}' (meta {metas[c]})")
    return contagem


def coletar_pasta_extra(pasta, n, pasta_bruta):
    """Copia até n imagens de uma pasta local (com subpastas) para a classe 'nenhum'."""
    arquivos = [
        p for p in Path(pasta).rglob("*")
        if p.suffix.lower() in EXTENSOES
        and not any(parte.lower() in PASTAS_PROIBIDAS for parte in p.relative_to(pasta).parts[:-1])
    ]
    arquivos.sort()
    random.Random(SEED).shuffle(arquivos)
    copiados = 0
    for origem in arquivos:
        if copiados >= n:
            break
        try:
            salvar_jpeg(origem, pasta_bruta / "nenhum" / f"extra_{copiados:05d}.jpg")
            copiados += 1
        except Exception as erro:
            print(f"  ignorada {origem.name}: {erro}")
    return copiados


def limpar(pasta_bruta):
    """Remove arquivos corrompidos, fora do padrão RGB ou pequenos demais."""
    removidas = []
    for caminho in sorted(pasta_bruta.rglob("*.jpg")):
        try:
            with Image.open(caminho) as img:
                img.verify()
            with Image.open(caminho) as img:  # é preciso reabrir depois de verify()
                if img.mode != "RGB" or min(img.size) < 32:
                    raise ValueError(f"modo {img.mode}, tamanho {img.size}")
        except Exception as erro:
            caminho.unlink()
            removidas.append(f"{caminho.name}: {erro}")
    return removidas


def dividir(pasta_bruta, pasta_dados):
    """Divisão estratificada 70/15/15, feita separadamente para cada classe."""
    resumo = {conj: {} for conj in CONJUNTOS}
    for classe in CLASSES:
        arquivos = sorted(p.name for p in (pasta_bruta / classe).glob("*.jpg"))
        treino, resto = train_test_split(arquivos, test_size=0.30, random_state=SEED)
        validacao, teste = train_test_split(resto, test_size=0.50, random_state=SEED)
        for conj, lista in zip(CONJUNTOS, [treino, validacao, teste]):
            destino = pasta_dados / conj / classe
            destino.mkdir(parents=True, exist_ok=True)
            for nome in lista:
                shutil.copy(pasta_bruta / classe / nome, destino / nome)
            resumo[conj][classe] = len(lista)

    # Nenhuma imagem pode estar em mais de um conjunto
    por_conjunto = {c: {p.name for p in (pasta_dados / c).rglob("*.jpg")} for c in CONJUNTOS}
    for i, a in enumerate(CONJUNTOS):
        for b in CONJUNTOS[i + 1:]:
            assert not (por_conjunto[a] & por_conjunto[b]), f"imagens repetidas entre {a} e {b}"
    return resumo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-por-classe", type=int, default=1500, help="imagens por classe (padrão 1500)")
    parser.add_argument("--extra-nenhum", type=Path, help="pasta com outros animais para a classe 'nenhum'")
    parser.add_argument("--fracao-extra", type=float, default=0.5,
                        help="fração da classe 'nenhum' vinda da pasta extra (padrão 0.5)")
    parser.add_argument("--saida", type=Path, default=Path("dados"), help="pasta de saída (padrão: dados)")
    parser.add_argument("--sobrescrever", action="store_true",
                        help="apaga uma divisão existente (isso troca o conjunto de teste!)")
    parser.add_argument("--nenhum-sintetico", action="store_true",
                        help="gera imagens artificiais para 'nenhum' em vez de baixar a Imagenette (só teste)")
    args = parser.parse_args()

    if (args.saida / "teste").exists() and not args.sobrescrever:
        raise SystemExit(
            f"A pasta '{args.saida}/teste' já existe. O conjunto de teste deve ficar fixo entre versões do\n"
            "modelo, senão as métricas deixam de ser comparáveis. Use --sobrescrever só se tiver certeza.")

    pasta_bruta = args.saida.parent / f"{args.saida.name}_brutos"
    shutil.rmtree(pasta_bruta, ignore_errors=True)
    shutil.rmtree(args.saida, ignore_errors=True)
    for c in CLASSES:
        (pasta_bruta / c).mkdir(parents=True)
    n = args.n_por_classe

    print("1/4 Coletando cães e gatos (cats_vs_dogs)...")
    coletar_pets({"cao": n, "gato": n}, pasta_bruta, "pet")

    print("2/4 Coletando a classe 'nenhum'...")
    n_extra = 0
    if args.extra_nenhum:
        n_extra = coletar_pasta_extra(args.extra_nenhum, int(n * args.fracao_extra), pasta_bruta)
        print(f"  {n_extra} imagens da pasta extra")
    else:
        print("  dica: use --extra-nenhum com fotos de outros animais (ex.: Animals-10) para o modelo")
        print("  aprender a separar cães e gatos de bichos parecidos, como raposas e coelhos.")
    if args.nenhum_sintetico:
        print("  AVISO: gerando imagens artificiais para 'nenhum' (sem baixar a Imagenette).")
        print("  Serve só para ver a aplicação funcionando: fotos reais sem cão nem gato tendem a")
        print("  sair como cão ou gato. Para o trabalho, rode de novo sem --nenhum-sintetico.")
        gerar_nenhum_sintetico(n - n_extra, pasta_bruta)
    else:
        coletar_tfds("imagenette/320px-v2",
                     lambda rotulo: None if rotulo == IMAGENETTE_CACHORRO else "nenhum",
                     {"nenhum": n - n_extra}, pasta_bruta, "obj")

    print("3/4 Limpando imagens...")
    removidas = limpar(pasta_bruta)
    print(f"  removidas: {len(removidas)}")
    for linha in removidas:
        print("   -", linha)

    print("4/4 Dividindo em treino, validação e teste (70/15/15)...")
    resumo = dividir(pasta_bruta, args.saida)
    shutil.rmtree(pasta_bruta)

    info = {
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "classes": CLASSES,
        "divisao": resumo,
        "imagens_removidas": removidas,
        "fontes": {
            "cao_gato": "zip original cats_vs_dogs (Asirra)",
            "nenhum": (["imagens sintéticas (cores, formas e ruído) - só para teste"] if args.nenhum_sintetico
                       else ["tfds:imagenette/320px-v2 (sem a classe de cachorro)"])
                      + ([f"pasta local: {args.extra_nenhum} ({n_extra} imagens)"] if n_extra else []),
        },
    }
    (args.saida / "resumo.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDivisão final:")
    for conj in CONJUNTOS:
        partes = ", ".join(f"{c}={resumo[conj][c]}" for c in CLASSES)
        print(f"  {conj:<10} {sum(resumo[conj].values()):>5}  ({partes})")
    print(f"\nPronto. Resumo salvo em {args.saida / 'resumo.json'}")


if __name__ == "__main__":
    main()
