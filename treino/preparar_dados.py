"""Etapa 1 do treino: monta a base com três classes e divide em treino, validação e teste.

Fontes das imagens:
  cão e gato -> TensorFlow Datasets "cats_vs_dogs" (projeto Asirra, Microsoft Research)
  nenhum     -> TensorFlow Datasets "imagenette" (objetos e cenas; a classe de cachorro é excluída)
                + opcionalmente uma pasta local com outros animais (--extra-nenhum),
                  por exemplo a base Animals-10 do Kaggle

Uso (a partir da pasta raiz do projeto):
  python treino/preparar_dados.py
  python treino/preparar_dados.py --extra-nenhum caminho/para/animals10 --n-por-classe 1500
"""
import argparse
import json
import random
import shutil
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


def salvar_jpeg(origem, destino):
    """Salva uma imagem (array NumPy ou caminho) como JPEG RGB."""
    if isinstance(origem, np.ndarray):
        if origem.ndim == 3 and origem.shape[-1] == 1:
            origem = origem[..., 0]
        img = Image.fromarray(origem)
    else:
        img = Image.open(origem)
    img.convert("RGB").save(destino, "JPEG", quality=95)


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
    mapa_pets = {"cat": "gato", "dog": "cao"}
    coletar_tfds("cats_vs_dogs", mapa_pets.get, {"cao": n, "gato": n}, pasta_bruta, "pet")

    print("2/4 Coletando a classe 'nenhum'...")
    n_extra = 0
    if args.extra_nenhum:
        n_extra = coletar_pasta_extra(args.extra_nenhum, int(n * args.fracao_extra), pasta_bruta)
        print(f"  {n_extra} imagens da pasta extra")
    else:
        print("  dica: use --extra-nenhum com fotos de outros animais (ex.: Animals-10) para o modelo")
        print("  aprender a separar cães e gatos de bichos parecidos, como raposas e coelhos.")
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
            "cao_gato": "tfds:cats_vs_dogs",
            "nenhum": ["tfds:imagenette/320px-v2 (sem a classe de cachorro)"]
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
