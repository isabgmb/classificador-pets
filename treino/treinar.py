"""Etapa 2 do treino: treina o classificador de três classes e exporta para TensorFlow Lite.

Modelo: transferência de aprendizado com MobileNetV2 (pré-treinado no ImageNet), em duas fases:
  1. base congelada, treinando só a nova camada de saída;
  2. ajuste fino das últimas camadas da base, com taxa de aprendizado bem menor.

Uso (a partir da pasta raiz do projeto, depois de preparar_dados.py):
  python treino/treinar.py
  python treino/treinar.py --epocas 10 --epocas-fino 10

Gera a pasta modelos/vN/ com:
  modelo.keras     modelo completo (para retreinar ou inspecionar)
  modelo.tflite    modelo usado pela aplicação
  metadados.json   classes, tamanho de entrada e informações do treino
  historico.png    curvas de treino e validação
"""
import argparse
import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from comum import CLASSES, NOMES, SEED


def proxima_versao(pasta_modelos):
    numeros = [int(m.group(1)) for p in pasta_modelos.glob("v*") if (m := re.fullmatch(r"v(\d+)", p.name))]
    return f"v{max(numeros, default=0) + 1}"


def carregar(pasta, tamanho, lote, embaralhar):
    return keras.utils.image_dataset_from_directory(
        pasta, labels="inferred", label_mode="int", class_names=CLASSES,
        image_size=(tamanho, tamanho), batch_size=lote, shuffle=embaralhar, seed=SEED)


def criar_modelos(tamanho, pesos):
    """Retorna (nucleo, modelo_treino). Só o núcleo vai para produção, sem o aumento de dados."""
    base = keras.applications.MobileNetV2(
        input_shape=(tamanho, tamanho, 3), include_top=False, weights=pesos)
    base.trainable = False

    entrada = keras.Input(shape=(tamanho, tamanho, 3), name="imagem")
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="normalizacao")(entrada)  # pixels 0..255 -> -1..1
    x = base(x, training=False)  # mantém a BatchNorm da base em modo de inferência
    x = layers.GlobalAveragePooling2D(name="pooling_global")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    saida = layers.Dense(len(CLASSES), activation="softmax", name="probabilidades")(x)
    nucleo = keras.Model(entrada, saida, name="classificador")

    aumento = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ], name="aumento_de_dados")
    modelo_treino = keras.Sequential(
        [keras.Input(shape=(tamanho, tamanho, 3)), aumento, nucleo], name="modelo_treino")
    return base, nucleo, modelo_treino


def compilar(modelo, taxa):
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=taxa),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def exportar_tflite(nucleo, destino):
    """Exporta via SavedModel, com quantização de pesos (arquivo ~4x menor)."""
    with tempfile.TemporaryDirectory() as tmp:
        nucleo.export(tmp, verbose=False)
        conversor = tf.lite.TFLiteConverter.from_saved_model(tmp)
        conversor.optimizations = [tf.lite.Optimize.DEFAULT]
        destino.write_bytes(conversor.convert())


def salvar_curvas(historicos, inicio_fino, destino):
    h = {}
    for hist in historicos:
        for chave, valores in hist.history.items():
            h.setdefault(chave, []).extend(valores)
    epocas = range(1, len(h["loss"]) + 1)
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
    for eixo, metrica, titulo in [(eixos[0], "accuracy", "Acurácia"), (eixos[1], "loss", "Perda")]:
        eixo.plot(epocas, h[metrica], "o-", label="treino")
        eixo.plot(epocas, h["val_" + metrica], "o-", label="validação")
        eixo.axvline(inicio_fino + 0.5, ls="--", c="gray", label="início do ajuste fino")
        eixo.set_title(titulo)
        eixo.set_xlabel("época")
        eixo.grid(alpha=0.3)
        eixo.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    melhor = int(np.argmin(h["val_loss"]))
    return len(h["loss"]), melhor + 1, float(h["val_accuracy"][melhor])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dados", type=Path, default=Path("dados"))
    parser.add_argument("--modelos", type=Path, default=Path("modelos"))
    parser.add_argument("--versao", help="nome da versão (padrão: próxima vN livre)")
    parser.add_argument("--tamanho", type=int, default=160, help="lado da imagem de entrada (padrão 160)")
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--epocas", type=int, default=10, help="épocas com a base congelada")
    parser.add_argument("--epocas-fino", type=int, default=10, help="épocas de ajuste fino")
    parser.add_argument("--camadas-fino", type=int, default=30, help="camadas da base descongeladas no ajuste fino")
    parser.add_argument("--pesos", default="imagenet", help="'imagenet' ou 'nenhum' (só para testes)")
    args = parser.parse_args()

    keras.utils.set_random_seed(SEED)
    pesos = None if args.pesos == "nenhum" else args.pesos
    versao = args.versao or proxima_versao(args.modelos)
    pasta_saida = args.modelos / versao
    if pasta_saida.exists():
        raise SystemExit(f"A versão {versao} já existe em {pasta_saida}.")
    print(f"Treinando a versão {versao}")
    print("GPU:", tf.config.list_physical_devices("GPU") or "nenhuma (o treino em CPU é bem mais lento)")

    autotune = tf.data.AUTOTUNE
    treino = carregar(args.dados / "treino", args.tamanho, args.lote, True).prefetch(autotune)
    validacao = carregar(args.dados / "validacao", args.tamanho, args.lote, False).cache().prefetch(autotune)

    base, nucleo, modelo = criar_modelos(args.tamanho, pesos)
    pasta_saida.mkdir(parents=True)
    ponto = pasta_saida / "melhor_treino.keras"

    def callbacks():
        return [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(ponto, monitor="val_loss", save_best_only=True),
        ]

    try:
        print("\nFase 1: base congelada")
        compilar(modelo, 1e-3)
        hist1 = modelo.fit(treino, validation_data=validacao, epochs=args.epocas, callbacks=callbacks())

        print(f"\nFase 2: ajuste fino das últimas {args.camadas_fino} camadas da base")
        base.trainable = True
        for camada in base.layers[:-args.camadas_fino]:
            camada.trainable = False
        compilar(modelo, 1e-5)
        inicio_fino = len(hist1.history["loss"])
        hist2 = modelo.fit(treino, validation_data=validacao, epochs=inicio_fino + args.epocas_fino,
                           initial_epoch=inicio_fino, callbacks=callbacks())
    except BaseException:
        shutil.rmtree(pasta_saida, ignore_errors=True)  # não deixa uma versão pela metade
        raise
    ponto.unlink(missing_ok=True)

    print("\nSalvando e exportando...")
    nucleo.save(pasta_saida / "modelo.keras")
    exportar_tflite(nucleo, pasta_saida / "modelo.tflite")
    epocas, melhor, acc_val = salvar_curvas([hist1, hist2], inicio_fino, pasta_saida / "historico.png")

    resumo_dados = args.dados / "resumo.json"
    metadados = {
        "versao": versao,
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "classes": CLASSES,
        "nomes": NOMES,
        "tamanho_entrada": [args.tamanho, args.tamanho],
        "entrada": "RGB, pixels de 0 a 255 (a normalização está dentro do modelo)",
        "arquitetura": "MobileNetV2 (ImageNet) + GlobalAveragePooling + Dropout(0.3) + Dense softmax",
        "treino": {
            "epocas_executadas": epocas,
            "melhor_epoca": melhor,
            "acuracia_validacao": round(acc_val, 4),
            "camadas_ajuste_fino": args.camadas_fino,
            "pesos_iniciais": args.pesos,
        },
        "dados": json.loads(resumo_dados.read_text(encoding="utf-8")) if resumo_dados.exists() else None,
    }
    (pasta_saida / "metadados.json").write_text(json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8")

    tamanho_mb = (pasta_saida / "modelo.tflite").stat().st_size / 1e6
    print(f"\nPronto: {pasta_saida}/")
    print(f"  {epocas} épocas, melhor época {melhor}, acurácia de validação {acc_val:.3f}")
    print(f"  modelo.tflite: {tamanho_mb:.1f} MB")
    print(f"\nPróximo passo: python treino/avaliar.py --versao {versao}")


if __name__ == "__main__":
    main()
