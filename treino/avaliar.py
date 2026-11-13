"""Etapa 3 do treino: avalia o modelo exportado e sugere o limiar de confiança.

A avaliação usa o arquivo modelo.tflite e o mesmo código de previsão da aplicação
(app/classificador.py), então mede exatamente o que vai para produção.

  1. Calibra o limiar no conjunto de VALIDAÇÃO: o menor valor de confiança acima do qual
     o modelo acerta pelo menos --alvo (padrão 98%) das vezes. Abaixo dele, a aplicação
     consulta o Gemini.
  2. Mede o desempenho no conjunto de TESTE (fixo entre versões): acurácia, precisão,
     recall, F1, matriz de confusão e quantas imagens iriam para o Gemini.

Uso:
  python treino/avaliar.py --versao v1
  python treino/avaliar.py --versao v2 --comparar-com v1

Gera em modelos/vN/: metricas.json, matriz_confusao.png e erros_teste.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar o pacote app
from app.classificador import Classificador, preparar_imagem  # noqa: E402


def prever_conjunto(modelo, pasta):
    caminhos, y_true, probs = [], [], []
    for indice, classe in enumerate(modelo.classes):
        for caminho in sorted((pasta / classe).glob("*.jpg")):
            with Image.open(caminho) as img:
                probs.append(modelo.prever_array(preparar_imagem(img, modelo.tamanho)))
            caminhos.append(caminho)
            y_true.append(indice)
    return caminhos, np.array(y_true), np.array(probs)


def calibrar_limiar(y_true, probs, alvo):
    """Menor limiar com acurácia >= alvo entre as imagens de confiança acima dele."""
    confianca = probs.max(axis=1)
    acerto = probs.argmax(axis=1) == y_true
    tabela = []
    for limiar in np.round(np.arange(0.50, 1.00, 0.01), 2):
        cobertos = confianca >= limiar
        if cobertos.sum() == 0:
            continue
        tabela.append((float(limiar), float(acerto[cobertos].mean()), float(cobertos.mean())))
    for limiar, acc, cobertura in tabela:
        if acc >= alvo:
            return limiar, acc, cobertura, False
    limiar, acc, cobertura = tabela[-1] if tabela else (0.99, 0.0, 0.0)
    return limiar, acc, cobertura, True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versao", required=True)
    parser.add_argument("--dados", type=Path, default=Path("dados"))
    parser.add_argument("--modelos", type=Path, default=Path("modelos"))
    parser.add_argument("--alvo", type=float, default=0.98,
                        help="acurácia mínima exigida acima do limiar (padrão 0.98)")
    parser.add_argument("--comparar-com", help="outra versão já avaliada, para comparação")
    args = parser.parse_args()

    pasta = args.modelos / args.versao
    modelo = Classificador(pasta)
    nomes = [modelo.nomes[c] for c in modelo.classes]
    print(f"Avaliando {args.versao} (classes: {', '.join(nomes)})")

    # 1. Calibração do limiar na validação
    _, y_val, p_val = prever_conjunto(modelo, args.dados / "validacao")
    limiar, acc_lim, cob_val, nao_atingiu = calibrar_limiar(y_val, p_val, args.alvo)
    if nao_atingiu:
        print(f"  aviso: nenhum limiar atingiu {args.alvo:.0%} de acurácia na validação; usando {limiar}")

    # 2. Teste
    caminhos, y_teste, p_teste = prever_conjunto(modelo, args.dados / "teste")
    y_pred = p_teste.argmax(axis=1)
    relatorio = classification_report(y_teste, y_pred, labels=range(len(nomes)), target_names=nomes,
                                      digits=3, zero_division=0)
    rel_dict = classification_report(y_teste, y_pred, labels=range(len(nomes)), target_names=nomes,
                                     output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_teste, y_pred, labels=range(len(nomes)))

    conf_teste = p_teste.max(axis=1)
    cobertos = conf_teste >= limiar
    acc_cobertos = float((y_pred[cobertos] == y_teste[cobertos]).mean()) if cobertos.any() else None

    print("\nTeste (todas as imagens, só com o modelo):")
    print(relatorio)
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print("  " + " " * 17 + " | ".join(f"{n:>16}" for n in nomes))
    for nome, linha in zip(nomes, matriz):
        print(f"  {nome:<16} " + " | ".join(f"{v:>16}" for v in linha))
    print(f"\nLimiar sugerido (calibrado na validação): {limiar:.2f}")
    print(f"  no teste, {cobertos.mean():.1%} das imagens ficam com o modelo"
          + (f", com acurácia de {acc_cobertos:.1%}" if acc_cobertos is not None else ""))
    print(f"  e {1 - cobertos.mean():.1%} iriam para o Gemini")

    ConfusionMatrixDisplay(matriz, display_labels=nomes).plot(cmap="Blues", colorbar=False)
    plt.title(f"Matriz de confusão – {args.versao} (teste)")
    plt.xlabel("classe prevista")
    plt.ylabel("classe real")
    plt.tight_layout()
    plt.savefig(pasta / "matriz_confusao.png", dpi=120)
    plt.close()

    with open(pasta / "erros_teste.csv", "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["arquivo", "real", "previsto", "confianca"])
        erros = np.where(y_pred != y_teste)[0]
        for i in erros[np.argsort(-conf_teste[erros])]:
            escritor.writerow([caminhos[i].as_posix(), modelo.classes[y_teste[i]],
                               modelo.classes[y_pred[i]], f"{conf_teste[i]:.3f}"])

    metricas = {
        "versao": args.versao,
        "teste": {
            "imagens": int(len(y_teste)),
            "acuracia": round(float(rel_dict["accuracy"]), 4),
            "f1_macro": round(float(rel_dict["macro avg"]["f1-score"]), 4),
            "por_classe": {
                c: {m: round(float(rel_dict[n][k]), 4)
                    for m, k in [("precisao", "precision"), ("recall", "recall"), ("f1", "f1-score")]}
                for c, n in zip(modelo.classes, nomes)
            },
            "matriz_confusao": matriz.tolist(),
            "erros": int((y_pred != y_teste).sum()),
        },
        "limiar": {
            "sugerido": limiar,
            "alvo_acuracia": args.alvo,
            "acuracia_validacao_acima_do_limiar": round(acc_lim, 4),
            "cobertura_validacao": round(cob_val, 4),
            "cobertura_teste": round(float(cobertos.mean()), 4),
            "acuracia_teste_acima_do_limiar": round(acc_cobertos, 4) if acc_cobertos is not None else None,
        },
    }
    (pasta / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvo em {pasta}/: metricas.json, matriz_confusao.png, erros_teste.csv")

    if args.comparar_com:
        arquivo = args.modelos / args.comparar_com / "metricas.json"
        if not arquivo.exists():
            print(f"\n{args.comparar_com} ainda não foi avaliada.")
            return
        outra = json.loads(arquivo.read_text(encoding="utf-8"))
        print(f"\nComparação no mesmo conjunto de teste:")
        print(f"  {'':<26}{args.comparar_com:>10}{args.versao:>10}")
        linhas = [
            ("acurácia", outra["teste"]["acuracia"], metricas["teste"]["acuracia"]),
            ("F1 macro", outra["teste"]["f1_macro"], metricas["teste"]["f1_macro"]),
            ("fica com o modelo", outra["limiar"]["cobertura_teste"], metricas["limiar"]["cobertura_teste"]),
        ]
        for nome, a, b in linhas:
            print(f"  {nome:<26}{a:>10.3f}{b:>10.3f}")
        melhor = metricas["teste"]["f1_macro"] > outra["teste"]["f1_macro"]
        print("\n  " + (f"{args.versao} é melhor: pode ser promovida (MODELO_ATIVO={args.versao} no .env)."
                        if melhor else f"{args.versao} não superou {args.comparar_com}: mantenha a versão atual."))


if __name__ == "__main__":
    main()
