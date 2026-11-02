# -*- coding: utf-8 -*-
"""Entrena el clasificador de sentimiento sobre sentiment_analysis_dataset.csv.

    .venv/bin/python src/train_sentiment.py                # sentimiento (6 clases)
    .venv/bin/python src/train_sentiment.py --target emotion  # emocion (20 clases)

Deja en models/:
    sentiment_model.joblib   pipeline listo para inferencia
    sentiment_metrics.txt    metricas de la validacion cruzada
"""
import argparse
import csv
import os
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_union
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_clean import clean_text  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_POR_DEFECTO = os.path.join(RAIZ, "sentiment_analysis_dataset.csv")
DIR_MODELOS = os.path.join(RAIZ, "models")


def cargar(ruta_csv, columna_objetivo):
    textos, etiquetas = [], []
    with open(ruta_csv, encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            limpio = clean_text(fila.get("text", ""))
            etiqueta = (fila.get(columna_objetivo) or "").strip()
            if limpio and etiqueta:
                textos.append(limpio)
                etiquetas.append(etiqueta)
    return textos, np.array(etiquetas)


def construir_vectorizador():
    """Palabras (1-2 gramas) + caracteres (3-5) unidos.

    Los n-gramas de caracteres son los que sostienen el castellano informal:
    absorben faltas de ortografia, formas flexionadas ("abrumado"/"abrumada")
    y emojis sin necesidad de lematizar.
    """
    return make_union(
        TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, strip_accents=None,
        ),
        TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3,
            sublinear_tf=True,
        ),
    )


def candidatos():
    return {
        "logreg": Pipeline([
            ("tfidf", construir_vectorizador()),
            ("clf", LogisticRegression(
                C=10.0, max_iter=3000, class_weight="balanced",
                solver="lbfgs",
            )),
        ]),
        "linearsvc": Pipeline([
            ("tfidf", construir_vectorizador()),
            ("clf", LinearSVC(C=1.0, class_weight="balanced")),
        ]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_POR_DEFECTO)
    ap.add_argument("--target", default="sentiment", choices=["sentiment", "emotion"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tolerancia", type=float, default=0.03,
                    help="accuracy que estoy dispuesto a ceder para tener "
                         "probabilidades calibradas (ver mas abajo)")
    ap.add_argument("--precision-objetivo", type=float, default=0.80,
                    help="precision que debe tener el robot cuando decide "
                         "responder, para calcular el umbral de duda")
    args = ap.parse_args()

    textos, etiquetas = cargar(args.csv, args.target)
    print("Filas utilizables: %d | clases: %d" % (len(textos), len(set(etiquetas))))

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    resultados = {}
    for nombre, pipe in candidatos().items():
        pred = cross_val_predict(pipe, textos, etiquetas, cv=cv, n_jobs=-1)
        acc = accuracy_score(etiquetas, pred)
        resultados[nombre] = (acc, pred)
        print("  %-10s accuracy CV = %.4f" % (nombre, acc))

    mejor = max(resultados, key=lambda k: resultados[k][0])

    # Para el robot preferimos un modelo con probabilidades reales aunque pierda
    # un poco de accuracy: LinearSVC solo da margenes, y un softmax sobre
    # margenes queda tan plano que el NAO no puede distinguir "estoy seguro" de
    # "estoy adivinando". Esa distincion importa mas que 2 puntos de accuracy,
    # porque es la que decide si responde o pide que le repitan la frase.
    if not hasattr(candidatos()[mejor], "predict_proba"):
        probabilisticos = [n for n in resultados
                           if hasattr(candidatos()[n], "predict_proba")]
        if probabilisticos:
            alterno = max(probabilisticos, key=lambda k: resultados[k][0])
            brecha = resultados[mejor][0] - resultados[alterno][0]
            if brecha <= args.tolerancia:
                print("\n%s gana por %.4f, pero no da probabilidades calibradas."
                      % (mejor, brecha))
                print("Me quedo con %s (dentro de la tolerancia de %.2f)."
                      % (alterno, args.tolerancia))
                mejor = alterno

    acc, pred = resultados[mejor]
    print("\nElegido: %s (%.4f)\n" % (mejor, acc))

    informe = classification_report(etiquetas, pred, digits=3, zero_division=0)
    clases = sorted(set(etiquetas))
    mat = confusion_matrix(etiquetas, pred, labels=clases)
    print(informe)

    # Umbral de duda: por encima de que confianza conviene que el robot afirme.
    umbral, cobertura, precision = 0.0, 1.0, acc
    if hasattr(candidatos()[mejor], "predict_proba"):
        probas = cross_val_predict(candidatos()[mejor], textos, etiquetas,
                                   cv=cv, n_jobs=-1, method="predict_proba")
        clases_cv = np.array(sorted(set(etiquetas)))
        confianza = probas.max(axis=1)
        predicho = clases_cv[probas.argmax(axis=1)]
        acierto = (predicho == etiquetas)

        print("confianza   cobertura   precision")
        for u in np.arange(0.20, 0.96, 0.05):
            masc = confianza >= u
            if masc.sum() < 30:
                break
            print("  >= %.2f      %5.1f%%      %5.1f%%"
                  % (u, 100 * masc.mean(), 100 * acierto[masc].mean()))
            if acierto[masc].mean() >= args.precision_objetivo and umbral == 0.0:
                umbral, cobertura, precision = u, masc.mean(), acierto[masc].mean()
        if umbral:
            print("\nUmbral de duda sugerido: %.2f "
                  "(responde en %.0f%% de los casos, acierta %.0f%% de esas veces)"
                  % (umbral, 100 * cobertura, 100 * precision))
        else:
            print("\nNinguna confianza alcanza la precision objetivo de %.0f%%."
                  % (100 * args.precision_objetivo))
        print("")

    # El modelo final se entrena con TODO el dataset; la metrica honesta es la de CV.
    final = candidatos()[mejor]
    final.fit(textos, etiquetas)

    os.makedirs(DIR_MODELOS, exist_ok=True)
    ruta_modelo = os.path.join(DIR_MODELOS, "%s_model.joblib" % args.target)
    joblib.dump({
        "pipeline": final,
        "target": args.target,
        "classes": list(final.classes_),
        "cv_accuracy": acc,
        "algoritmo": mejor,
        "n_train": len(textos),
        "umbral_duda": umbral,
        "umbral_cobertura": cobertura,
        "umbral_precision": precision,
    }, ruta_modelo)

    ruta_metricas = os.path.join(DIR_MODELOS, "%s_metrics.txt" % args.target)
    with open(ruta_metricas, "w", encoding="utf-8") as fh:
        fh.write("objetivo: %s\nalgoritmo: %s\nfilas: %d\n" % (args.target, mejor, len(textos)))
        fh.write("accuracy CV (%d folds): %.4f\n" % (args.folds, acc))
        if umbral:
            fh.write("umbral de duda: %.2f (cobertura %.1f%%, precision %.1f%%)\n"
                     % (umbral, 100 * cobertura, 100 * precision))
        fh.write("\n")
        fh.write(informe)
        fh.write("\n\nMatriz de confusion (filas = real, columnas = predicho)\n")
        fh.write("%-12s%s\n" % ("", "".join("%12s" % c[:11] for c in clases)))
        for nombre_clase, fila in zip(clases, mat):
            fh.write("%-12s%s\n" % (nombre_clase[:11], "".join("%12d" % v for v in fila)))

    print("Modelo  -> %s" % ruta_modelo)
    print("Metricas-> %s" % ruta_metricas)


if __name__ == "__main__":
    main()
