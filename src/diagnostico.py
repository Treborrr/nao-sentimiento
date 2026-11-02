# -*- coding: utf-8 -*-
"""Mide hasta donde llega de verdad el clasificador.

El dataset se recolecto buscando sinonimos: cada clase es un conjunto de
adjetivos ("abrumado", "desbordado", "sobrecargado" -> scared). Entonces la
accuracy global premia el reconocer esos adjetivos, que es la parte facil.

Este script parte el dataset en dos y mide cada mitad por separado:

  CON lexico : la frase trae alguno de los adjetivos fuertes de su clase
  SIN lexico : la frase expresa el sentimiento sin ninguno de esos adjetivos

La segunda cifra es la que predice como se va a portar el robot con una frase
dicha al aire, y es bastante mas baja. Hay que citar las dos.

    .venv/bin/python src/diagnostico.py
"""
import argparse
import csv
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_clean import clean_text          # noqa: E402
from train_sentiment import candidatos     # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lexico_por_clase(textos, etiquetas, n_palabras=40):
    """Palabras con mas peso positivo por clase: el 'lexico cosechado'."""
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=2)
    matriz = vec.fit_transform(textos)
    nombres = np.array(vec.get_feature_names_out())
    clf = LinearSVC(C=1.0, class_weight="balanced").fit(matriz, etiquetas)

    lexico = {}
    for i, clase in enumerate(clf.classes_):
        orden = np.argsort(clf.coef_[i])[::-1][:n_palabras]
        lexico[clase] = set(nombres[orden])
    return lexico


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(RAIZ, "sentiment_analysis_dataset.csv"))
    ap.add_argument("--target", default="sentiment", choices=["sentiment", "emotion"])
    ap.add_argument("--palabras", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    textos, etiquetas = [], []
    with open(args.csv, encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            limpio = clean_text(fila.get("text", ""))
            etiqueta = (fila.get(args.target) or "").strip()
            if limpio and etiqueta:
                textos.append(limpio)
                etiquetas.append(etiqueta)
    etiquetas = np.array(etiquetas)

    lexico = lexico_por_clase(textos, etiquetas, args.palabras)
    tiene_lexico = np.array([
        any(p in t.split() for p in lexico[e]) for t, e in zip(textos, etiquetas)
    ])

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    lineas = []

    def emitir(texto):
        print(texto)
        lineas.append(texto)

    emitir("objetivo: %s   filas: %d   clases: %d"
           % (args.target, len(textos), len(set(etiquetas))))
    emitir("lexico cosechado: %d palabras por clase" % args.palabras)
    emitir("")
    emitir("Filas CON alguno de esos adjetivos: %d (%.1f%%)"
           % (tiene_lexico.sum(), 100 * tiene_lexico.mean()))
    emitir("Filas SIN ninguno de ellos      : %d (%.1f%%)"
           % ((~tiene_lexico).sum(), 100 * (~tiene_lexico).mean()))
    emitir("")

    mayoritaria = max(set(etiquetas), key=list(etiquetas).count)
    base = (etiquetas == mayoritaria).mean()
    emitir("Linea base (siempre '%s'): %.3f" % (mayoritaria, base))
    emitir("")

    for nombre, pipe in candidatos().items():
        pred = cross_val_predict(pipe, textos, etiquetas, cv=cv, n_jobs=-1)
        total = accuracy_score(etiquetas, pred)
        con = accuracy_score(etiquetas[tiene_lexico], pred[tiene_lexico])
        sin = accuracy_score(etiquetas[~tiene_lexico], pred[~tiene_lexico])
        emitir("%-10s  global %.3f   CON lexico %.3f   SIN lexico %.3f"
               % (nombre, total, con, sin))

    emitir("")
    emitir("Lectura: la columna 'SIN lexico' es la que hay que mirar para")
    emitir("estimar que va a pasar cuando alguien le hable al robot sin usar")
    emitir("justo uno de los adjetivos que el dataset fue a buscar.")

    destino = os.path.join(RAIZ, "models", "%s_diagnostico.txt" % args.target)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print("\n-> %s" % destino)


if __name__ == "__main__":
    main()
