# -*- coding: utf-8 -*-
"""Entrena NUESTRO modelo con el dataset v2 y lo compara.  ---  PYTHON 3

Cierra la historia de análisis de datos:
  - Dataset original (Rueda de Emociones) -> modelo TF-IDF = 73.9%, confuso.
  - Lo reprocesamos y re-etiquetamos (construir_dataset_v2.py) -> dataset v2.
  - Acá entrenamos un TF-IDF + Regresión Logística con el v2 y medimos.

Es "destilar" pysentimiento (el maestro) en un modelo chico y propio (el alumno)
que SÍ usa el dataset. El robot sigue corriendo con pysentimiento (opción B).

    .venv/bin/python src/analisis_v2.py
"""
import csv
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sentiment import construir_vectorizador   # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_V2 = os.path.join(RAIZ, "dataset_v2.csv")
MODELO = os.path.join(RAIZ, "models", "emocion_v2_model.joblib")
METRICAS = os.path.join(RAIZ, "models", "emocion_v2_metrics.txt")
CONF_MIN = 0.5   # entrenamos con las re-etiquetas en que el maestro está seguro


def main():
    textos, etiquetas = [], []
    with open(CSV_V2, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            if float(f["confianza"]) >= CONF_MIN:
                textos.append(f["text"])
                etiquetas.append(f["emocion_v2"])
    etiquetas = np.array(etiquetas)
    print("Filas de entrenamiento (confianza >= %.1f): %d | clases: %d"
          % (CONF_MIN, len(textos), len(set(etiquetas))))

    pipe = Pipeline([
        ("tfidf", construir_vectorizador()),
        ("clf", LogisticRegression(C=10.0, max_iter=3000,
                                   class_weight="balanced", solver="lbfgs")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pred = cross_val_predict(pipe, textos, etiquetas, cv=cv, n_jobs=-1)
    acc = accuracy_score(etiquetas, pred)

    mayor = max(set(etiquetas), key=list(etiquetas).count)
    base = (etiquetas == mayor).mean()

    informe = classification_report(etiquetas, pred, digits=3, zero_division=0)
    clases = sorted(set(etiquetas))
    mat = confusion_matrix(etiquetas, pred, labels=clases)

    lineas = []
    def em(s):
        print(s); lineas.append(s)

    em("Modelo propio entrenado con dataset v2 (TF-IDF + Regresión Logística)")
    em("=" * 60)
    em("filas: %d   clases: %d" % (len(textos), len(clases)))
    em("línea base (siempre '%s'): %.3f" % (mayor, base))
    em("accuracy CV (5 folds)     : %.3f" % acc)
    em("")
    em("Comparación de enfoques:")
    em("  dataset ORIGINAL (Rueda) -> TF-IDF           : 0.739  (confuso: feliz=peaceful)")
    em("  dataset V2 (re-etiquetado) -> TF-IDF (alumno): %.3f  (intuitivo)" % acc)
    em("  pysentimiento (maestro, en el robot)         : ~0.90+ en tweets es")
    em("")
    em(informe)
    em("Matriz de confusión (fila=real, col=predicho)")
    em("%-11s%s" % ("", "".join("%10s" % c[:9] for c in clases)))
    for nombre, fila in zip(clases, mat):
        em("%-11s%s" % (nombre[:10], "".join("%10d" % v for v in fila)))

    joblib.dump({"pipeline": pipe.fit(textos, etiquetas),
                 "clases": clases, "cv_accuracy": acc}, MODELO)
    with open(METRICAS, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print("\nModelo  -> %s" % MODELO)
    print("Métricas-> %s" % METRICAS)


if __name__ == "__main__":
    main()
