# -*- coding: utf-8 -*-
"""Clasificador de emoción con pysentimiento (BERT en español).  ---  PYTHON 3

A diferencia del TF-IDF + Regresión Logística anterior (bolsa de palabras,
atado a la taxonomía rara del dataset), esto usa un modelo preentrenado con
millones de tweets en español que entiende la frase entera. Da emociones
intuitivas: "estoy feliz" -> alegría, "tengo miedo" -> miedo.

Uso directo:
    .venv/bin/python src/classify.py "me siento muy feliz"

Devuelve el mismo formato que usaba el clasificador anterior, así que
demo_nao.py no nota el cambio: dict con etiqueta, confianza, ranking y objetivo.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_clean import clean_text  # noqa: E402

# pysentimiento (emoción, español) usa estas etiquetas; las traducimos a las
# nuestras. 'disgust' lo doblamos en enojo y 'others' en neutral.
MAPA_EMOCION = {
    "joy": "alegria",
    "anger": "enojo",
    "disgust": "enojo",
    "sadness": "tristeza",
    "fear": "miedo",
    "surprise": "sorpresa",
    "others": "neutral",
}

_ANALIZADOR = {}


def _analizador():
    """Carga el modelo una sola vez (tarda unos segundos la primera vez)."""
    if "a" not in _ANALIZADOR:
        from pysentimiento import create_analyzer
        _ANALIZADOR["a"] = create_analyzer(task="emotion", lang="es")
    return _ANALIZADOR["a"]


def clasificar(texto, modelo=None):
    """-> dict con etiqueta, confianza, ranking completo y texto normalizado.

    `modelo` se ignora (queda por compatibilidad con la llamada anterior).
    """
    limpio = clean_text(texto)
    if not limpio:
        return {"texto": texto, "texto_limpio": "", "etiqueta": None,
                "confianza": 0.0, "ranking": [], "objetivo": "emocion",
                "umbral_duda": None}

    salida = _analizador().predict(limpio)

    # Agrupamos las probabilidades por nuestra etiqueta (joy+... -> alegria, etc.)
    agrupado = {}
    for emo, prob in salida.probas.items():
        etiqueta = MAPA_EMOCION.get(emo, emo)
        agrupado[etiqueta] = agrupado.get(etiqueta, 0.0) + float(prob)

    ranking = sorted(agrupado.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "texto": texto,
        "texto_limpio": limpio,
        "etiqueta": ranking[0][0],
        "confianza": ranking[0][1],
        "ranking": ranking,
        "objetivo": "emocion",
        "umbral_duda": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texto", nargs="+")
    args = ap.parse_args()
    res = clasificar(" ".join(args.texto))
    print("Frase      : %s" % res["texto"])
    print("Prediccion : %s  (confianza %.1f%%)"
          % (res["etiqueta"], res["confianza"] * 100))
    print("Ranking    :")
    for etiqueta, p in res["ranking"]:
        print("   %-10s %5.1f%%" % (etiqueta, p * 100))


if __name__ == "__main__":
    main()
