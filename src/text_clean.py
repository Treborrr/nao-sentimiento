# -*- coding: utf-8 -*-
"""Normalizacion de texto compartida entre el entrenamiento y la inferencia.

Se aplica SIEMPRE antes de vectorizar. Si cambia esto, hay que reentrenar
(el modelo guardado espera texto ya normalizado).
"""
import re
import unicodedata

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_RISA = re.compile(r"\b(?:[ja]{4,}|[je]{4,}|x+d+)\b", re.IGNORECASE)
RE_REPETIDA = re.compile(r"(.)\1{2,}")
RE_ESPACIOS = re.compile(r"\s+")


def quitar_acentos(texto):
    """Quita tildes y dieresis, pero NO toca los emojis.

    Descompone cada caracter y descarta solo las marcas combinantes (categoria
    Mn), que es donde viven las tildes. Los emojis no tienen marcas, asi que
    pasan intactos.

    Hace falta porque el texto del dataset viene acentuado y el de whisper
    tambien, pero cualquier frase escrita a mano sin tildes caia en tokens
    distintos: "día" y "dia" eran palabras diferentes para el modelo, y la
    confianza se movia 12 puntos por una tilde. Medido sobre el dataset,
    unificarlas no cuesta accuracy (0.739 vs 0.737).
    """
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto
                   if unicodedata.category(c) != "Mn")


def clean_text(texto):
    """Deja el tweet (o la frase dictada al NAO) en forma canonica.

    Conserva los emojis a proposito: son de las senales mas fuertes que hay
    para el sentimiento, y los n-gramas de caracteres los aprovechan.
    """
    if not texto:
        return ""
    t = unicodedata.normalize("NFC", texto)
    t = t.lower()
    t = quitar_acentos(t)
    t = RE_URL.sub(" ", t)
    t = RE_MENCION.sub(" ", t)          # el autor no aporta al sentimiento
    t = RE_HASHTAG.sub(r" \1 ", t)      # #NadieComoTu -> nadiecomotu
    t = RE_RISA.sub(" risa ", t)        # jajaja / jejeje / xdddd -> un solo token
    t = RE_REPETIDA.sub(r"\1\1", t)     # "holaaaaa" -> "holaa"
    t = RE_ESPACIOS.sub(" ", t)
    return t.strip()
