# -*- coding: utf-8 -*-
"""Reprocesa el dataset y lo re-etiqueta de forma intuitiva.  ---  PYTHON 3

El dataset original tiene dos problemas: se cosechó por palabra clave y usa la
"Rueda de Emociones" (feliz -> peaceful), que choca con la intuición. Acá
construimos una versión v2 MEJOR:

  1. Limpieza: normaliza el texto, quita vacíos y duplicados.
  2. Re-etiquetado: pasa cada tweet por pysentimiento (BERT español) y le pone
     la emoción INTUITIVA (alegria/enojo/tristeza/miedo/sorpresa/neutral), con
     su confianza.
  3. Guarda dataset_v2.csv y un informe de limpieza.

    .venv/bin/python src/construir_dataset_v2.py

Esto tarda unos minutos (el modelo procesa los 2590 tweets). El robot NO usa
esto: corre con pysentimiento en vivo. Esto es la parte de análisis de datos.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_clean import clean_text          # noqa: E402
from classify import MAPA_EMOCION          # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_ORIG = os.path.join(RAIZ, "sentiment_analysis_dataset.csv")
CSV_V2 = os.path.join(RAIZ, "dataset_v2.csv")
INFORME = os.path.join(RAIZ, "models", "dataset_v2_informe.txt")


def main():
    filas = list(csv.DictReader(open(CSV_ORIG, encoding="utf-8")))
    n0 = len(filas)

    # 1. Limpieza: normalizar, quitar vacíos y duplicados (por texto limpio).
    vistos = set()
    limpias = []
    for f in filas:
        t = clean_text(f.get("text", ""))
        if not t or t in vistos:
            continue
        vistos.add(t)
        limpias.append({
            "text": f["text"].replace("\n", " ").strip(),
            "texto_limpio": t,
            "sentiment_orig": (f.get("sentiment") or "").strip(),
            "emotion_orig": (f.get("emotion") or "").strip(),
        })
    print("Filas: %d -> %d (quitados %d vacíos/duplicados)"
          % (n0, len(limpias), n0 - len(limpias)))

    # 2. Re-etiquetar con pysentimiento. Inferencia batcheada directa sobre el
    #    modelo (en la GPU de la Mac si hay MPS): ~30 s en vez de ~2 h.
    print("Re-etiquetando con pysentimiento...")
    import torch
    from pysentimiento import create_analyzer
    analizador = create_analyzer(task="emotion", lang="es")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    modelo = analizador.model.to(dev).eval()
    tok = analizador.tokenizer
    id2label = modelo.config.id2label   # {0:'others',1:'joy',...}

    textos = [f["texto_limpio"] for f in limpias]
    B = 128
    with torch.no_grad():
        for ini in range(0, len(textos), B):
            enc = tok(textos[ini:ini + B], padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(dev)
            probs = torch.softmax(modelo(**enc).logits, dim=-1).cpu()
            for j, f in enumerate(limpias[ini:ini + B]):
                # Agrupar por nuestra etiqueta (disgust->enojo, others->neutral)
                agr = {}
                for idx, p in enumerate(probs[j].tolist()):
                    k = MAPA_EMOCION.get(id2label[idx], id2label[idx])
                    agr[k] = agr.get(k, 0.0) + p
                etiqueta = max(agr, key=agr.get)
                f["emocion_v2"] = etiqueta
                f["confianza"] = round(agr[etiqueta], 4)
            print("  %d/%d" % (min(ini + B, len(textos)), len(textos)), end="\r")
    print()

    # 3. Guardar dataset v2.
    campos = ["text", "emocion_v2", "confianza", "sentiment_orig", "emotion_orig"]
    with open(CSV_V2, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for f in limpias:
            w.writerow(f)
    print("Dataset v2 -> %s" % CSV_V2)

    # 4. Informe: distribución nueva y acuerdo con la taxonomía vieja.
    import collections
    dist = collections.Counter(f["emocion_v2"] for f in limpias)
    conf_alta = sum(1 for f in limpias if f["confianza"] >= 0.5)

    lineas = []
    def em(s):
        print(s); lineas.append(s)

    em("Dataset v2 — informe")
    em("=" * 40)
    em("filas originales : %d" % n0)
    em("filas v2 (limpias): %d" % len(limpias))
    em("con confianza >=0.5: %d (%.0f%%)" % (conf_alta, 100*conf_alta/len(limpias)))
    em("")
    em("Distribución de emociones v2 (intuitivas):")
    for e, c in dist.most_common():
        em("  %-10s %4d (%.0f%%)" % (e, c, 100*c/len(limpias)))
    em("")
    em("Ejemplo del cambio de taxonomía — tweets con 'feliz':")
    fel = [f for f in limpias if "feliz" in f["texto_limpio"]]
    vieja = collections.Counter(f["sentiment_orig"] for f in fel)
    nueva = collections.Counter(f["emocion_v2"] for f in fel)
    em("  etiqueta vieja (Rueda): %s" % dict(vieja))
    em("  etiqueta v2 (intuitiva): %s" % dict(nueva))

    os.makedirs(os.path.dirname(INFORME), exist_ok=True)
    with open(INFORME, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print("\nInforme -> %s" % INFORME)


if __name__ == "__main__":
    main()
