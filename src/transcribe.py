# -*- coding: utf-8 -*-
"""WAV -> texto con whisper.cpp (binario whisper-cli, todo local, sin red).

    .venv/bin/python src/transcribe.py audio/frase.wav
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RE_MARCA = re.compile(r"^\s*\[[^\]]+\]\s*")   # "[00:00:00.000 --> 00:00:03.000]  texto"
RE_RUIDO = re.compile(r"[\(\[](?:musica|m[uú]sica|risas|aplausos|silencio)[\)\]]", re.I)

# Tamano esperado de cada modelo, en MB. Si la descarga quedo a medias,
# whisper-cli no da un error: revienta con un volcado nativo ilegible. Mas vale
# detectarlo aca y decir que hay que reanudar la descarga.
TAMANOS_MB = {
    "tiny": 74, "base": 141, "small": 465, "medium": 1456, "large": 2950,
}
TOLERANCIA = 0.92   # margen por variaciones entre versiones del archivo


def _absoluta(ruta):
    return ruta if os.path.isabs(ruta) else os.path.join(RAIZ, ruta)


def _verificar_modelo(modelo):
    """Corta temprano si el .bin quedo a medio descargar."""
    nombre = os.path.basename(modelo)
    mb = os.path.getsize(modelo) / (1024.0 * 1024.0)

    esperado = next((v for k, v in TAMANOS_MB.items()
                     if nombre.startswith("ggml-%s" % k)), None)
    if esperado is None:
        if mb < 30:
            raise SystemExit(
                "El modelo %s pesa solo %.0f MB: parece incompleto.\n"
                "Volvé a descargarlo:  bash scripts/descargar_modelo_voz.sh small"
                % (nombre, mb)
            )
        return

    if mb < esperado * TOLERANCIA:
        etiqueta = nombre.replace("ggml-", "").replace(".bin", "")
        raise SystemExit(
            "El modelo %s esta incompleto: %.0f MB de los %d MB esperados.\n"
            "La descarga se corto. Reanudala (el script sigue desde donde quedo):\n"
            "    bash scripts/descargar_modelo_voz.sh %s"
            % (nombre, mb, esperado, etiqueta)
        )


def transcribir(ruta_wav, modelo="models/ggml-small.bin", binario="whisper-cli",
                idioma="es", hilos=None):
    """Devuelve el texto dictado. Lanza SystemExit con un mensaje util si falta algo."""
    ruta_wav = _absoluta(ruta_wav)
    modelo = _absoluta(modelo)

    if not shutil.which(binario):
        raise SystemExit(
            "No encuentro '%s' en el PATH.\n"
            "En macOS:  brew install whisper-cpp" % binario
        )
    if not os.path.exists(modelo):
        raise SystemExit(
            "Falta el modelo de voz: %s\n"
            "Descargalo con:  bash scripts/descargar_modelo_voz.sh small" % modelo
        )
    _verificar_modelo(modelo)
    if not os.path.exists(ruta_wav):
        raise SystemExit("No existe el audio: %s" % ruta_wav)

    # whisper.cpp exige PCM 16 bits mono a 16 kHz. El NAO ya graba asi, pero el
    # microfono local puede no hacerlo: normalizamos siempre con ffmpeg.
    wav_normalizado = os.path.join(os.path.dirname(ruta_wav), "_whisper_in.wav")
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", ruta_wav,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_normalizado],
            check=True,
        )
        entrada = wav_normalizado
    else:
        entrada = ruta_wav

    cmd = [binario, "-m", modelo, "-l", idioma, "-nt", "-f", entrada]
    if hilos:
        cmd += ["-t", str(hilos)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("whisper-cli fallo:\n%s" % proc.stderr.strip())

    lineas = []
    for linea in proc.stdout.splitlines():
        linea = RE_MARCA.sub("", linea).strip()
        linea = RE_RUIDO.sub("", linea).strip()
        if linea:
            lineas.append(linea)
    return " ".join(lineas).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--modelo", default="models/ggml-small.bin")
    ap.add_argument("--idioma", default="es")
    args = ap.parse_args()
    texto = transcribir(args.wav, modelo=args.modelo, idioma=args.idioma)
    print(json.dumps({"texto": texto}, ensure_ascii=False))


if __name__ == "__main__":
    main()
