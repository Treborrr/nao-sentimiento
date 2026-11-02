# -*- coding: utf-8 -*-
"""Graba del microfono de esta maquina con ffmpeg (sin dependencias de Python).

Es el camino de prueba para cuando el NAO no esta encendido o a mano: deja un
WAV con el mismo formato que produce el robot (16 kHz, mono, PCM 16 bits), asi
que el resto del pipeline no nota la diferencia.

    .venv/bin/python src/audio_local.py --segundos 5
    .venv/bin/python src/audio_local.py --listar-dispositivos
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _entrada_ffmpeg(dispositivo):
    sistema = platform.system()
    if sistema == "Darwin":
        return ["-f", "avfoundation", "-i", dispositivo or ":0"]
    if sistema == "Windows":
        return ["-f", "dshow", "-i", dispositivo or "audio=Microphone"]
    return ["-f", "alsa", "-i", dispositivo or "default"]


def listar_dispositivos():
    """ffmpeg lista los dispositivos por stderr y sale con codigo != 0: es normal."""
    sistema = platform.system()
    if sistema == "Darwin":
        cmd = ["ffmpeg", "-hide_banner", "-f", "avfoundation",
               "-list_devices", "true", "-i", ""]
    elif sistema == "Windows":
        cmd = ["ffmpeg", "-hide_banner", "-f", "dshow",
               "-list_devices", "true", "-i", "dummy"]
    else:
        cmd = ["arecord", "-l"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stderr or proc.stdout)


def grabar(destino="audio/frase_local.wav", segundos=5, dispositivo=None):
    if not shutil.which("ffmpeg"):
        raise SystemExit("Falta ffmpeg.\nEn macOS:  brew install ffmpeg")

    if not os.path.isabs(destino):
        destino = os.path.join(RAIZ, destino)
    os.makedirs(os.path.dirname(destino), exist_ok=True)

    cmd = (["ffmpeg", "-y", "-loglevel", "error"]
           + _entrada_ffmpeg(dispositivo)
           + ["-t", str(segundos), "-ar", "16000", "-ac", "1",
              "-c:a", "pcm_s16le", destino])

    print("[mic local] grabando %s s..." % segundos, file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            "ffmpeg no pudo grabar:\n%s\n"
            "Revisa el dispositivo con --listar-dispositivos y los permisos de "
            "microfono de la terminal en Ajustes del sistema > Privacidad."
            % proc.stderr.strip()
        )
    print("[mic local] -> %s" % destino, file=sys.stderr)
    return destino


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default="audio/frase_local.wav")
    ap.add_argument("--segundos", type=int, default=5)
    ap.add_argument("--dispositivo", default=None)
    ap.add_argument("--listar-dispositivos", action="store_true")
    args = ap.parse_args()

    if args.listar_dispositivos:
        listar_dispositivos()
        return
    print(grabar(args.destino, args.segundos, args.dispositivo))


if __name__ == "__main__":
    main()
