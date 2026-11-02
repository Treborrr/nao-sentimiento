#!/usr/bin/env bash
# Deja el proyecto listo para usar: entorno de Python, dependencias y modelo de
# voz. Se corre UNA vez después de clonar. Es idempotente: si ya está, no repite.
#
#   ./setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "==> NAO sentimiento · instalación"

# --- 1. Dependencias del sistema (ffmpeg, whisper-cpp) --------------------
faltan=""
command -v ffmpeg      >/dev/null 2>&1 || faltan="$faltan ffmpeg"
command -v whisper-cli >/dev/null 2>&1 || faltan="$faltan whisper-cpp"
if [ -n "$faltan" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "==> Instalando con Homebrew:$faltan"
    brew install $faltan
  else
    echo "FALTA instalar:$faltan"
    echo "  macOS : instalá Homebrew (https://brew.sh) y luego: brew install$faltan"
    echo "  Linux : sudo apt install ffmpeg  (whisper.cpp: compilalo o usá el paquete de tu distro)"
    exit 1
  fi
fi

# --- 2. Entorno de Python + dependencias ----------------------------------
if [ ! -d .venv ]; then
  echo "==> Creando entorno .venv"
  python3 -m venv .venv
fi
echo "==> Instalando dependencias de Python (qi, pysentimiento, torch...)"
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install -r requirements.txt

# --- 3. Modelo de voz de whisper (465 MB) ---------------------------------
if [ ! -s models/ggml-small.bin ] || \
   [ "$(wc -c < models/ggml-small.bin)" -lt 400000000 ]; then
  echo "==> Descargando el modelo de voz (465 MB, una sola vez)"
  bash scripts/descargar_modelo_voz.sh small
else
  echo "==> Modelo de voz ya presente"
fi

# --- 4. Verificación rápida ------------------------------------------------
echo "==> Verificando..."
./.venv/bin/python src/demo_nao.py --fuente texto --texto "estoy feliz" --voz mac \
  >/dev/null 2>&1 && echo "   clasificador OK" || echo "   (aviso: revisá la instalación)"

echo
echo "Listo. Para usarlo:"
echo "  ./run.sh --fuente mic-local                 # probar con el micro de tu compu"
echo "  ./run.sh --robot nyx --fuente robot --loop  # con el NAO (ajustá la IP en config.json)"
