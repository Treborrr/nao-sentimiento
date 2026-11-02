#!/usr/bin/env bash
# Lanza la demo. Si el proyecto todavía no está instalado, lo instala solo la
# primera vez (crea .venv, instala dependencias, baja el modelo de voz) y
# después corre. Todos los argumentos se pasan a demo_nao.py.
#
#   ./run.sh --fuente mic-local
#   ./run.sh --robot nyx --fuente robot --loop --voz robot
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ¿Falta instalar? (sin entorno o sin modelo de voz) -> setup automático.
if [ ! -x .venv/bin/python ] || [ ! -s models/ggml-small.bin ]; then
  echo "(primera vez: instalando, esto tarda unos minutos...)"
  ./setup.sh
fi

exec ./.venv/bin/python src/demo_nao.py "$@"
