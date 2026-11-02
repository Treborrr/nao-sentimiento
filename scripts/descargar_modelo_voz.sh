#!/usr/bin/env bash
# Descarga un modelo de whisper.cpp a models/.
#
#   bash scripts/descargar_modelo_voz.sh small    # 465 MB  (recomendado en espanol)
#   bash scripts/descargar_modelo_voz.sh base     # 142 MB  (mas rapido, peor)
#   bash scripts/descargar_modelo_voz.sh medium   # 1.5 GB  (mejor, mas lento)
#
# Son modelos multilingues: NO usar las variantes ".en", que son solo ingles.
set -euo pipefail

MODELO="${1:-small}"
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINO="$RAIZ/models/ggml-${MODELO}.bin"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODELO}.bin"

# Tamano esperado en MB, igual que la tabla de src/transcribe.py.
case "$MODELO" in
  tiny)   ESPERADO=74   ;;
  base)   ESPERADO=141  ;;
  small)  ESPERADO=465  ;;
  medium) ESPERADO=1456 ;;
  large*) ESPERADO=2950 ;;
  *)      ESPERADO=0    ;;
esac

mkdir -p "$RAIZ/models"

# Ojo: NO alcanza con preguntar si el archivo existe. Una descarga cortada deja
# un .bin incompleto, y con eso whisper.cpp no da un error: revienta con un
# volcado nativo. Si esta corto hay que reanudar, no darlo por bueno.
if [ -s "$DESTINO" ]; then
  MB=$(( $(wc -c < "$DESTINO") / 1048576 ))
  if [ "$ESPERADO" -eq 0 ] || [ "$MB" -ge $(( ESPERADO * 92 / 100 )) ]; then
    echo "Ya esta completo: $DESTINO (${MB} MB)"
    exit 0
  fi
  echo "Descarga previa incompleta (${MB} MB de ${ESPERADO} MB). Reanudando..."
fi

echo "Descargando ggml-${MODELO}.bin ..."
# -C - reanuda desde donde quedo. --retry reintenta si la red se corta sola,
# que es exactamente lo que pasa con este archivo en una conexion lenta.
curl -L --fail -C - --retry 10 --retry-delay 5 --retry-all-errors \
     -o "$DESTINO" "$URL"

MB=$(( $(wc -c < "$DESTINO") / 1048576 ))
if [ "$ESPERADO" -ne 0 ] && [ "$MB" -lt $(( ESPERADO * 92 / 100 )) ]; then
  echo "AVISO: quedo en ${MB} MB de ${ESPERADO} MB. Volvé a correr este script."
  exit 1
fi

echo "Listo: $DESTINO (${MB} MB)"
echo
echo "Si no es el modelo 'small', apuntá config.json a este archivo:"
echo "    \"whisper\": { \"modelo\": \"models/ggml-${MODELO}.bin\" }"
