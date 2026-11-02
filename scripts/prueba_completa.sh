#!/usr/bin/env bash
# Prueba la cadena audio -> texto -> sentimiento SIN robot y SIN microfono.
#
# Sintetiza frases en espanol con la voz del sistema (macOS `say`), las pasa por
# whisper.cpp y despues por el clasificador. Sirve para confirmar que la parte
# de la computadora funciona antes de ir al laboratorio a probar con el NAO.
#
#   bash scripts/prueba_completa.sh
#
# Ojo: la voz sintetica se transcribe mas facil que una persona hablando en una
# sala con ruido. Que esto pase no garantiza que el audio del NAO transcriba
# igual de bien; es una prueba de que el circuito esta conectado, no de calidad.
set -uo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"

PY="$RAIZ/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

# Primera voz en espanol que exista en el sistema.
VOZ=""
for candidata in "Paulina" "Monica" "Mónica" "Jorge" "Juan" "Diego"; do
  if say -v '?' 2>/dev/null | grep -q "^${candidata} "; then VOZ="$candidata"; break; fi
done
if [ -z "$VOZ" ]; then
  VOZ="$(say -v '?' 2>/dev/null | grep -iE 'es_(ES|MX|AR)' | head -1 | sed 's/ *(.*//')"
fi
if [ -z "$VOZ" ]; then
  echo "No encontre ninguna voz en espanol en el sistema."
  echo "Instalala en Ajustes del sistema > Accesibilidad > Contenido leido > Voz."
  exit 1
fi
echo "Voz del sistema: $VOZ"
echo

FRASES=(
  "estoy harto de esperar, me tienen podrido"
  "que lindo dia, estoy muy feliz"
  "me siento abrumado por todo lo que tengo que entregar"
  "lo logre, me siento imparable"
)

mkdir -p audio
FALLAS=0

for frase in "${FRASES[@]}"; do
  echo "--------------------------------------------------------------"
  echo "  Dicho    : $frase"

  say -v "$VOZ" -o audio/_prueba.aiff "$frase"
  ffmpeg -y -loglevel error -i audio/_prueba.aiff \
         -ar 16000 -ac 1 -c:a pcm_s16le audio/_prueba.wav

  SALIDA="$("$PY" src/demo_nao.py --fuente texto \
            --texto "$("$PY" src/transcribe.py audio/_prueba.wav \
                       | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["texto"])')" 2>&1)"

  if [ -z "$SALIDA" ]; then
    echo "  FALLO: no hubo salida"
    FALLAS=$((FALLAS + 1))
  else
    echo "$SALIDA" | grep -E "Escuché|Sentimiento|NAO responde" | sed 's/^/  /'
  fi
  echo
done

rm -f audio/_prueba.aiff audio/_prueba.wav audio/_whisper_in.wav

echo "--------------------------------------------------------------"
if [ "$FALLAS" -eq 0 ]; then
  echo "La cadena audio -> texto -> sentimiento funciona."
  echo "Falta probar con el robot: .venv/bin/python src/demo_nao.py --robot nyx"
else
  echo "$FALLAS de ${#FRASES[@]} frases fallaron."
  exit 1
fi
