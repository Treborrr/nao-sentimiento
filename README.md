# NAO v6 — escuchar una frase y clasificarla en un sentimiento

El robot escucha una frase por su micrófono, se transcribe en la Mac, un
clasificador entrenado con `sentiment_analysis_dataset.csv` decide el
sentimiento, y el NAO responde en voz alta con un gesto y un color de ojos.

```
        LA MAC  (un solo proceso Python 3)                EL NAO
  ┌──────────────────────────────────────┐        ┌──────────────────┐
  │ demo_nao.py                           │◀──qi──▶│  NAOqi            │
  │   micrófono del NAO ─(qi, por red)────┼────────┤  ALAudioDevice    │
  │   whisper.cpp  → texto                │        │                  │
  │   clasificador → sentimiento          │────────▶  ALTextToSpeech   │
  │   respuesta + gesto + color de ojos   │        │  ALMotion / ALLeds│
  └──────────────────────────────────────┘        └──────────────────┘
```

**Todo corre en la Mac.** Se conecta al robot con el módulo `qi` (LibQi) por la
red — **sin SSH, sin cuenta de Aldebaran, sin el SDK viejo de Python 2.7.**

## Por qué `qi` y no SSH ni el SDK de Python 2.7

El puerto de NAOqi (9559/9561) habla un protocolo binario propio. Para hablarlo
hay dos caminos: el módulo `qi`/`naoqi`, o entrar al robot por SSH. La forma
estándar —la que usan los demás— es `qi` instalado en la computadora, que se
conecta por la red igual que el `ALProxy(IP, port)` del script original
`scripts bsae/codlines_nao.py`.

`qi` no está en el PyPI normal, pero sí hay wheels de la comunidad (incluida
**macOS Apple Silicon**), así que en la Mac se instala y funciona:

```bash
pip install "qi" --extra-index-url https://funwithagents.github.io/libqi-python/simple/
```

El micrófono se captura **sin transferir archivos**: `ALAudioDevice` transmite
los bloques de audio del NAO a este proceso por la red, y se arma el WAV acá.

## Clonar y usar

```bash
git clone <URL-del-repo> nao
cd nao
./run.sh --fuente mic-local            # la 1a vez instala todo solo, después corre
```

La primera corrida de `./run.sh` (o `./setup.sh`) hace **todo automáticamente**:
crea el entorno de Python, instala las dependencias (`qi`, `pysentimiento`,
torch…) y descarga el modelo de voz de whisper (465 MB). Tarda unos minutos
**sólo la primera vez**; después arranca al instante.

Requisitos previos (lo único que no se automatiza):
- **macOS** con [Homebrew](https://brew.sh): `setup.sh` instala solo `ffmpeg` y
  `whisper-cpp` si faltan.
- **Python 3** (3.11–3.14).

Con el robot:
```bash
./run.sh --robot nyx --fuente robot --loop --voz robot   # ajustá la IP en config.json
```

> ¿Por qué no "clonar y nada más"? El modelo de voz pesa 465 MB (GitHub no
> acepta >100 MB) y el entorno de Python pesa >1 GB y es distinto en cada
> máquina. Nada de eso puede ir en el repo; por eso se baja/crea en el primer
> `run.sh`. Es **un comando**, no una lista de pasos.

### Dónde se pone la IP del robot

En **`config.json`**, en `nao.robots`. La IP la dice el NAO si le apretás el
botón del pecho:

```json
"robots": { "laboratorio": { "ip": "172.22.200.90", "port": 9561 } }
```

## Uso

```bash
# Con el robot: escucha por su micrófono, clasifica y responde
.venv/bin/python src/demo_nao.py --robot laboratorio

# Una frase tras otra hasta Ctrl+C
.venv/bin/python src/demo_nao.py --robot laboratorio --loop

# Sin robot, para probar el pipeline
.venv/bin/python src/demo_nao.py --fuente mic-local
.venv/bin/python src/demo_nao.py --fuente texto --texto "estoy harto de esperar"
```

Cuando los **ojos se ponen azules**, hablá: esa es la ventana de grabación
(`grabacion.segundos` en el config, 6 s por defecto).

### El robot no tiene voz en español

Este NAO sólo trae `Japanese / Chinese / English`. Por eso:

- La **consola** siempre muestra todo en español.
- `--voz robot` (por defecto con el robot): el NAO responde **en inglés**.
- `--voz mac`: el NAO hace ojos + gesto y la **Mac dice la frase en español**.
- `--voz ambos`: las dos cosas.

Para instalar español en el robot hace falta el admin; estas opciones lo evitan.

### Los ojos como indicador de estado

- **Conexión OK** → barrido arcoíris (chroma) al conectar.
- **Escuchando** → azul.
- **Respondiendo** → verde, y después el color del sentimiento.

Colores en `config.json` → `nao.colores`; el color por sentimiento, en
`GESTOS` de `src/nao_qi.py`. Hay una vista offline de la cara en
`vista_ojos.html` (doble clic, sin internet) para mostrar el comportamiento
sin el robot.

## Estado: qué está probado

| Tramo | Estado |
|---|---|
| Entrenamiento y métricas | probado |
| Clasificador (frase → sentimiento + confianza) | probado |
| Transcripción WAV → texto (whisper.cpp) | probado |
| Conexión `qi` a NAOqi desde la Mac | **probado contra el robot real** |
| Micrófono del NAO por red (ALAudioDevice) | **probado** (llega audio, pico ✓) |
| Voz + ojos + gesto del NAO | **probado contra el robot real** |

Sin robot, se verifica la cadena audio→texto→sentimiento con la voz del
sistema: `bash scripts/prueba_completa.sh`.

## El clasificador: pysentimiento

El robot clasifica con **pysentimiento** (modelo BERT preentrenado con millones
de tweets en español). Entiende la frase entera, no cuenta palabras, y da
emociones intuitivas: **alegria, enojo, tristeza, miedo, sorpresa, neutral**.

```
"estoy muy feliz"        → alegria  99%
"estoy harto de todo"    → enojo    87%
"tengo mucho miedo"      → miedo    96%
"no paro de llorar"      → tristeza 99%
```

Debajo del umbral de duda (`config.json` → `umbral_duda`, 0.40) el NAO dice
*"No estoy seguro…"* en vez de afirmar. Las respuestas tienen **varias frases
por emoción** (se elige al azar) para que no repita siempre la misma.

## Análisis del dataset (por qué NO se usa para clasificar)

El `sentiment_analysis_dataset.csv` se analizó a fondo; la conclusión fue que
**no sirve como motor del robot**, y pysentimiento es mejor. La historia, con
sus scripts, queda como la parte de procesamiento de datos:

**1. Dataset original → modelo TF-IDF propio** (`src/train_sentiment.py`):
73.9% en CV de 5 folds, 6 clases. Pero **inflado y confuso**: el dataset se
cosechó por palabra clave y usa la *Rueda de Emociones*, donde `joyful` =
juguetón/atrevido y **"feliz" cae en `peaceful`**. Partiéndolo
(`src/diagnostico.py`): 83% cuando la frase trae el adjetivo cosechado, 57%
cuando no.

**2. Reproceso + re-etiquetado → dataset v2** (`src/construir_dataset_v2.py`):
limpieza (2590 → 2175, sin vacíos ni duplicados) y **re-etiquetado con
pysentimiento** a emociones intuitivas. Los tweets con "feliz" pasaron de
dispersos (peaceful 10, powerful 8, joyful 2…) a **alegria 23**. Genera
`dataset_v2.csv` y `models/dataset_v2_informe.txt`.

**3. Modelo propio sobre v2** (`src/analisis_v2.py`): TF-IDF + Regresión
Logística entrenado con las etiquetas intuitivas = **63.5%** CV.

| enfoque | accuracy | nota |
|---|---|---|
| dataset original (Rueda) → TF-IDF | 73.9% | confuso (feliz=peaceful) |
| dataset v2 (intuitivo) → TF-IDF (alumno) | 63.5% | honesto, pero techo bajo |
| **pysentimiento (maestro, en el robot)** | **~90%+** | el que se usa |

El 63.5% < 73.9% **no es peor**: miden tareas distintas. El 73.9% premiaba
reconocer la palabra-sinónimo (fácil); el 63.5% es sobre emociones reales
(difícil, honesto). La lección: un modelo de bolsa-de-palabras no alcanza al
BERT — **por eso el robot usa pysentimiento**. Re-entrenar/afinar el BERT con
este dataset sería contraproducente (le re-enseñaría la taxonomía rara).

## Mapa de archivos

```
config.json                      IP/puerto, idioma, colores, umbral
requirements.txt
sentiment_analysis_dataset.csv   2590 tweets etiquetados (dataset original)
dataset_v2.csv                   dataset limpio y re-etiquetado (intuitivo)
scripts bsae/                    los scripts originales, sin tocar
scripts/descargar_modelo_voz.sh  baja el modelo de whisper (reanuda si se corta)
scripts/prueba_completa.sh       prueba audio→texto→sentimiento sin robot
vista_ojos.html                  vista offline de los ojos del NAO
models/                          métricas, informes y modelos
audio/                           WAVs de cada corrida
src/
  demo_nao.py             orquestador: escucha → transcribe → clasifica → responde
  nao_qi.py               control del NAO por `qi` (micrófono, voz, gesto, LEDs)
  classify.py             frase → emoción + confianza (pysentimiento) ← lo que usa el robot
  transcribe.py           WAV → texto (whisper.cpp)
  text_clean.py           normalización de texto
  audio_local.py          grabar del micrófono de la Mac (modo prueba)
  — análisis de datos (no lo usa el robot) —
  train_sentiment.py      modelo TF-IDF sobre el dataset original
  diagnostico.py          mide cuánto depende del léxico cosechado
  construir_dataset_v2.py limpia y re-etiqueta → dataset_v2.csv
  analisis_v2.py          entrena el modelo propio sobre v2 y compara
```

## Detalles que cuestan una tarde

- **`qi` en Python 3.14**: el PyPI normal no tiene wheel; usar el índice de
  `funwithagents`. Instalado: `qi-3.1.8` para `macosx_11_0_arm64`.
- **Captura por red**: la sesión qi debe escuchar en todas las interfaces
  (`--qi-listen-url=tcp://0.0.0.0:0`) para que el robot devuelva el audio, y hay
  que `registerService` ANTES de `subscribe`. Se piden los 4 micrófonos (0) y se
  toma el **frontal** (índice 0); pedir un solo canal por esa API daba silencio.
- **Voz del robot**: depende del paquete instalado. Si no tiene español
  (`idioma_robot`), responde en inglés o lo dice la Mac (`--voz mac`).
- **Gestos sin caídas**: sólo mueven cabeza y brazos (`Head/LArm/RArm`), nunca
  piernas ni postura; con fall manager activado no puede volcarse.
- **Re-etiquetar con pysentimiento** rápido: inferencia batcheada directa sobre
  el modelo en MPS (GPU de la Mac) → ~30 s para 2590 tweets, no ~2 h.
