# -*- coding: utf-8 -*-
"""El NAO escucha una frase y dice qué sentimiento oyó.  ---  PYTHON 3

    micrófono del NAO (por red, qi)  ->  whisper.cpp  ->  clasificador  ->  NAO responde

TODO corre en la Mac, en este único proceso Python 3. Se conecta al robot con
el módulo `qi` (LibQi) por la red: sin SSH, sin el SDK viejo de Python 2.7.
Ver src/nao_qi.py.

    # con robot
    .venv/bin/python src/demo_nao.py --robot laboratorio
    .venv/bin/python src/demo_nao.py --robot laboratorio --loop

    # sin robot, para probar el pipeline
    .venv/bin/python src/demo_nao.py --fuente mic-local
    .venv/bin/python src/demo_nao.py --fuente texto --texto "estoy harto de esperar"
"""
import argparse
import json
import os
import random
import subprocess
import sys

# Generador independiente de cualquier semilla global (transformers/torch fijan
# random.seed al importarse, lo que haría que el robot repitiera siempre la
# misma frase). SystemRandom usa entropía del sistema y no se ve afectado.
_azar = random.SystemRandom()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import clasificar          # noqa: E402
from transcribe import transcribir       # noqa: E402
import audio_local                       # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Emociones que devuelve el clasificador (pysentimiento, en español).
ETIQUETAS_ES = {
    "alegria": "alegre", "enojo": "enojado", "tristeza": "triste",
    "miedo": "con miedo", "sorpresa": "sorprendido", "neutral": "neutral",
}
# Gesto del NAO por emoción (reusa los gestos de src/nao_qi.py: GESTOS).
GESTO_POR_EMOCION = {
    "alegria": "joyful", "enojo": "mad", "tristeza": "sad",
    "miedo": "scared", "sorpresa": "powerful", "neutral": "peaceful",
}
# Varias respuestas por emoción: se elige una al azar para no repetir siempre.
RESPUESTAS_ES = {
    "alegria": [
        "Te escucho muy alegre. ¡Me contagiás esa energía!",
        "¡Qué bueno escucharte así de feliz!",
        "Se te siente contento, y me alegra un montón.",
        "¡Esa alegría es hermosa! Disfrutala.",
    ],
    "enojo": [
        "Te escucho enojado. Respirá, que ya pasa.",
        "Noto bronca en tu voz. Tranquilo, hablemos.",
        "Se te siente molesto. Está bien, dejalo salir.",
        "Percibo enojo. Contá hasta diez, estoy con vos.",
    ],
    "tristeza": [
        "Te escucho triste. Estoy acá si querés hablar.",
        "Se te siente apagado. No estás solo.",
        "Noto tristeza. Mandame un abrazo, va a mejorar.",
        "Percibo pena en tu voz. Acá estoy para escucharte.",
    ],
    "miedo": [
        "Te escucho con miedo. Tranquilo, vamos de a poco.",
        "Noto que algo te asusta. Respirá, estoy con vos.",
        "Se te siente temeroso. Paso a paso, sin apuro.",
        "Percibo miedo. Estás a salvo, vamos juntos.",
    ],
    "sorpresa": [
        "¡Te escucho sorprendido! ¿Qué pasó?",
        "¡Guau, eso suena inesperado!",
        "Noto sorpresa en tu voz. ¡Contame más!",
        "¡Qué sorpresa! Se te siente impactado.",
    ],
    "neutral": [
        "Te escucho tranquilo. Contame qué más.",
        "Noto un tono neutro. ¿Cómo te sentís?",
        "Se te siente sereno. Acá estoy.",
        "Entendido. ¿Querés contarme algo más?",
    ],
}
# Versión en inglés (por si el robot no tiene voz en español).
RESPUESTAS_EN = {
    "alegria": ["You sound really happy. That energy is contagious!"],
    "enojo":   ["You sound angry. Take a breath, it will pass."],
    "tristeza":["You sound sad. I am here if you want to talk."],
    "miedo":   ["You sound scared. It is okay, one step at a time."],
    "sorpresa":["You sound surprised! What happened?"],
    "neutral": ["You sound calm. Tell me more."],
}
UMBRAL_DUDA_POR_DEFECTO = 0.50


def cargar_config(ruta=None):
    ruta = ruta or os.path.join(RAIZ, "config.json")
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


def datos_robot(cfg, nombre=None):
    nombre = nombre or cfg["nao"]["robot"]
    robots = cfg["nao"]["robots"]
    if nombre not in robots:
        raise SystemExit("Robot '%s' desconocido. Definidos: %s"
                         % (nombre, ", ".join(robots)))
    r = robots[nombre]
    return nombre, r["ip"], int(r["port"])


def _decir_en_la_mac(frase, cfg):
    """Dice la frase en español con la voz del sistema (macOS `say`)."""
    voz = cfg.get("voz_local")
    if not voz and sys.platform == "darwin":
        disponibles = subprocess.run(["say", "-v", "?"], capture_output=True,
                                     text=True).stdout
        for candidata in ("Paulina", "Monica", "Mónica", "Jorge", "Juan"):
            if ("\n" + candidata + " ") in ("\n" + disponibles):
                voz = candidata
                break
    cmd = ["say"] + (["-v", voz] if voz else []) + [frase]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[aviso] no pude usar la voz del sistema: %s"
              % proc.stderr.strip(), file=sys.stderr)


def _mensajes(etiqueta, confianza, umbral):
    """Arma (frase_es, frase_en, gesto) según la confianza.

    El 'gesto' siempre tiene valor para que el robot se exprese DESPUÉS DE CADA
    MENSAJE: el del sentimiento si está seguro, "duda" si no, "no_entiendo" si
    no transcribió nada.
    """
    if etiqueta is None:
        return ("No te escuché bien. ¿Me lo repetís?",
                "I didn't catch that. Can you say it again?", "no_entiendo")
    if confianza < umbral:
        # Debajo del umbral el modelo no está seguro: que el robot lo diga en
        # vez de afirmar con aplomo, y acompañe con el gesto de duda.
        es = "No estoy seguro. Me suena %s, pero decímelo otra vez." % \
            ETIQUETAS_ES.get(etiqueta, etiqueta)
        en = "I'm not sure. Tell me again?"
        return es, en, "duda"
    es = _azar.choice(RESPUESTAS_ES.get(etiqueta, ["Te escucho %s." % etiqueta]))
    en = _azar.choice(RESPUESTAS_EN.get(etiqueta, ["You sound %s." % etiqueta]))
    return es, en, GESTO_POR_EMOCION.get(etiqueta, "peaceful")


def conseguir_texto(cfg, args, nao):
    """Devuelve la frase: del micrófono del NAO, del mic local, o literal."""
    if args.fuente == "texto":
        if not args.texto:
            raise SystemExit("--fuente texto necesita --texto \"la frase\"")
        return args.texto

    if args.fuente == "robot":
        wav = os.path.join(RAIZ, "audio", "frase_nao.wav")
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        print("  (escuchando %.0f s por el micrófono del NAO...)" % args.segundos)
        nao.grabar(args.segundos, wav)
    else:
        wav = audio_local.grabar("audio/frase_local.wav", args.segundos,
                                 cfg["grabacion"].get("dispositivo_local"))

    return transcribir(wav, modelo=cfg["whisper"]["modelo"],
                       binario=cfg["whisper"]["binario"],
                       idioma=cfg["whisper"]["idioma"])


def una_vuelta(cfg, args, nao):
    # 1-2. Frase (y transcripción si vino de audio)
    texto = conseguir_texto(cfg, args, nao)
    print("\n  Escuché      : %s" % (texto or "(nada)"))

    # 3. Clasificar
    resultado = clasificar(texto)
    etiqueta = resultado["etiqueta"]
    if etiqueta:
        print("  Sentimiento  : %s  (%.1f%% de confianza)"
              % (etiqueta, resultado["confianza"] * 100))
        detalle = "  ".join("%s %.0f%%" % (e, p * 100)
                            for e, p in resultado["ranking"][1:4])
        print("  Le siguen    : %s" % detalle)

    umbral = (args.umbral or cfg.get("umbral_duda")
              or resultado.get("umbral_duda") or UMBRAL_DUDA_POR_DEFECTO)
    frase_es, frase_en, etiqueta_gesto = _mensajes(
        etiqueta, resultado["confianza"], umbral)
    print("  NAO responde : %s" % frase_es)

    # 4. Responder: el robot con ojos + gesto + voz; o la Mac con voz en español.
    voz = args.voz
    if voz == "auto":
        voz = "robot" if nao is not None else "mac"

    if nao is not None:
        robot_habla = (voz in ("robot", "ambos"))
        # Si el robot tiene voz en español, que hable en español con la suya;
        # si no (como el robot viejo), en inglés. Lo decide el idioma efectivo
        # que quedó fijado al conectar (nao.idioma_efectivo).
        frase_robot = frase_es if nao.idioma_efectivo.startswith("Spanish") else frase_en
        nao.responder(frase_robot, sentimiento=etiqueta_gesto,
                      con_gesto=not args.sin_gesto, con_voz=robot_habla,
                      color_habla=_color(cfg, "habla", 0x00D070))
        # La Mac sólo hace falta si el robot NO habló en español.
        if voz in ("mac", "ambos") and not (robot_habla and
                                            nao.idioma_efectivo.startswith("Spanish")):
            _decir_en_la_mac(frase_es, cfg)
    else:
        _decir_en_la_mac(frase_es, cfg)

    return resultado


def _color(cfg, clave, por_defecto):
    valor = cfg["nao"].get("colores", {}).get(clave)
    return int(valor, 0) if valor else por_defecto


def conectar_robot(cfg, ip, port):
    """Crea la conexión qi y hace el chroma de bienvenida. None si falla."""
    import nao_qi
    nao = nao_qi.NaoQi(ip, port, idioma_robot=cfg["nao"].get("idioma_robot", "English"))
    if cfg["nao"].get("chroma_al_conectar", True):
        nao.chroma_conectado()
    return nao


def main():
    ap = argparse.ArgumentParser(description="NAO escucha una frase y la clasifica")
    ap.add_argument("--config", default=None)
    ap.add_argument("--robot", default=None, help="nombre del robot en config.json")
    ap.add_argument("--fuente", default="robot",
                    choices=["robot", "mic-local", "texto"],
                    help="de dónde sale la frase (por defecto: el micrófono del NAO)")
    ap.add_argument("--texto", default=None, help="frase literal con --fuente texto")
    ap.add_argument("--segundos", type=float, default=None)
    ap.add_argument("--sin-gesto", action="store_true")
    ap.add_argument("--voz", default="auto",
                    choices=["auto", "robot", "mac", "ambos"],
                    help="quién dice la respuesta: robot (inglés), mac (español), "
                         "ambos, o auto")
    ap.add_argument("--umbral", type=float, default=None,
                    help="pisa el umbral de duda calculado al entrenar")
    ap.add_argument("--loop", action="store_true",
                    help="escuchar una frase tras otra hasta Ctrl+C")
    args = ap.parse_args()

    cfg = cargar_config(args.config)
    if args.segundos is None:
        args.segundos = cfg["grabacion"]["segundos"]

    # Conectar al robot salvo que la frase venga del mic local o de texto Y la
    # voz no sea del robot (así --fuente texto --voz robot igual usa el NAO).
    nao = None
    necesita_robot = ((args.fuente == "robot") or (args.voz in ("robot", "ambos"))
                      or (args.robot is not None))
    if necesita_robot:
        nombre, ip, port = datos_robot(cfg, args.robot)
        print("Robot: %s  (%s:%d)" % (nombre, ip, port))
        nao = conectar_robot(cfg, ip, port)

    if not args.loop:
        una_vuelta(cfg, args, nao)
        return

    print("Modo continuo. Ctrl+C para cortar.")
    vuelta = 1
    try:
        while True:
            print("\n--- frase %d ---" % vuelta)
            una_vuelta(cfg, args, nao)
            vuelta += 1
    except KeyboardInterrupt:
        print("\nListo.")


if __name__ == "__main__":
    main()
