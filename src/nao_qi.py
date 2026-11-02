# -*- coding: utf-8 -*-
"""Control del NAO desde la Mac con el modulo `qi` (LibQi).  ---  PYTHON 3

Esta es la forma en que se maneja el robot SIN SSH y SIN el SDK viejo de
Python 2.7: `qi` habla el protocolo de NAOqi por la red. Se instala con:

    pip install "qi" --extra-index-url https://funwithagents.github.io/libqi-python/simple/

Reemplaza por completo a los scripts de src/nao/*.py (que iban por SSH): todo
corre en el mismo proceso Python 3 que transcribe y clasifica.

Un robot de catedra suele tener la voz en el idioma de fabrica (este tiene
japones/chino/ingles, NO espanol), asi que las respuestas al robot van en
ingles; la consola y, si se quiere, la voz de la Mac, siguen en espanol.
"""
import colorsys
import struct
import threading
import time
import wave

GRUPO_OJOS = "FaceLeds"

# sentimiento -> color de ojos (0xRRGGBB) y gesto corto.
#
# SEGURIDAD: todos los gestos mueven SOLO cabeza y brazos (hombros, codos).
# Nunca tocan las piernas ni cambian la postura de base, así que el robot NO
# puede perder el equilibrio ni caerse por un gesto. Los ángulos están holgados
# dentro de los límites del NAO v6 para que ALMotion no los recorte.
GESTOS = {
    "joyful":   (0xFFD500, [
        (["LShoulderPitch", "RShoulderPitch"], [-0.6, -0.6], [[0.8], [0.8]]),
        (["LShoulderRoll", "RShoulderRoll"], [0.9, -0.9], [[0.8], [0.8]]),
        (["HeadPitch"], [-0.25], [[0.6]]),
    ]),
    "mad":      (0xFF0000, [
        (["LShoulderPitch", "RShoulderPitch"], [1.2, 1.2], [[0.5], [0.5]]),
        (["LElbowRoll", "RElbowRoll"], [-1.4, 1.4], [[0.4], [0.4]]),
        (["HeadPitch"], [0.35], [[0.4]]),
    ]),
    "peaceful": (0x00C86E, [
        (["LShoulderRoll", "RShoulderRoll"], [0.8, -0.8], [[1.6], [1.6]]),
        (["HeadPitch"], [-0.1], [[1.4]]),
    ]),
    "powerful": (0xFF7A00, [
        (["RShoulderPitch"], [-1.0], [[0.6]]),
        (["RElbowRoll"], [1.3], [[0.4]]),
        (["HeadPitch"], [-0.3], [[0.5]]),
    ]),
    "sad":      (0x1E3CFF, [
        (["HeadPitch"], [0.45], [[1.2]]),
        (["LShoulderPitch", "RShoulderPitch"], [1.45, 1.45], [[1.4], [1.4]]),
        (["LShoulderRoll", "RShoulderRoll"], [0.35, -0.35], [[1.4], [1.4]]),
    ]),
    "scared":   (0xC800FF, [
        (["LElbowRoll", "RElbowRoll"], [-1.5, 1.5], [[0.4], [0.4]]),
        (["LShoulderPitch", "RShoulderPitch"], [0.9, 0.9], [[0.4], [0.4]]),
        (["HeadYaw"], [-0.5], [[0.4]]),
    ]),
    # Duda: encogida de hombros + cabeza ladeada (cuando no está seguro).
    "duda":     (0xFFA500, [
        (["LShoulderRoll", "RShoulderRoll"], [0.5, -0.5], [[0.6], [0.6]]),
        (["LElbowRoll", "RElbowRoll"], [-1.0, 1.0], [[0.6], [0.6]]),
        (["HeadPitch"], [0.15], [[0.5]]),
    ]),
    # No entendió: negación suave con la cabeza (izquierda-derecha).
    "no_entiendo": (0x8899AA, [
        (["HeadYaw"], [-0.3], [[0.5]]),
        (["HeadYaw"], [0.3], [[0.6]]),
        (["HeadYaw"], [0.0], [[0.5]]),
    ]),
}
COLOR_NEUTRO = 0xFFFFFF

# Pose de reposo de los brazos (tren superior), para volver después del gesto.
# También son articulaciones de brazos/cabeza: no afecta el equilibrio.
BRAZOS_REPOSO = (
    ["LShoulderPitch", "RShoulderPitch", "LShoulderRoll", "RShoulderRoll",
     "LElbowRoll", "RElbowRoll", "HeadPitch", "HeadYaw"],
    [1.45, 1.45, 0.15, -0.15, -0.5, 0.5, 0.0, 0.0],
    [[1.0]] * 8,
)
# Articulaciones que se rigidizan para gesticular. NO incluye piernas a
# propósito: así el robot mantiene su equilibrio pase lo que pase.
MIEMBROS_GESTO = ["Head", "LArm", "RArm"]


def _hsv(h):
    r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
    return (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)


class _ReceptorAudio(object):
    """ALAudioDevice entrega los bloques del microfono a processRemote."""
    def __init__(self):
        self.frames = []
        self.n_canales = 4
    def processRemote(self, nCanales, nMuestras, timestamp, buffer):
        self.n_canales = nCanales
        self.frames.append(bytes(buffer))


class NaoQi(object):
    def __init__(self, ip, port, idioma_robot="English"):
        import qi  # import tardio: solo hace falta si se usa el robot
        self.ip = ip
        self.port = port
        self.idioma_robot = idioma_robot
        try:
            # --qi-listen-url: la sesión escucha en TODAS las interfaces
            # (no sólo 127.0.0.1), para que el robot pueda conectarse DE VUELTA
            # y entregarnos el audio del micrófono (ALAudioDevice). Sin esto,
            # el streaming falla con "Can't find service".
            self.app = qi.Application([
                "nao_qi",
                "--qi-url=tcp://%s:%d" % (ip, port),
                "--qi-listen-url=tcp://0.0.0.0:0",
            ])
            self.app.start()
            self.ses = self.app.session
        except Exception as e:
            raise SystemExit(
                "No pude conectar a NAOqi en tcp://%s:%d\n%s\n\n"
                "Revisá que el robot esté encendido y en la misma red "
                "(ping %s), y la IP/puerto en config.json." % (ip, port, e, ip)
            )
        self.tts = self.ses.service("ALTextToSpeech")
        self.leds = self.ses.service("ALLeds")
        self.audio = self.ses.service("ALAudioDevice")
        self._fijar_idioma()

    def _fijar_idioma(self):
        """Fija el idioma pedido si está; si no, avisa y usa el que haya.

        Deja en self.idioma_efectivo el idioma realmente activo, para que el
        orquestador sepa si el robot puede hablar español o hay que recurrir a
        la voz de la Mac.
        """
        self.idioma_efectivo = self.idioma_robot
        try:
            disponibles = self.tts.getAvailableLanguages()
            if self.idioma_robot in disponibles:
                self.tts.setLanguage(self.idioma_robot)
            else:
                print("[nao] el robot no tiene la voz '%s'; disponibles: %s"
                      % (self.idioma_robot, ", ".join(disponibles)))
                self.idioma_efectivo = self.tts.getLanguage()
        except Exception as e:
            print("[nao] no pude fijar el idioma: %s" % e)

    # ---- LEDs -------------------------------------------------------------
    def fijar_ojos(self, color, seg=0.3):
        try:
            self.leds.fadeRGB(GRUPO_OJOS, int(color), float(seg))
        except Exception as e:
            print("[nao] LEDs: %s" % e)

    def chroma_conectado(self, vueltas=2, seg_por_vuelta=1.2, pasos=18):
        """Barrido arcoiris en los ojos: senal de 'conectado'."""
        try:
            espera = float(seg_por_vuelta) / pasos
            for _ in range(int(vueltas)):
                for i in range(pasos):
                    self.leds.fadeRGB(GRUPO_OJOS, _hsv(float(i) / pasos), espera)
            self.fijar_ojos(COLOR_NEUTRO, 0.3)
        except Exception as e:
            print("[nao] chroma: %s" % e)

    # ---- Microfono por streaming -----------------------------------------
    def grabar(self, segundos, destino_wav):
        """Graba desde el microfono del NAO por la red y guarda un WAV local.

        No usa archivos en el robot ni scp: ALAudioDevice transmite los bloques
        de audio a este proceso y los escribimos acá.
        """
        receptor = _ReceptorAudio()
        idserv = self.ses.registerService("ReceptorNao", receptor)
        # El servicio recién registrado tarda un instante en propagarse al
        # directorio de NAOqi; sin esta pausa, ALAudioDevice no lo encuentra
        # ("Can't find service: ReceptorNao").
        time.sleep(0.6)
        self.fijar_ojos(0x1E90FF, 0.2)   # ojos azules = escuchando
        try:
            # Pedimos los 4 micrófonos (0) intercalados y después nos quedamos
            # con el frontal. Pedir un solo canal por esta API daba silencio;
            # medido en el robot, el frontal es el que capta la voz.
            self.audio.setClientPreferences("ReceptorNao", 16000, 0, 0)
            self.audio.subscribe("ReceptorNao")
            time.sleep(segundos)
            self.audio.unsubscribe("ReceptorNao")
        finally:
            try:
                self.ses.unregisterService(idserv)
            except Exception:
                pass
        self.fijar_ojos(COLOR_NEUTRO, 0.2)

        datos = b"".join(receptor.frames)
        if not datos:
            raise SystemExit(
                "El robot no envió audio. ¿El micrófono está activo? "
                "(ALAudioDevice sin datos)"
            )
        # Llegan los 4 micrófonos intercalados; nos quedamos con el frontal
        # (índice 0), que es el que capta la voz.
        nC = receptor.n_canales or 4
        todas = struct.unpack("<%dh" % (len(datos) // 2), datos)
        frente = todas[0::nC]
        datos = struct.pack("<%dh" % len(frente), *frente)

        # Nivel de audio: ayuda a saber si el robot te escuchó o agarró silencio.
        muestras = frente
        pico = max(abs(x) for x in muestras) if muestras else 0
        if pico < 500:
            aviso = "muy bajo: hablá más fuerte / más cerca"
        elif pico < 2000:
            aviso = "bajo"
        else:
            aviso = "ok"
        print("  (nivel de audio: %d — %s)" % (pico, aviso))

        with wave.open(destino_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(datos)
        return destino_wav

    # ---- Hablar + gesto ---------------------------------------------------
    def responder(self, texto, sentimiento=None, con_gesto=True, con_voz=True,
                  color_habla=0x00D070):
        """El robot reacciona: ojos + (opcional) gesto + (opcional) voz.

        con_voz=False sirve cuando la frase la dice la Mac en español y el robot
        sólo acompaña con ojos y gesto (el robot no tiene voz en español).
        """
        color_sent, movimientos = GESTOS.get(sentimiento, (COLOR_NEUTRO, []))
        self.fijar_ojos(color_habla, 0.3)
        self.fijar_ojos(color_sent, 0.4)   # ojos = emoción (o duda/neutro)

        # Hablar en un hilo para que hable MIENTRAS gesticula.
        hilo = None
        if con_voz:
            hilo = threading.Thread(target=self._decir_bloqueante, args=(texto,))
            hilo.start()

        if con_gesto and movimientos:
            self._gesto_seguro(movimientos)

        if hilo is not None:
            hilo.join()
        self.fijar_ojos(COLOR_NEUTRO, 0.5)

    def _gesto_seguro(self, movimientos):
        """Ejecuta un gesto SIN riesgo de caída.

        Sólo rigidiza y mueve cabeza y brazos (MIEMBROS_GESTO). No toca las
        piernas ni la postura de base, así que el equilibrio del robot no se
        altera: con el robot de pie o sentado, un gesto no lo puede voltear.
        Deja activo el gestor de caídas por las dudas.
        """
        motion = self.ses.service("ALMotion")
        try:
            motion.setFallManagerEnabled(True)
        except Exception:
            pass
        try:
            # Rigidez SOLO en tren superior; las piernas quedan como estaban.
            motion.setStiffnesses(MIEMBROS_GESTO, 1.0)
        except Exception as e:
            print("[nao] no pude rigidizar brazos/cabeza (omito gesto): %s" % e)
            return
        try:
            for articulaciones, angulos, tiempos in movimientos:
                motion.angleInterpolation(articulaciones, angulos, tiempos, True)
            # Volver los brazos a reposo, suave (también tren superior).
            motion.angleInterpolation(
                BRAZOS_REPOSO[0], BRAZOS_REPOSO[1], BRAZOS_REPOSO[2], True)
        except Exception as e:
            print("[nao] gesto: %s" % e)

    def _decir_bloqueante(self, texto):
        try:
            self.tts.say(texto)
        except Exception as e:
            print("[nao] TTS: %s" % e)
