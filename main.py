"""
main.py
Punto de entrada principal: escucha continua del micrófono y transcripción local con faster-whisper.

Flujo:
MICRÓFONO -> AUDIO -> FASTER-WHISPER -> TEXTO EN CONSOLA
"""

import sys
import time
from audio_capture import AudioCapture
from speech_to_text import SpeechToText

# ID del micrófono a usar (None para usar el predeterminado de Windows)
# Para cambiarlo, consulta la lista que se muestra al iniciar y coloca aquí el número de ID deseado.
MIC_DEVICE_ID = None

# Tamaño del modelo Whisper: "tiny", "base", "small", "medium"
# "base" es rápido y liviano (~145MB). "small" ofrece mayor precisión (~460MB).
MODEL_SIZE = "base"

# Idioma de entrada: "es" (español), "en" (inglés), o None (autodetección continua)
INPUT_LANGUAGE = "es"


def main():
    # Asegurar compatibilidad de salida UTF-8 en terminales Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65)
    print("   RECONOCIMIENTO DE VOZ EN TIEMPO REAL (100% LOCAL)")
    print("=" * 65)

    # 1. Listar y verificar micrófonos disponibles
    microphones = AudioCapture.list_microphones()
    if not microphones:
        print("\n[ERROR] No se detectó ningún micrófono de entrada en el sistema.")
        print("Por favor, conecta o habilita un micrófono y vuelve a ejecutar el programa.")
        sys.exit(1)

    print("\nMicrófonos de entrada disponibles:")
    for mic in microphones:
        marca = " (Predeterminado)" if mic["is_default"] else ""
        print(f"  [ID {mic['id']}] {mic['name']}{marca}")

    dispositivo_seleccionado = "Predeterminado de Windows" if MIC_DEVICE_ID is None else f"ID {MIC_DEVICE_ID}"
    print(f"\nDispositivo seleccionado: {dispositivo_seleccionado}")

    # 2. Inicializar el motor de transcripción faster-whisper
    print(f"\nCargando modelo local Whisper ('{MODEL_SIZE}')...")
    try:
        engine = SpeechToText(model_size=MODEL_SIZE, language=INPUT_LANGUAGE)
    except Exception as e:
        print(f"\n[ERROR] No se pudo inicializar el modelo Whisper: {e}")
        sys.exit(1)

    print(f"[OK] Modelo listo. Ejecutando en: [{engine.device.upper()} - {engine.compute_type}]")

    # 3. Inicializar el capturador de audio con detección de voz
    try:
        recorder = AudioCapture(device_index=MIC_DEVICE_ID)
        recorder.start()
    except RuntimeError as e:
        print(f"\n[ERROR] Dispositivo de audio incorrecto o no disponible:\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado al iniciar la captura de audio: {e}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("Escuchando micrófono continuamente...")
    print("Habla libremente. Cuando termines una frase, se transcribirá.")
    print("Presiona [Ctrl + C] para detener el programa en cualquier momento.")
    print("=" * 65 + "\n")

    try:
        while True:
            # Esperar el próximo fragmento de voz detectado
            speech_audio = recorder.get_speech_segment(timeout=0.1)
            if speech_audio is None:
                continue

            # Transcribir el audio usando faster-whisper
            try:
                texto = engine.transcribe(speech_audio)
                if texto:
                    print(f"Texto detectado: {texto}")
            except Exception as e:
                print(f"[AVISO] Ocurrió un error transcribiendo el fragmento: {e}")

    except KeyboardInterrupt:
        print("\n\nDetención solicitada por el usuario...")

    finally:
        recorder.stop()
        print("Captura de audio finalizada y recursos liberados correctamente.")


if __name__ == "__main__":
    main()
