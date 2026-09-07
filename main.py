"""
main.py
Reconocimiento de voz local en tiempo real de alta precisión con faster-whisper.

Flujo:
MICRÓFONO (Muestreo nativo) -> REMUESTREO 16kHz + VAD -> FASTER-WHISPER 'SMALL' -> TEXTO EN CONSOLA
"""

import sys
import os
from pathlib import Path
from audio_capture import AudioCapture
from speech_to_text import SpeechToText

# =====================================================================
# CONFIGURACIÓN DEL USUARIO
# =====================================================================

# ID del micrófono a usar (por ejemplo: 1 para 'Microphone Array (AMD Audio Device)')
# Si se deja en None, seleccionará automáticamente el predeterminado del sistema.
MIC_DEVICE_ID = 1

# Modelo Whisper de alta precisión
MODEL_SIZE = "small"

# Idioma forzado para evitar confusiones de pronunciación
INPUT_LANGUAGE = "es"

# Modo de diagnóstico: guarda los audios en WAV para verificar la calidad física
DEBUG_SAVE_AUDIO = True
DEBUG_DIR = Path(__file__).resolve().parent / "debug_audio"

# =====================================================================


def main():
    # Asegurar compatibilidad UTF-8 en consola de Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # 1. Listar micrófonos disponibles y resolver el seleccionado
    mics = AudioCapture.list_microphones()
    if not mics:
        print("\n[ERROR] No se detectó ningún micrófono de entrada disponible en el sistema.")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("Micrófonos de entrada disponibles:")
    for m in mics:
        marca = " (Predeterminado)" if m["is_default"] else ""
        print(f"  [ID {m['id']}] {m['name']}{marca} (Nativo: {int(m['default_samplerate'])} Hz)")

    try:
        selected_info = AudioCapture.resolve_microphone(MIC_DEVICE_ID)
    except Exception as e:
        print(f"\n[ERROR] No se pudo resolver el micrófono solicitado (ID {MIC_DEVICE_ID}): {e}")
        sys.exit(1)

    print("\nMicrófono seleccionado:")
    print(f"  [ID {selected_info['id']}] {selected_info['name']}")

    # 2. Inicializar el capturador de audio
    try:
        recorder = AudioCapture(
            device_index=selected_info["id"],
            target_sample_rate=16000,
            energy_threshold=0.012,
            pre_buffer_ms=300,
            silence_duration_ms=800,
            min_speech_duration_ms=500,
            max_speech_duration_ms=12000,
        )
    except Exception as e:
        print(f"\n[ERROR] Error al configurar el capturador de audio: {e}")
        sys.exit(1)

    # 3. Inicializar el motor faster-whisper (una sola vez)
    print(f"\nInicializando motor local Whisper ('{MODEL_SIZE}')...")
    print("(La primera vez descargará los pesos de 'small' a la carpeta local ./models/)")
    try:
        engine = SpeechToText(
            model_size=MODEL_SIZE,
            language=INPUT_LANGUAGE,
            beam_size=5,
        )
    except Exception as e:
        print(f"\n[ERROR] No se pudo inicializar Whisper: {e}")
        sys.exit(1)

    # 4. Mostrar panel informativo claro
    print("\n" + "=" * 65)
    print("   RECONOCIMIENTO DE VOZ EN TIEMPO REAL - MODO ALTA PRECISIÓN")
    print("=" * 65)
    print(f" - Modelo               : {MODEL_SIZE}")
    print(f" - Idioma               : español ('{INPUT_LANGUAGE}')")
    print(f" - Dispositivo          : {engine.device.upper()} ({engine.compute_type})")
    print(f" - Micrófono            : [ID {selected_info['id']}] {selected_info['name']}")
    print(f" - Sample rate Whisper  : 16000 Hz")
    print(f" - Sample rate micrófono: {recorder.native_sample_rate} Hz (Remuestreado con scipy)")
    print(f" - Guardado WAV debug   : {'Activado (./debug_audio/)' if DEBUG_SAVE_AUDIO else 'Desactivado'}")
    print("=" * 65)

    print("\nPRUEBA:")
    print("Di claramente:")
    print('  "Hola, esto es una prueba de reconocimiento de voz."\n')
    print("=" * 65)
    print("Escuchando continuamente... Presiona [Ctrl + C] para salir.\n")

    if DEBUG_SAVE_AUDIO:
        DEBUG_DIR.mkdir(exist_ok=True)

    recorder.start()
    capture_count = 0

    try:
        while True:
            # Esperar fragmento de voz detectado
            item = recorder.get_speech_segment(timeout=0.1)
            if item is None:
                continue

            audio_16k, duration_sec, rms_level = item
            capture_count += 1

            # Mostrar diagnóstico de audio capturado
            print(f"Duración capturada: {duration_sec:.2f} s")
            print(f"Nivel RMS: {rms_level:.5f}")

            # Guardar WAV para inspección manual
            if DEBUG_SAVE_AUDIO:
                wav_path = DEBUG_DIR / f"capture_{capture_count:03d}.wav"
                AudioCapture.save_wav(audio_16k, str(wav_path), sample_rate=16000)
                print(f"[Audio guardado en: debug_audio/{wav_path.name}]")

            # Transcribir con faster-whisper
            texto = engine.transcribe(audio_16k)
            if texto:
                print(f"Texto detectado: {texto}\n")
            else:
                print("Texto detectado: (No se distinguió texto claro en este fragmento)\n")

    except KeyboardInterrupt:
        print("\n\nDetención solicitada por el usuario...")

    finally:
        recorder.stop()
        print("Captura finalizada y recursos liberados correctamente.")


if __name__ == "__main__":
    main()
