"""
main.py
Traductor y reconocedor de voz en tiempo real de baja latencia.

Flujo optimizado:
MICRÓFONO -> VAD RÁPIDO (corte a 400ms) -> FASTER-WHISPER (~0.3s) -> TRADUCTOR LOCAL (~0.1s) -> CONSOLA
"""

import sys
import time
from pathlib import Path
from colorama import init, Fore, Style

from audio_capture import AudioCapture
from speech_to_text import SpeechToText
from translator import LocalTranslator

init(autoreset=True)

# =====================================================================
# CONFIGURACIÓN PRINCIPAL
# =====================================================================

# Micrófono a usar:
# ID 1 suele ser 'Microphone Array (AMD Audio Device)'
# Pon None para usar el predeterminado de Windows
MIC_DEVICE_ID = 1

# Idioma en el que habla la persona / el video:
# "en" = inglés (recomendado si estás viendo videos en inglés)
# "es" = español
# None = detección automática
INPUT_LANGUAGE = "en"

# ¿Traducir automáticamente al español al terminar la frase?
# True = muestra el texto original en inglés y su traducción inmediata en español
# False = solo muestra el texto transcripto
ENABLE_TRANSLATION = True

# Modelo Whisper:
# "small" = excelente precisión (~460MB)
# "base"  = velocidad extrema sub-segundo (~145MB)
MODEL_SIZE = "small"

# Guardar archivos WAV para diagnóstico en debug_audio/
DEBUG_SAVE_AUDIO = False
DEBUG_DIR = Path(__file__).resolve().parent / "debug_audio"

# =====================================================================


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(Fore.CYAN + "=" * 65)
    print(Fore.CYAN + Style.BRIGHT + "   TRADUCTOR EN TIEMPO REAL - MODO ULTRA BAJA LATENCIA")
    print(Fore.CYAN + "=" * 65)

    # 1. Resolver micrófono
    mics = AudioCapture.list_microphones()
    if not mics:
        print(Fore.RED + "\n[ERROR] No se detectó ningún micrófono de entrada disponible.")
        sys.exit(1)

    try:
        selected_info = AudioCapture.resolve_microphone(MIC_DEVICE_ID)
    except Exception as e:
        print(Fore.RED + f"\n[ERROR] ID de micrófono no válido: {e}")
        sys.exit(1)

    # 2. Inicializar capturador de audio con corte rápido (400ms silencio)
    recorder = AudioCapture(
        device_index=selected_info["id"],
        target_sample_rate=16000,
        energy_threshold=0.012,
        pre_buffer_ms=250,
        silence_duration_ms=400,       # Corta a los 400ms de terminar la frase
        min_speech_duration_ms=400,    # Descarta clics o ruidos menores a 0.4s
        max_speech_duration_ms=4000,   # Corta cada 4s si el habla es continua
    )

    # Capturar antes de cargar modelos; conservar las primeras frases en memoria.
    recorder.start()
    try:
        print(Fore.WHITE + f"\nCapturando audio. Cargando Whisper ('{MODEL_SIZE}'); el inicio queda en memoria...")
        engine = SpeechToText(model_size=MODEL_SIZE, default_language=INPUT_LANGUAGE, beam_size=1)
        translator = None
        if ENABLE_TRANSLATION and INPUT_LANGUAGE == "en":
            print(Fore.WHITE + "Cargando traductor local EN -> ES...")
            translator = LocalTranslator(device="cpu")
    except BaseException:
        recorder.stop()
        raise

    # 5. Panel informativo
    print("\n" + Fore.CYAN + "=" * 65)
    print(Fore.WHITE + f" - Micrófono seleccionado : {Fore.YELLOW}[ID {selected_info['id']}] {selected_info['name']}")
    print(Fore.WHITE + f" - Idioma de entrada     : {Fore.YELLOW}{INPUT_LANGUAGE.upper() if INPUT_LANGUAGE else 'Autodetección'}")
    print(Fore.WHITE + f" - Traducción automática : {Fore.GREEN + 'Activada (Inglés -> Español)' if translator else Fore.YELLOW + 'Desactivada'}")
    print(Fore.WHITE + f" - Modelo Whisper        : {Fore.YELLOW}{MODEL_SIZE} (beam_size=1, multi-hilo)")
    print(Fore.WHITE + f" - Silencio de corte     : {Fore.YELLOW}400 ms (corte inmediato de oración)")
    print(Fore.CYAN + "=" * 65)
    print(Fore.GREEN + Style.BRIGHT + "\nEscuchando continuamente... Reproduce el video o habla.")
    print(Fore.YELLOW + "Presiona [Ctrl + C] para salir en cualquier momento.\n")
    print(Fore.CYAN + "-" * 65)

    if DEBUG_SAVE_AUDIO:
        DEBUG_DIR.mkdir(exist_ok=True)

    capture_count = 0

    try:
        while True:
            # Esperar el siguiente fragmento de voz
            item = recorder.get_speech_segment(timeout=0.1)
            if item is None:
                continue

            audio_16k, duration_sec, rms_level = item
            capture_count += 1

            if DEBUG_SAVE_AUDIO:
                wav_path = DEBUG_DIR / f"capture_{capture_count:03d}.wav"
                AudioCapture.save_wav(audio_16k, str(wav_path))

            # Transcribir audio
            asr_res = engine.transcribe(audio_16k, language=INPUT_LANGUAGE)
            original_text = asr_res["text"]
            t_asr = asr_res["elapsed_time"]

            if not original_text:
                continue

            timestamp = time.strftime("%H:%M:%S")

            # Si la traducción está activada y el audio está en inglés
            if translator is not None:
                t0_tr = time.perf_counter()
                trad_text = translator.translate_en_to_es(original_text)
                t_tr = time.perf_counter() - t0_tr
                t_total = t_asr + t_tr

                print(
                    f"{Fore.CYAN}[{timestamp}] "
                    f"{Fore.LIGHTBLACK_EX}[Audio: {duration_sec:.1f}s | Latencia total: {t_total:.2f}s]\n"
                    f"  {Fore.YELLOW}Original (EN) : {Fore.WHITE}{original_text}\n"
                    f"  {Fore.GREEN}{Style.BRIGHT}Traducción(ES): {Fore.WHITE}{Style.BRIGHT}{trad_text}\n"
                )
            else:
                print(
                    f"{Fore.CYAN}[{timestamp}] "
                    f"{Fore.LIGHTBLACK_EX}[Audio: {duration_sec:.1f}s | Whisper: {t_asr:.2f}s]\n"
                    f"  {Fore.WHITE}{Style.BRIGHT}Texto detectado: {original_text}\n"
                )

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nDetención solicitada por el usuario...")

    finally:
        recorder.stop()
        print(Fore.GREEN + "Captura finalizada y recursos liberados correctamente.")


if __name__ == "__main__":
    main()
