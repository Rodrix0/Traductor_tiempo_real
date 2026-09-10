"""
main.py
Traductor en tiempo real de ultra baja latencia y alta precisión.
Incorpora:
  - Modelo Whisper especializado en inglés (small.en) o multilingüe (small).
  - Prompts contextuales (initial_prompt) para evitar palabras mal escuchadas.
  - Motor de traducción Meta NLLB-200 (modismos y lenguaje moderno) o MarianMT.
  - Glosario personalizable (glossary.json) para calibración exacta de términos.
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

# Micrófono a usar (None para usar el predeterminado de Windows)
MIC_DEVICE_ID = 1

# Idioma de entrada del audio/video:
# "en" = inglés (recomendado para videos en inglés)
# "es" = español
INPUT_LANGUAGE = "en"

# Modelo Whisper:
# "small.en" = Especializado 100% en inglés (máxima precisión léxica y de modismos)
# "small"    = Multilingüe general
# "base"     = Velocidad extrema (< 0.2s)
MODEL_SIZE = "small.en"

# Prompt contextual (None = 100% independiente y autónomo):
WHISPER_PROMPT = None

# ¿Traducir automáticamente al español al terminar la frase?
ENABLE_TRANSLATION = True

# Motor de traducción local:
# "nllb"   = Meta NLLB-200 (Mayor fidelidad en modismos, lenguaje moderno y coloquial)
# "marian" = Helsinki-NLP MarianMT (Ultra-liviano y veloz, ~150ms)
TRANSLATION_ENGINE = "nllb"

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

    print(Fore.CYAN + "=" * 68)
    print(Fore.CYAN + Style.BRIGHT + "   TRADUCTOR EN TIEMPO REAL - MODO ALTA PRECISIÓN Y BAJA LATENCIA")
    print(Fore.CYAN + "=" * 68)

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
        min_speech_duration_ms=400,    # Descarta ruidos menores a 0.4s
        max_speech_duration_ms=4000,   # Máximo 4s para evitar retrasos
    )

    recorder.start()

    try:
        # 3. Cargar faster-whisper con el modelo especializado
        print(Fore.WHITE + f"\nCargando motor de reconocimiento Whisper ('{MODEL_SIZE}')...")
        engine = SpeechToText(
            model_size=MODEL_SIZE,
            default_language=None if MODEL_SIZE.endswith(".en") else INPUT_LANGUAGE,
            beam_size=1,  # Inferencia rápida sin pérdida de precisión en habla clara
        )

        # 4. Cargar traductor local (NLLB-200 o MarianMT)
        translator = None
        if ENABLE_TRANSLATION and INPUT_LANGUAGE == "en":
            print(Fore.WHITE + f"Cargando motor de traducción local [{TRANSLATION_ENGINE.upper()}] con glosario...")
            translator = LocalTranslator(engine=TRANSLATION_ENGINE, device="cpu")

        # 5. Panel informativo
        print("\n" + Fore.CYAN + "=" * 68)
        print(Fore.WHITE + f" - Micrófono seleccionado : {Fore.YELLOW}[ID {selected_info['id']}] {selected_info['name']}")
        print(Fore.WHITE + f" - Idioma de entrada     : {Fore.YELLOW}{INPUT_LANGUAGE.upper()}")
        print(Fore.WHITE + f" - Modelo Whisper        : {Fore.YELLOW}{MODEL_SIZE} (con prompt contextual)")
        print(Fore.WHITE + f" - Motor de traducción   : {Fore.YELLOW}{'Meta NLLB-200 (Alta fidelidad)' if TRANSLATION_ENGINE == 'nllb' else 'MarianMT (Ultra-liviano)'}")
        print(Fore.WHITE + f" - Términos en glosario  : {Fore.GREEN}{len(translator.glossary) if translator else 0} reglas activas (glossary.json)")
        print(Fore.WHITE + f" - Silencio de corte VAD : {Fore.YELLOW}400 ms (corte inmediato de frase)")
        print(Fore.CYAN + "=" * 68)
        print(Fore.GREEN + Style.BRIGHT + "\nEscuchando continuamente... Reproduce el video o habla.")
        print(Fore.YELLOW + "Presiona [Ctrl + C] para salir en cualquier momento.\n")
        print(Fore.CYAN + "-" * 68)

        if DEBUG_SAVE_AUDIO:
            DEBUG_DIR.mkdir(exist_ok=True)

        capture_count = 0

        while True:
            item = recorder.get_speech_segment(timeout=0.1)
            if item is None:
                continue

            audio_16k, duration_sec, rms_level = item
            capture_count += 1

            if DEBUG_SAVE_AUDIO:
                wav_path = DEBUG_DIR / f"capture_{capture_count:03d}.wav"
                AudioCapture.save_wav(audio_16k, str(wav_path))

            # Transcribir audio con prompt contextual
            prompt = WHISPER_PROMPT if INPUT_LANGUAGE == "en" else None
            lang = None if MODEL_SIZE.endswith(".en") else INPUT_LANGUAGE

            asr_res = engine.transcribe(
                audio_16k,
                language=lang,
                initial_prompt=prompt,
            )
            original_text = asr_res["text"]
            t_asr = asr_res["elapsed_time"]

            if not original_text:
                continue

            timestamp = time.strftime("%H:%M:%S")

            # Traducir y mostrar
            if translator is not None:
                t0_tr = time.perf_counter()
                trad_text = translator.translate_en_to_es(original_text)
                t_tr = time.perf_counter() - t0_tr
                t_total = t_asr + t_tr

                print(
                    f"{Fore.CYAN}[{timestamp}] "
                    f"{Fore.LIGHTBLACK_EX}[Audio: {duration_sec:.1f}s | ASR: {t_asr:.2f}s | Trad: {t_tr:.2f}s | Total: {t_total:.2f}s]\n"
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
