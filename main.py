"""
Punto de entrada principal para la Etapa 1:
Captura continua de audio desde micrófono, detección de actividad vocal (VAD)
y transcripción local en tiempo real con faster-whisper.
"""

import sys
import time
import logging
from colorama import init, Fore, Style

from config.settings import (
    SAMPLE_RATE,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    INPUT_LANGUAGE,
    ENERGY_THRESHOLD,
    SILENCE_DURATION_MS,
)
from src.audio.recorder import AudioRecorder
from src.audio.vad import VoiceActivityDetector
from src.asr.whisper_engine import WhisperEngine

# Inicializar colorama para terminales Windows
init(autoreset=True)

# Forzar utf-8 si la terminal lo soporta
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TraductorTiempoReal")


def print_banner() -> None:
    """Muestra el banner informativo en la consola."""
    print(Fore.CYAN + "=" * 65)
    print(Fore.CYAN + Style.BRIGHT + "   TRADUCTOR EN TIEMPO REAL - ETAPA 1 (Audio + Transcripción Local)")
    print(Fore.CYAN + "=" * 65)
    print(Fore.WHITE + f" - Modelo Whisper : {Fore.YELLOW}{WHISPER_MODEL_SIZE}")
    print(Fore.WHITE + f" - Dispositivo     : {Fore.YELLOW}{WHISPER_DEVICE}")
    print(Fore.WHITE + f" - Idioma entrada  : {Fore.YELLOW}{INPUT_LANGUAGE or 'Autodetección'}")
    print(Fore.WHITE + f" - Sensibilidad VAD: {Fore.YELLOW}Umbral {ENERGY_THRESHOLD} | Silencio {SILENCE_DURATION_MS}ms")
    print(Fore.WHITE + f" - Frecuencia audio: {Fore.YELLOW}{SAMPLE_RATE} Hz (Mono)")
    print(Fore.CYAN + "-" * 65)


def main() -> None:
    print_banner()

    # 1. Comprobar dispositivos de audio disponibles
    devices = AudioRecorder.list_input_devices()
    if not devices:
        print(Fore.RED + "Error: No se detectó ningún micrófono de entrada disponible.")
        sys.exit(1)

    print(Fore.GREEN + f"Micrófonos detectados ({len(devices)}):")
    for d in devices[:3]:  # Mostrar los primeros 3
        print(Fore.WHITE + f"  [{d['id']}] {d['name']}")
    if len(devices) > 3:
        print(Fore.WHITE + f"  ... y {len(devices) - 3} dispositivo(s) más.")

    # 2. Inicializar el motor de transcripción Whisper
    print(Fore.BLUE + "\nInicializando motor local faster-whisper...")
    try:
        engine = WhisperEngine(
            model_size=WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
        )
    except Exception as e:
        print(Fore.RED + f"Error crítico al inicializar Whisper: {e}")
        sys.exit(1)

    print(Fore.GREEN + f"[OK] Motor listo en dispositivo: [{engine.actual_device.upper()} - {engine.actual_compute_type}]")

    # 3. Inicializar grabador de audio y detector VAD
    recorder = AudioRecorder(sample_rate=SAMPLE_RATE)
    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)

    print(Fore.MAGENTA + "\nIniciando escucha activa del micrófono...")
    print(Fore.YELLOW + "Presiona [Ctrl + C] en cualquier momento para detener.\n")
    print(Fore.CYAN + "=" * 65)

    recorder.start()

    speaking_indicator_shown = False

    try:
        while recorder.is_recording:
            # Obtener frame de audio (30ms por frame)
            frame = recorder.get_frame(timeout=0.05)
            if frame is None:
                continue

            # Mostrar indicador visual cuando comienza a hablar
            if vad.is_speaking and not speaking_indicator_shown:
                print(Fore.YELLOW + "[>>] [Escuchando voz...]", end="\r", flush=True)
                speaking_indicator_shown = True

            # Procesar frame en el VAD
            phrase_audio = vad.process_frame(frame)

            if phrase_audio is not None:
                speaking_indicator_shown = False
                audio_duration = len(phrase_audio) / SAMPLE_RATE
                print(" " * 40, end="\r")  # Limpiar indicador de escucha

                # Transcribir el segmento completo detectado
                res = engine.transcribe(phrase_audio, language=INPUT_LANGUAGE)
                texto = res["text"]
                tiempo = res["elapsed_time"]
                lang = res["language"]

                if texto:
                    timestamp = time.strftime("%H:%M:%S")
                    print(
                        f"{Fore.GREEN}[{timestamp}]{Style.RESET_ALL} "
                        f"{Fore.WHITE}{Style.BRIGHT}{texto} "
                        f"{Fore.CYAN}(Audio: {audio_duration:.1f}s | Latencia: {tiempo:.2f}s | Lang: {lang})"
                    )

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nDetención solicitada por el usuario...")

    finally:
        recorder.stop()
        print(Fore.GREEN + "Captura finalizada y recursos liberados correctamente.")


if __name__ == "__main__":
    main()
