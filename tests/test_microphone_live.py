"""
Prueba interactiva rápida del micrófono y transcripción en vivo (5 segundos).
Permite verificar que el micrófono físico del usuario esté recibiendo volumen y que Whisper lo transcriba.
"""

import time
import numpy as np
from colorama import init, Fore, Style

from config.settings import SAMPLE_RATE
from src.audio.recorder import AudioRecorder
from src.asr.whisper_engine import WhisperEngine

init(autoreset=True)


def test_live_microphone():
    print(Fore.CYAN + "=" * 65)
    print(Fore.CYAN + Style.BRIGHT + "   PRUEBA INTERACTIVA DE MICRÓFONO Y TRANSCRIPCIÓN (5 SEG)")
    print(Fore.CYAN + "=" * 65)

    devices = AudioRecorder.list_input_devices()
    print(Fore.GREEN + f"Dispositivos de entrada detectados: {len(devices)}")
    for d in devices[:3]:
        print(f"  [{d['id']}] {d['name']}")

    print(Fore.BLUE + "\nCargando motor Whisper...")
    engine = WhisperEngine(model_size="base")
    print(Fore.GREEN + f"[OK] Motor cargado en: {engine.actual_device}")

    recorder = AudioRecorder(sample_rate=SAMPLE_RATE)
    print(Fore.MAGENTA + "\nPrepárate para hablar en 3...")
    for sec in [3, 2, 1]:
        print(Fore.YELLOW + f"  {sec}...")
        time.sleep(1)

    print(Fore.GREEN + Style.BRIGHT + "\n[GRABANDO AHORA] ¡Habla algo por el micrófono durante 5 segundos!...")
    recorder.start()

    collected_frames = []
    start_time = time.time()
    max_energy = 0.0

    while time.time() - start_time < 5.0:
        frame = recorder.get_frame(timeout=0.1)
        if frame is not None:
            energy = float(np.sqrt(np.mean(frame**2)))
            if energy > max_energy:
                max_energy = energy
            collected_frames.append(frame)

    recorder.stop()
    print(Fore.CYAN + "\n[Fin de la grabación]")
    print(Fore.WHITE + f"Pico máximo de energía detectado: {max_energy:.4f}")

    if max_energy < 0.005:
        print(Fore.RED + "AVISO: La energía detectada fue muy baja. Verifica que el micrófono no esté silenciado o el volumen sea adecuado.")
    else:
        print(Fore.GREEN + "[OK] Señal de voz detectada correctamente.")

    if collected_frames:
        full_audio = np.concatenate(collected_frames)
        print(Fore.BLUE + "Transcribiendo audio grabado...")
        res = engine.transcribe(full_audio, language="es")
        print(Fore.CYAN + "-" * 65)
        print(Fore.WHITE + Style.BRIGHT + f"Texto reconocido : \"{res['text']}\"")
        print(Fore.WHITE + f"Idioma detectado : {res['language']} (Probabilidad: {res['probability']:.2f})")
        print(Fore.WHITE + f"Tiempo inferencia: {res['elapsed_time']:.2f} segundos")
        print(Fore.CYAN + "=" * 65)


if __name__ == "__main__":
    test_live_microphone()
