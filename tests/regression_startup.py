"""Opt-in real WASAPI startup regression; plays a synthetic clip through speakers."""
import time
import winsound
from pathlib import Path
from src.audio.windows_capture import WindowsCapture, list_sources
from src.translation.local import LocalTranslator


def main():
    path = Path('models/startup_regression.wav')
    assert path.is_file(), 'Generá primero la frase sintética de prueba'
    capture = WindowsCapture(list_sources()[1])
    capture.start()
    started = time.monotonic()
    try:
        # Audio starts before imports/model loads, exactly as a video does on startup.
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        from src.asr.whisper_engine import WhisperEngine
        engine = WhisperEngine(model_size='base', device='cpu')
        translator = LocalTranslator()
        # Deliberately emulate twenty seconds of model setup without consuming audio.
        time.sleep(max(0,20-(time.monotonic()-started)))
        capture.stop()
        assert capture.error is None, capture.error
        segments = []
        while not capture.segments.empty():
            segments.append(capture.segments.get_nowait())
        assert segments, 'No se capturó audio de los altavoces'
        text = ' '.join(engine.transcribe(segment.audio,language='en')['text'] for segment in segments)
        translated = translator.translate(text,'en','es')
        print('Original capturado durante los 20 s de carga:', text, flush=True)
        print('Traducción:',translated,flush=True)
        assert 'press' in text.lower() and ('jump' in text.lower()), 'No se reconoció correctamente la primera instrucción'
        assert 'prensa x' not in translated.lower()
        assert 'salt' in translated.lower()
        print('CAPTURA DURANTE CARGA + PRIMERA FRASE + TRADUCCIÓN: OK',flush=True)
    finally:
        capture.stop()
        winsound.PlaySound(None, 0)


if __name__ == '__main__':
    main()
