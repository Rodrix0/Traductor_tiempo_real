"""Opt-in real-model smoke test: python -m tests.integration_local.

Uses only a synthetic fixture in models/speech_smoke.wav; no microphone recording.
"""
from pathlib import Path
import threading
import tempfile
import av
import numpy as np
from src.media import translate_media, media_chunks
from src.asr.whisper_engine import WhisperEngine
from src.translation.local import LocalTranslator
from src.subtitles import export_captions


def create_video(wav, output):
    with av.open(str(wav)) as source, av.open(str(output), 'w') as target:
        video = target.add_stream('mpeg4', rate=10)
        video.width = video.height = 64
        video.pix_fmt = 'yuv420p'
        audio = target.add_stream('aac', rate=16000)
        audio.layout = 'mono'
        for frame in source.decode(audio=0):
            for packet in audio.encode(frame):
                target.mux(packet)
        for packet in audio.encode(None):
            target.mux(packet)
        for index in range(60):
            frame = av.VideoFrame.from_ndarray(np.zeros((64,64,3), dtype=np.uint8), format='rgb24')
            frame.pts = index
            for packet in video.encode(frame):
                target.mux(packet)
        for packet in video.encode(None):
            target.mux(packet)


def main():
    wav = Path('models/speech_smoke.wav')
    if not wav.is_file():
        raise RuntimeError('Falta models/speech_smoke.wav: generar una frase sintética antes de esta prueba.')
    engine = WhisperEngine(model_size='base')
    translator = LocalTranslator()
    with tempfile.TemporaryDirectory() as folder:
        video = Path(folder) / 'sample.mp4'
        create_video(wav, video)
        for path in (wav, video):
            captions = list(translate_media(path, engine, translator, 'en', 'es', threading.Event(), print))
            assert captions and all(c.translated.strip() for c in captions)
            assert all(c.start < c.end for c in captions)
            for format in ('srt', 'vtt'):
                exported = Path(folder) / f'{path.stem}.{format}'
                exported.write_text(export_captions(captions, format), encoding='utf-8')
                assert '-->' in exported.read_text(encoding='utf-8')
            print(path.suffix, [(c.start,c.end,c.translated) for c in captions], flush=True)
    print('ARCHIVOS WAV / MP4 + SRT / VTT OK', flush=True)


if __name__ == '__main__':
    main()
