"""Decode local media in bounded chunks without loading a movie into memory."""
from pathlib import Path
import numpy as np
from src.audio.windows_capture import AudioSegment
from src.subtitles import Caption
from src.translation.local import LANGUAGES


def media_chunks(path, stopped, chunk_seconds=30):
    import av
    path = Path(path)
    if not path.is_file():
        raise ValueError('El archivo no existe.')
    size = int(chunk_seconds * 16000)
    if size <= 0:
        raise ValueError('Duración de bloque inválida.')
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError('El archivo no contiene una pista de audio.')
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        buffer = np.empty(0, dtype=np.float32)
        position = None
        origin = (container.start_time or 0) / av.time_base

        def converted_frames():
            for frame in container.decode(stream):
                if stopped.is_set():
                    return
                yield from resampler.resample(frame)
            if not stopped.is_set():
                yield from resampler.resample(None)

        for frame in converted_frames():
            if stopped.is_set():
                return
            frame_time = max(0, float(frame.time) - origin) if frame.time is not None else None
            if position is None:
                position = frame_time or 0
            elif frame_time is not None and abs(frame_time - (position + len(buffer) / 16000)) > 0.1:
                previous_end = position + len(buffer) / 16000
                if len(buffer):
                    yield AudioSegment(buffer, position, position + len(buffer) / 16000)
                buffer = np.empty(0, dtype=np.float32)
                position = max(previous_end, frame_time)
            buffer = np.concatenate((buffer, frame.to_ndarray().reshape(-1)))
            while len(buffer) >= size:
                yield AudioSegment(buffer[:size].copy(), position, position + size / 16000)
                buffer = buffer[size:]
                position += size / 16000
        if len(buffer) and not stopped.is_set():
            yield AudioSegment(buffer, position, position + len(buffer) / 16000)


def translate_media(path, engine, translator, source, target, stopped, progress):
    for chunk in media_chunks(path, stopped):
        if stopped.is_set():
            return
        progress(f'Procesando audio: {chunk.end:.0f} segundos…')
        result = engine.transcribe(chunk.audio, language=source)
        if stopped.is_set():
            return
        if not result['text']:
            continue
        language = result['language']
        if language not in LANGUAGES:
            raise ValueError('Se detectó otro idioma. Seleccioná el idioma del archivo manualmente.')
        for segment in result['segments']:
            if stopped.is_set():
                return
            start = max(chunk.start, chunk.start + segment['start'])
            end = min(chunk.end, chunk.start + segment['end'])
            if end <= start:
                continue
            translated = translator.translate(segment['text'], language, target)
            if stopped.is_set():
                return
            yield Caption(start, end, segment['text'], translated, language)
