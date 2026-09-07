"""Shared caption representation and UTF-8 exports."""
from dataclasses import dataclass, asdict
import json
import math


@dataclass
class Caption:
    start: float
    end: float
    original: str
    translated: str
    language: str

    def __post_init__(self):
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.start < 0 or self.end <= self.start:
            raise ValueError("Las marcas de tiempo de los subtítulos no son válidas.")


def timestamp(seconds, separator=','):
    milliseconds = max(0, round(seconds * 1000))
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}'


def export_captions(captions, format):
    if format == 'json':
        return json.dumps([asdict(c) for c in captions], ensure_ascii=False, indent=2)
    if format == 'txt':
        return '\n\n'.join(f'[{timestamp(c.start)}] {c.original}\n→ {c.translated}' for c in captions) + '\n'
    if format not in ('srt', 'vtt'):
        raise ValueError('Formato no admitido.')
    result = ['WEBVTT\n'] if format == 'vtt' else []
    for i, caption in enumerate(captions, 1):
        separator = '.' if format == 'vtt' else ','
        text = ' '.join(caption.translated.split())
        if format == 'vtt':
            import html
            text = html.escape(text)
        result.append(f'{i}\n{timestamp(caption.start, separator)} --> {timestamp(caption.end, separator)}\n{text}\n')
    return '\n'.join(result) + '\n'
