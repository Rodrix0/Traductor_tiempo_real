"""Shared caption representation and UTF-8 exports."""
from dataclasses import dataclass, asdict
import json
import math


from typing import Optional


@dataclass
class Caption:
    start: float
    end: float
    original: str
    translated: str
    language: str
    speaker_id: Optional[str] = None

    def __post_init__(self):
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.start < 0 or self.end <= self.start:
            raise ValueError("Las marcas de tiempo de los subtítulos no son válidas.")

    @property
    def start_time(self) -> float:
        return self.start

    @property
    def end_time(self) -> float:
        return self.end


def timestamp(seconds, separator=','):
    milliseconds = max(0, round(seconds * 1000))
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}'


def export_captions(captions, format, include_speakers: bool = False):
    if format == 'json':
        return json.dumps([asdict(c) for c in captions], ensure_ascii=False, indent=2)
    if format == 'txt':
        lines = []
        for c in captions:
            spk_prefix = f"[{c.speaker_id}] " if include_speakers and c.speaker_id else ""
            lines.append(f'[{timestamp(c.start)}] {spk_prefix}{c.original}\n→ {spk_prefix}{c.translated}')
        return '\n\n'.join(lines) + '\n'
    if format not in ('srt', 'vtt'):
        raise ValueError('Formato no admitido.')
    result = ['WEBVTT\n'] if format == 'vtt' else []
    for i, caption in enumerate(captions, 1):
        separator = '.' if format == 'vtt' else ','
        spk_prefix = f"{caption.speaker_id}: " if include_speakers and caption.speaker_id else ""
        text = ' '.join(caption.translated.split())
        if format == 'vtt':
            import html
            text = html.escape(f"{spk_prefix}{text}")
        else:
            text = f"{spk_prefix}{text}"
        result.append(f'{i}\n{timestamp(caption.start, separator)} --> {timestamp(caption.end, separator)}\n{text}\n')
    return '\n'.join(result) + '\n'
