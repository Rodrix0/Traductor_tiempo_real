import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.io import wavfile
from src.media import media_chunks, translate_media
from src.subtitles import Caption, export_captions, timestamp
from src.audio.windows_capture import AudioSegment


class MediaTests(unittest.TestCase):
    def test_wav_chunks_keep_timeline_and_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audio.wav'
            wavfile.write(path, 16000, np.zeros(40000, dtype=np.int16))
            chunks = list(media_chunks(path, threading.Event(), chunk_seconds=1))
            self.assertEqual([len(c.audio) for c in chunks], [16000, 16000, 8000])
            self.assertEqual([(c.start, c.end) for c in chunks], [(0,1), (1,2), (2,2.5)])

    def test_invalid_file_and_cancellation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.mp4'
            path.write_text('invalid')
            with self.assertRaises(Exception):
                list(media_chunks(path, threading.Event()))
            wavfile.write(path.with_suffix('.wav'), 16000, np.zeros(16000, dtype=np.int16))
            stopped = threading.Event()
            stopped.set()
            self.assertEqual(list(media_chunks(path.with_suffix('.wav'), stopped)), [])

    def test_translation_offsets_and_cancel_during_inference(self):
        stopped = threading.Event()
        result = dict(text='Hello', language='en', segments=[dict(start=1, end=2, text='Hello')])
        engine = SimpleNamespace(transcribe=lambda *args, **kwargs: result)
        translator = SimpleNamespace(translate=lambda *args: 'Hola')
        chunks = [AudioSegment(np.zeros(48000), 30, 33)]
        with patch('src.media.media_chunks', return_value=iter(chunks)):
            captions = list(translate_media('fake', engine, translator, 'en', 'es', stopped, lambda _: None))
        self.assertEqual((captions[0].start, captions[0].end), (31,32))
        def cancel(*args, **kwargs):
            stopped.set()
            return result
        engine.transcribe = cancel
        with patch('src.media.media_chunks', return_value=iter(chunks)):
            self.assertEqual(list(translate_media('fake', engine, translator, 'en','es', stopped, lambda _: None)), [])

    def test_export_formats_unicode_and_rounding(self):
        captions = [Caption(59.9996, 62.125, 'Hello', 'Olá & <mundo>', 'en')]
        self.assertEqual(timestamp(59.9996), '00:01:00,000')
        self.assertIn('00:01:00,000 --> 00:01:02,125', export_captions(captions, 'srt'))
        vtt = export_captions(captions, 'vtt')
        self.assertTrue(vtt.startswith('WEBVTT\n\n'))
        self.assertIn('Olá &amp; &lt;mundo&gt;', vtt)
        self.assertIn('Olá', export_captions(captions, 'json'))
        with self.assertRaises(ValueError):
            Caption(3, 2, '', '', 'en')
