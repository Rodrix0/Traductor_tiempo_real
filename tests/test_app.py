import tkinter as tk
import unittest
import tempfile
from unittest.mock import patch
from traductor import TranslatorApp
from src.subtitles import Caption
from src.preferences import load_preferences
from types import SimpleNamespace
import numpy as np
import threading
from src.audio.windows_capture import WindowsCapture, AudioSegment


class AppTests(unittest.TestCase):
    def test_live_captures_first_twenty_seconds_during_model_loading(self):
        capture = WindowsCapture(10)
        capture.start = lambda: setattr(capture, 'started', True)
        capture.stop = lambda: None
        def load(model):
            self.assertTrue(capture.started, 'La captura debe abrirse antes de cargar modelos')
            for i in range(5):
                capture.enqueue(AudioSegment(
                    np.full(64000, i, dtype=np.float32), i * 4, (i + 1) * 4,
                    trailing_silence_ms=800,
                ))
            self.app.engine = SimpleNamespace(transcribe=lambda audio, **kwargs: {
                'text': f"{int(np.max(audio))}.", 'language': 'en'
            })
        emitted = []
        original_emit = self.app.emit
        def emit(kind, value=None):
            if kind == 'subtitle':
                emitted.append(value[0])
                if len(emitted) == 5:
                    self.app.stopped.set()
            original_emit(kind,value)
        with patch('src.audio.windows_capture.WindowsCapture',return_value=capture), patch.object(self.app,'ensure_engine',side_effect=load), patch.object(self.app,'emit',side_effect=emit), patch.object(self.app.translator,'translate',side_effect=lambda text,*args:text), patch('src.speakers.tracker.SpeakerTracker.register_solo_speech', return_value='SPEAKER_01'):
            deadline = threading.Timer(2,self.app.stopped.set)
            deadline.start()
            try:
                self.app.run(False,10,'en','es','base')
            finally:
                deadline.cancel()
        self.assertEqual(emitted,['0.','1.','2.','3.','4.'])
        self.assertEqual(self.app.store.search()[0]['count'],5)
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.folder = tempfile.TemporaryDirectory()
        with patch('src.audio.windows_capture.list_sources', return_value=([(10, 'Audio PC · Prueba')], 10)):
            self.app = TranslatorApp(self.root, data_dir=self.folder.name)

    def tearDown(self):
        self.app.close()
        self.folder.cleanup()

    def test_saved_session_can_be_selected_and_read(self):
        key = self.app.store.create('Prueba','archivo','en','pt','base')
        self.app.store.add(key, Caption(0,1,'Hello','Olá','en'))
        self.app.store.finish(key,'completado')
        self.app.search_text.set('Olá')
        self.app.reload_history()
        self.app.sessions.selection_set(key)
        self.app.select_history()
        self.assertIn('Olá', self.app.history_preview.get('1.0','end'))

    def test_settings_apply_to_both_modes_and_disk(self):
        self.app.pref_origin.set('Português')
        self.app.pref_target.set('English')
        self.app.pref_model.set('small')
        self.app.pref_compute.set('CPU')
        self.assertTrue(self.app.save_settings_action())
        self.assertEqual(self.app.origin.get(), 'Português')
        self.assertEqual(self.app.file_target.get(), 'English')
        self.assertEqual(self.app.file_model.get(), 'small')
        preferences, warning = load_preferences(self.app.data_dir/'preferences.json')
        self.assertEqual(preferences.compute, 'cpu')
        self.assertEqual(preferences.source, 'pt')
        self.assertEqual(warning, '')

    def test_processor_change_reloads_model(self):
        fake = SimpleNamespace(actual_device='cpu')
        with patch('src.asr.whisper_engine.WhisperEngine', return_value=fake) as factory:
            self.app.active_compute = 'cpu'
            self.app.ensure_engine('base')
            self.app.ensure_engine('base')
            self.assertEqual(factory.call_count,1)
            factory.assert_called_with(model_size='base',device='cpu')
            self.app.active_compute = 'auto'
            self.app.ensure_engine('base')
            self.assertEqual(factory.call_count,2)

    def test_file_worker_saves_history_and_partial_state(self):
        def cancelled(*args):
            yield Caption(0,1,'Hello','Hola','en')
            self.app.stopped.set()
        with patch.object(self.app,'ensure_engine'), patch('src.media.translate_media',side_effect=cancelled):
            self.app.run_file('test.wav','en','es','base')
        self.app.poll()
        row = self.app.store.search()[0]
        self.assertEqual(row['state'],'cancelado')
        self.assertEqual(row['count'],1)
        self.assertEqual(self.app.file_captions[0].translated,'Hola')

    def test_close_waits_for_worker(self):
        self.app.worker = SimpleNamespace(is_alive=lambda:True)
        self.app.close()
        self.assertTrue(self.app.stopped.is_set())
        self.assertTrue(self.root.winfo_exists())
        self.app.worker = None

    def test_file_result_reaches_preview(self):
        caption = Caption(1, 2, 'Hello', 'Olá', 'en')
        self.app.emit('file_caption', caption)
        self.app.poll()
        self.assertEqual(self.app.file_captions, [caption])
        self.assertIn('Olá', self.app.file_preview.get('1.0','end'))

    def test_notebook_manages_all_four_pages(self):
        self.assertEqual(len(self.app.tabs.tabs()),4)
        for name in self.app.tabs.tabs():
            self.assertEqual(self.root.nametowidget(name).winfo_manager(),'notebook')

    def test_invalid_file_does_not_start_worker(self):
        self.app.file_path.set('missing-file.wav')
        with patch('traductor.messagebox.showerror') as error:
            self.app.launch_file()
        error.assert_called_once()
        self.assertIsNone(self.app.worker)

    def test_source_failure_is_visible(self):
        with patch('src.audio.windows_capture.list_sources', side_effect=OSError('sin dispositivo')):
            self.app.load_sources()
        self.assertEqual(self.app.devices, [])
        self.assertIn('sin dispositivo', self.app.status.get())

    def test_live_overlap_speaker_separation_path_in_app(self):
        """
        Verifica que ante un segmento con solapamiento, TranslatorApp.run ejecute la separación,
        genere los Captions usando start_time/end_time sin AttributeError, y almacene ambos hablantes.
        """
        capture = WindowsCapture(10)
        capture.start = lambda: setattr(capture, 'started', True)
        capture.stop = lambda: None

        sr = 16000
        t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False, dtype=np.float32)
        v_a = (0.5 * np.sin(2 * np.pi * 130.0 * t) + 0.25 * np.sin(2 * np.pi * 260.0 * t)).astype(np.float32)
        v_b = (0.5 * np.sin(2 * np.pi * 240.0 * t) + 0.25 * np.sin(2 * np.pi * 480.0 * t)).astype(np.float32)
        mixed = v_a + v_b

        calls = [0]
        def mock_transcribe(audio, **kwargs):
            calls[0] += 1
            txt = "We should leave right now" if calls[0] % 2 == 1 else "No wait stay here please"
            return {'text': txt, 'language': 'en', 'elapsed_time': 0.05}

        def load(model):
            capture.enqueue(AudioSegment(mixed, 10.0, 11.5))
            self.app.engine = SimpleNamespace(transcribe=mock_transcribe)

        emitted = []
        original_emit = self.app.emit
        def emit(kind, value=None):
            if kind == 'subtitle':
                emitted.append(value)
                if len(emitted) >= 2:
                    self.app.stopped.set()
            original_emit(kind, value)

        from src.speakers.models import OverlapResult
        with patch('src.audio.windows_capture.WindowsCapture', return_value=capture), \
             patch.object(self.app, 'ensure_engine', side_effect=load), \
             patch.object(self.app, 'emit', side_effect=emit), \
             patch.object(self.app.translator, 'translate', side_effect=lambda text, *args: f"Trad: {text}"), \
             patch('src.speakers.overlap_detector.OverlapDetector.detect', return_value=OverlapResult(has_overlap=True, speaker_count=2, confidence=0.9)):
            deadline = threading.Timer(3, self.app.stopped.set)
            deadline.start()
            try:
                self.app.run(False, 10, 'en', 'es', 'base')
            finally:
                deadline.cancel()

        self.assertEqual(len(emitted), 2, "Debe emitir 2 subtítulos (uno por cada voz separada)")
        self.assertEqual(self.app.store.search()[0]['count'], 2)
        captions = self.app.store.captions(self.app.store.search()[0]['id'])
        self.assertAlmostEqual(captions[0].start, 10.0, places=2)
        self.assertAlmostEqual(captions[0].end, 11.5, places=2)

    def test_live_overlap_acoustic_rejection_fallback_in_app(self):
        """
        Verifica el flujo real donde OverlapDetector dispara solapamiento, pero SeparationValidator
        rechaza la separación acústica (fuga de voz dominante / duplicado) y ejecuta el fallback:
        - Ejecuta logger.info sin NameError.
        - Se recupera automáticamente procesando como voz individual.
        - Emite 1 solo subtítulo y NO dispara el diálogo de error 'No se pudo continuar'.
        """
        capture = WindowsCapture(10)
        capture.start = lambda: setattr(capture, 'started', True)
        capture.stop = lambda: None

        sr = 16000
        t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False, dtype=np.float32)
        # Una sola voz dominante
        v_a = (0.5 * np.sin(2 * np.pi * 130.0 * t) + 0.25 * np.sin(2 * np.pi * 260.0 * t)).astype(np.float32)

        def mock_transcribe(audio, **kwargs):
            return {'text': 'I love working on myself', 'language': 'en', 'elapsed_time': 0.04}

        def load(model):
            capture.enqueue(AudioSegment(v_a, 5.0, 6.5))
            self.app.engine = SimpleNamespace(transcribe=mock_transcribe)

        emitted = []
        errors = []
        original_emit = self.app.emit
        def emit(kind, value=None):
            if kind == 'subtitle':
                emitted.append(value)
                self.app.stopped.set()
            elif kind == 'error':
                errors.append(value)
            original_emit(kind, value)

        from src.speakers.models import OverlapResult
        with patch('src.audio.windows_capture.WindowsCapture', return_value=capture), \
             patch.object(self.app, 'ensure_engine', side_effect=load), \
             patch.object(self.app, 'emit', side_effect=emit), \
             patch.object(self.app.translator, 'translate', side_effect=lambda text, *args: f"Trad: {text}"), \
             patch('src.separation.validator.SeparationValidator.validate', return_value=SimpleNamespace(is_valid_two_speakers=False, reason='SAME_SPEAKER')), \
             patch('src.speakers.overlap_detector.OverlapDetector.detect', return_value=OverlapResult(has_overlap=True, speaker_count=2, confidence=0.85)):
            deadline = threading.Timer(3, self.app.stopped.set)
            deadline.start()
            try:
                self.app.run(False, 10, 'en', 'es', 'base')
            finally:
                deadline.cancel()

        self.assertEqual(errors, [], "No debe haber ningún error emitido en el fallback")
        self.assertEqual(len(emitted), 1, "Debe emitir exactamente 1 subtítulo tras el fallback a voz individual")
        self.assertIn("I love working on myself", emitted[0][0])

