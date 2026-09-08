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
                capture.enqueue(AudioSegment(np.full(64000,i,dtype=np.float32),i*4,(i+1)*4))
            self.app.engine = SimpleNamespace(transcribe=lambda audio, **kwargs: {'text':str(int(audio[0])), 'language':'en'})
        emitted = []
        original_emit = self.app.emit
        def emit(kind, value=None):
            if kind == 'subtitle':
                emitted.append(value[0])
                if len(emitted) == 5:
                    self.app.stopped.set()
            original_emit(kind,value)
        with patch('src.audio.windows_capture.WindowsCapture',return_value=capture), patch.object(self.app,'ensure_engine',side_effect=load), patch.object(self.app,'emit',side_effect=emit), patch.object(self.app.translator,'translate',side_effect=lambda text,*args:text):
            deadline = threading.Timer(2,self.app.stopped.set)
            deadline.start()
            try:
                self.app.run(False,10,'en','es','base')
            finally:
                deadline.cancel()
        self.assertEqual(emitted,['0','1','2','3','4'])
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
