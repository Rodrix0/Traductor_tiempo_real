import contextlib
import io
import unittest
from unittest.mock import Mock, patch
import main


class ConsoleStartupTests(unittest.TestCase):
    def test_capture_precedes_models_and_stops_after_exit(self):
        recorder = Mock()
        recorder.get_speech_segment.side_effect = KeyboardInterrupt
        order = []
        recorder.start.side_effect = lambda: order.append('capture')
        def engine(**kwargs):
            order.append('model')
            return Mock()
        with patch('main.AudioCapture',return_value=recorder) as capture_type, patch('main.SpeechToText',side_effect=engine), patch('main.LocalTranslator'), contextlib.redirect_stdout(io.StringIO()):
            capture_type.list_microphones.return_value = [{'id':1}]
            capture_type.resolve_microphone.return_value = {'id':1,'name':'Prueba'}
            main.main()
        self.assertEqual(order,['capture','model'])
        recorder.stop.assert_called_once()

    def test_model_failure_releases_capture(self):
        recorder = Mock()
        with patch('main.AudioCapture',return_value=recorder) as capture_type, patch('main.SpeechToText',side_effect=RuntimeError('modelo no disponible')), contextlib.redirect_stdout(io.StringIO()):
            capture_type.list_microphones.return_value = [{'id':1}]
            capture_type.resolve_microphone.return_value = {'id':1,'name':'Prueba'}
            with self.assertRaisesRegex(RuntimeError,'modelo no disponible'):
                main.main()
        recorder.start.assert_called_once()
        recorder.stop.assert_called_once()
