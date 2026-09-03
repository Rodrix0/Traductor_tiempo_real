"""Pruebas de integración para la captura de audio con sounddevice."""

import unittest
import time
import numpy as np

from src.audio.recorder import AudioRecorder


class TestAudioRecorder(unittest.TestCase):

    def test_list_input_devices(self):
        """Verifica que el sistema detecte al menos un dispositivo de entrada."""
        devices = AudioRecorder.list_input_devices()
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0, "No se encontraron micrófonos disponibles.")
        for dev in devices:
            self.assertIn("id", dev)
            self.assertIn("name", dev)
            self.assertGreater(dev["channels"], 0)

    def test_recorder_start_and_collect_frames(self):
        """Verifica que el capturador inicie, reciba frames del micrófono y se detenga limpiamente."""
        recorder = AudioRecorder(sample_rate=16000)
        try:
            recorder.start()
            self.assertTrue(recorder.is_recording)

            # Esperar 0.5 segundos para que se acumulen frames
            time.sleep(0.5)

            frame = recorder.get_frame(timeout=1.0)
            self.assertIsNotNone(frame, "No se recibió ningún frame de audio del micrófono.")
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.dtype, np.float32)
            self.assertEqual(len(frame), recorder.blocksize)

        finally:
            recorder.stop()
            self.assertFalse(recorder.is_recording)


if __name__ == "__main__":
    unittest.main()
