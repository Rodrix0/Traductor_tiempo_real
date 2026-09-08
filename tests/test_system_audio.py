import unittest
import numpy as np
from src.audio.windows_capture import to_mono_16k, WindowsCapture, AudioSegment
from unittest.mock import patch


class SystemAudioTests(unittest.TestCase):
    def test_backlog_preserves_first_segments_with_original_timestamps(self):
        capture = WindowsCapture(0)
        for i in range(6):
            capture.enqueue(AudioSegment(np.zeros(480), i, i + 1))
        self.assertEqual(capture.dropped, 0)
        self.assertEqual([capture.segments.get_nowait().start for _ in range(6)], [0,1,2,3,4,5])

    def test_buffer_limit_stops_capture_without_evicting_old_audio(self):
        capture = WindowsCapture(0, buffer_seconds=2)
        for i in range(3):
            capture.enqueue(AudioSegment(np.zeros(16000),i,i+1))
        self.assertTrue(capture.stopped.is_set())
        self.assertIsNotNone(capture.error)
        self.assertEqual(capture.pending_seconds,2)
        self.assertEqual([capture.segments.get_nowait().start for _ in range(2)],[0,1])

    def test_start_reports_device_failure(self):
        def failed(capture):
            capture.error = OSError("desconectado")
            capture.ready.set()
        with patch.object(WindowsCapture, '_record', failed):
            with self.assertRaisesRegex(RuntimeError, "desconectado"):
                WindowsCapture(99).start()

    def test_stereo_resampling_preserves_pitch(self):
        rate = 48000
        time = np.arange(rate) / rate
        wave = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        raw = np.column_stack((wave, wave)).tobytes()
        result = to_mono_16k(raw, 2, rate)
        self.assertEqual(result.shape, (16000,))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(np.argmax(abs(np.fft.rfft(result))), 440)
        self.assertAlmostEqual(float(np.max(result)), 0.2, places=2)

    def test_44100_stereo_silence(self):
        raw = np.zeros((1323, 2), dtype=np.float32).tobytes()
        result = to_mono_16k(raw, 2, 44100)
        self.assertEqual(len(result), 480)
        self.assertTrue(np.all(result == 0))


if __name__ == '__main__':
    unittest.main()
