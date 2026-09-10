"""
test_diagnostics.py
Pruebas para el módulo de diagnóstico y benchmark de hardware.
"""

import unittest
from unittest.mock import MagicMock
from src.ui.diagnostics import SystemDiagnostics, SystemBenchmark


class DiagnosticsTests(unittest.TestCase):

    def test_system_summary_keys(self):
        summary = SystemDiagnostics.get_system_summary()
        self.assertIn("cuda_available", summary)
        self.assertIn("recommended_device", summary)
        self.assertIn("recommended_compute", summary)

    def test_benchmark_runner_mock(self):
        engine_mock = MagicMock()
        engine_mock.device = "cpu"
        engine_mock.transcribe.return_value = {"text": "ok", "elapsed_time": 0.05}

        translator_mock = MagicMock()
        translator_mock.translate.return_value = "prueba"

        res = SystemBenchmark.run_benchmark(engine=engine_mock, translator=translator_mock)
        self.assertTrue(res["benchmark_passed"])
        self.assertGreaterEqual(res["asr_latency_ms"], 0.0)
        self.assertGreaterEqual(res["total_latency_ms"], 0.0)
        self.assertEqual(res["asr_device"], "cpu")


if __name__ == "__main__":
    unittest.main()
