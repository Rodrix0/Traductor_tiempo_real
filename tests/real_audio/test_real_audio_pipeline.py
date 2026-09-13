"""
tests/real_audio/test_real_audio_pipeline.py
Suite de pruebas de regresión acústica sobre audio real y ground truth.
Verifica:
  - Captura íntegra de frases limpias individuales (sin cortes).
  - Respuestas cortas (Short Utterance Recall 100%).
  - Robustez a consonantes iniciales tenues y desvanecimiento final (soft onset / offset).
  - Alternancia y memoria de dos hablantes (A -> B -> A).
  - Diarización intra-segmento con word-timestamps (IntraSegmentDiarizer).
  - Ejecución 100% local y offline (sin llamadas de red).
"""

import os
import sys
import wave
import socket
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

import numpy as np
import scipy.signal

from src.audio.vad import VoiceActivityDetector
from src.audio.windows_capture import AudioSegment
from src.asr.whisper_engine import WhisperEngine
from src.asr.retry_engine import ContextAwareSTTRetry
from src.asr.anomaly_detector import STTAnomalyDetector
from src.pipeline.segment_reassembler import SegmentReassembler
from src.speakers.tracker import SpeakerTracker
from src.speakers.intra_segment_diarizer import IntraSegmentDiarizer
from src.utils.metrics import compute_wer

REAL_AUDIO_DIR = Path(__file__).resolve().parent


def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
        if sr != 16000:
            data = scipy.signal.resample(data, int(len(data) * 16000 / sr))
        return data


class RealAudioPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = WhisperEngine(model_size="base")
        cls.retry_engine = ContextAwareSTTRetry(
            asr_engine=cls.engine,
            ring_buffer=None,
            anomaly_detector=STTAnomalyDetector(),
        )

    def test_single_speaker_clean(self):
        """Verifica que una frase completa de un solo hablante se transcriba íntegramente."""
        audio_path = REAL_AUDIO_DIR / "single_speaker_01.wav"
        self.assertTrue(audio_path.exists(), f"Falta fixture {audio_path}")
        audio = load_audio(audio_path)

        res = self.engine.transcribe(audio, language="en", vad_filter=False)
        text = res.get("text", "").strip()
        self.assertTrue(len(text) > 0)
        wer = compute_wer("Thank you very much for coming today.", text)
        self.assertLessEqual(wer, 0.15, f"WER demasiado alto ({wer:.2f}) para audio limpio: '{text}'")

    def test_short_utterance_recall(self):
        """Verifica que respuestas ultra-cortas no sean descartadas por VAD o Whisper."""
        short_words = ["yeah", "right", "perfect", "well"]
        for word in short_words:
            p = REAL_AUDIO_DIR / f"short_{word}.wav"
            self.assertTrue(p.exists(), f"Falta fixture {p}")
            audio = load_audio(p)

            vad = VoiceActivityDetector(sample_rate=16000)
            chunk_size = 480
            detected = []
            for i in range(0, len(audio), chunk_size):
                seg = vad.process_frame(audio[i : i + chunk_size])
                if seg is not None:
                    detected.append(seg)
            rem = vad.flush()
            if rem is not None:
                detected.append(rem)

            self.assertGreater(len(detected), 0, f"VAD descartó respuesta corta '{word}'")
            stt = self.engine.transcribe(detected[0], language="en", vad_filter=False)
            txt = stt.get("text", "").strip().lower()
            self.assertTrue(word in txt, f"Whisper no transcribió '{word}'. Salida: '{txt}'")

    def test_soft_onset_and_offset_clipping(self):
        """Verifica que consonantes iniciales débiles y finales no se pierdan por recorte en bordes."""
        onset_path = REAL_AUDIO_DIR / "soft_onset.wav"
        offset_path = REAL_AUDIO_DIR / "soft_ending.wav"

        audio_onset = load_audio(onset_path)
        audio_offset = load_audio(offset_path)

        res_onset = self.engine.transcribe(audio_onset, language="en", vad_filter=False)
        txt_onset = res_onset.get("text", "").strip().lower()
        self.assertTrue("probably" in txt_onset, f"Fonema inicial débil cortado: '{txt_onset}'")

        res_offset = self.engine.transcribe(audio_offset, language="en", vad_filter=False)
        txt_offset = res_offset.get("text", "").strip().lower()
        self.assertTrue("believe" in txt_offset, f"Palabra final desvanecida cortada: '{txt_offset}'")

    def test_two_speakers_turn_taking_memory(self):
        """Verifica que una conversación A -> B -> A registre SPEAKER_01 y SPEAKER_02 sin confusiones."""
        spk_a1 = load_audio(REAL_AUDIO_DIR / "single_speaker_01.wav")
        spk_b1 = load_audio(Path(__file__).resolve().parent.parent.parent / "models" / "speech_smoke.wav")
        spk_a2 = load_audio(REAL_AUDIO_DIR / "soft_onset.wav")

        tracker = SpeakerTracker(sample_rate=16000, similarity_threshold=0.91)

        id1 = tracker.register_solo_speech(spk_a1, duration=len(spk_a1)/16000.0)
        id2 = tracker.register_solo_speech(spk_b1, duration=len(spk_b1)/16000.0)
        id3 = tracker.register_solo_speech(spk_a2, duration=len(spk_a2)/16000.0)

        self.assertEqual(id1, "SPEAKER_01")
        self.assertEqual(id2, "SPEAKER_02", "No se detectó el cambio de hablante a SPEAKER_02")
        self.assertEqual(id3, "SPEAKER_01", "No se reconoció a SPEAKER_01 cuando volvió a hablar")

    def test_intra_segment_diarization(self):
        """Verifica que IntraSegmentDiarizer divida un bloque continuo de audio con 2 hablantes."""
        audio_path = REAL_AUDIO_DIR / "intra_segment_two_speakers.wav"
        self.assertTrue(audio_path.exists(), f"Falta fixture {audio_path}")
        audio = load_audio(audio_path)
        dur = len(audio) / 16000.0

        tracker = SpeakerTracker(sample_rate=16000, similarity_threshold=0.91)
        diarizer = IntraSegmentDiarizer(
            speaker_tracker=tracker,
            sample_rate=16000,
            min_split_duration_seconds=0.8,
            min_word_pause_split_seconds=0.08,
        )

        res = self.engine.transcribe(audio, language="en", vad_filter=False)
        words = res.get("words", [])
        subsegs = diarizer.diarize_segment(
            audio=audio,
            base_start_time=0.0,
            base_end_time=dur,
            words=words,
            fallback_text=res.get("text", ""),
        )

        self.assertGreaterEqual(len(subsegs), 2, "IntraSegmentDiarizer no dividió el bloque continuo con dos hablantes")
        self.assertNotEqual(subsegs[0].speaker_id, subsegs[1].speaker_id, "Ambos subsegmentos tienen el mismo speaker_id")

    def test_real_audio_offline_compliance(self):
        """Verifica que todo el pipeline acústico opere 100% offline sin peticiones de red."""
        orig_socket = socket.socket
        def forbidden_socket(*args, **kwargs):
            raise RuntimeError("Operación de red denegada en modo offline estricto.")

        socket.socket = forbidden_socket
        try:
            audio = load_audio(REAL_AUDIO_DIR / "single_speaker_01.wav")
            res = self.engine.transcribe(audio, language="en", vad_filter=False)
            self.assertTrue(len(res["text"]) > 0)
        finally:
            socket.socket = orig_socket


if __name__ == "__main__":
    unittest.main()
