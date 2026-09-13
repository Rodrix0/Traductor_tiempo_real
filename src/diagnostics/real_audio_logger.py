"""
src/diagnostics/real_audio_logger.py
Registrador y volcado físico de audio real e inspección palabra por palabra (Word-Level Debug).
Guarda exactamente el array recibido por Whisper en un archivo WAV físico y metadatos JSON completos,
manteniendo por separado las cuatro etapas del texto:
  - RAW_WHISPER_TEXT
  - RETRY_WHISPER_TEXT
  - REASSEMBLED_TEXT
  - FINAL_SOURCE_TEXT
"""

import os
import json
import wave
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from config.settings import DEBUG_AUDIO_DIR, REAL_AUDIO_DEBUG

logger = logging.getLogger("real_audio_debug")


class RealAudioLogger:
    """
    Gestiona el almacenamiento exacto del audio que entra a faster-whisper
    y genera trazabilidad completa de palabras, confianzas y etapas de texto.
    """

    def __init__(self, output_dir: Optional[Path] = None, enabled: bool = REAL_AUDIO_DEBUG):
        self.output_dir = Path(output_dir) if output_dir else DEBUG_AUDIO_DIR
        self.enabled = enabled
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seq_counter = 0

    def get_next_segment_id(self) -> str:
        self._seq_counter += 1
        return f"segment_{self._seq_counter:06d}"

    def save_exact_audio(self, audio: np.ndarray, sample_rate: int = 16000, segment_id: Optional[str] = None) -> str:
        """
        Guarda el array EXACTO de audio en formato WAV PCM 16-bit.
        Retorna la ruta absoluta del archivo generado.
        """
        if not self.enabled or audio is None or len(audio) == 0:
            return ""

        seg_id = segment_id or self.get_next_segment_id()
        wav_path = self.output_dir / f"{seg_id}.wav"

        # Asegurar escalado a int16 sin distorsión
        data = np.clip(audio, -1.0, 1.0)
        pcm_16 = (data * 32767.0).astype(np.int16)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_16.tobytes())

        logger.debug("Exact WAV saved: %s (samples=%d, dur=%.2fs)", wav_path, len(audio), len(audio) / sample_rate)
        return str(wav_path)

    def log_words(self, words: List[Dict[str, Any]]) -> None:
        """
        Emite en log el desglose palabra por palabra con timestamps y confianza.
        """
        if not words:
            return
        for w in words:
            word_str = w.get("word", "").strip()
            start = float(w.get("start", 0.0))
            end = float(w.get("end", 0.0))
            conf = float(w.get("confidence", w.get("probability", 1.0)))
            log_line = f"[WORD]\n{word_str}\nstart={start:.2f}s\nend={end:.2f}s\nconfidence={conf:.2f}"
            logger.info(log_line)

    def log_text_stages(
        self,
        raw_whisper: str,
        retry_whisper: str,
        reassembled: str,
        final_source: str,
        speaker_id: str = "SPEAKER_01",
    ) -> None:
        """
        Emite de forma estructurada las cuatro etapas del texto procesado.
        """
        stage_info = (
            f"\n--- [TEXT STAGES - {speaker_id}] ---\n"
            f"RAW WHISPER:  \"{raw_whisper}\"\n"
            f"RETRY:        \"{retry_whisper}\"\n"
            f"REASSEMBLED:  \"{reassembled}\"\n"
            f"FINAL SOURCE: \"{final_source}\"\n"
            f"------------------------------------"
        )
        logger.info(stage_info)

    def save_metadata(
        self,
        segment_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        capture_start: float = 0.0,
        capture_end: float = 0.0,
        vad_start: float = 0.0,
        vad_end: float = 0.0,
        pre_roll_ms: int = 300,
        post_roll_ms: int = 250,
        speaker_id: str = "SPEAKER_01",
        speaker_confidence: float = 1.0,
        raw_stt: str = "",
        retry_stt: str = "",
        reassembled_stt: str = "",
        final_stt: str = "",
        avg_logprob: float = 0.0,
        no_speech_probability: float = 0.0,
        stt_confidence: float = 1.0,
        retry_used: bool = False,
        retry_reason: str = "none",
        words: Optional[List[Dict[str, Any]]] = None,
        loss_class: str = "NONE",
    ) -> str:
        """
        Genera y persiste el archivo metadata JSON complementario al WAV.
        """
        if not self.enabled:
            return ""

        json_path = self.output_dir / f"{segment_id}.json"
        duration = len(audio) / sample_rate if audio is not None and len(audio) > 0 else 0.0

        clean_words = []
        for w in (words or []):
            clean_words.append({
                "word": w.get("word", "").strip(),
                "start": round(float(w.get("start", 0.0)), 3),
                "end": round(float(w.get("end", 0.0)), 3),
                "confidence": round(float(w.get("confidence", w.get("probability", 1.0))), 3),
            })

        metadata = {
            "segment_id": segment_id,
            "capture_start": round(capture_start, 3),
            "capture_end": round(capture_end, 3),
            "vad_start": round(vad_start, 3),
            "vad_end": round(vad_end, 3),
            "pre_roll_ms": int(pre_roll_ms),
            "post_roll_ms": int(post_roll_ms),
            "samples": len(audio) if audio is not None else 0,
            "sample_rate": sample_rate,
            "duration": round(duration, 3),
            "speaker_id": speaker_id,
            "speaker_confidence": round(speaker_confidence, 3),
            "raw_stt": raw_stt,
            "retry_stt": retry_stt,
            "reassembled_stt": reassembled_stt,
            "final_stt": final_stt,
            "avg_logprob": round(avg_logprob, 3),
            "no_speech_probability": round(no_speech_probability, 3),
            "stt_confidence": round(stt_confidence, 3),
            "retry_used": bool(retry_used),
            "retry_reason": retry_reason,
            "loss_class": loss_class,
            "words": clean_words,
            "created_at": time.time(),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Log word details si está habilitado
        self.log_words(clean_words)
        self.log_text_stages(raw_stt, retry_stt, reassembled_stt, final_stt, speaker_id=speaker_id)

        return str(json_path)


# Instancia singleton accesible para el pipeline
default_real_audio_logger = RealAudioLogger()
