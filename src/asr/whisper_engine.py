"""
Motor de reconocimiento de voz usando faster-whisper.
Optimizado para baja latencia con CTranslate2, selección automática de GPU (CUDA) o CPU,
y guardado persistente de modelos para funcionamiento 100% offline.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import logging
import math
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from config.settings import (
    MODELS_DIR,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    BEAM_SIZE,
)
from src.asr.base import ASREngine

logger = logging.getLogger(__name__)


def _resolve_local_model(model_size: str, download_root: str) -> str:
    """Resolve an installed Hugging Face snapshot without making a hub request."""
    root = Path(download_root)
    direct = Path(model_size)
    if direct.is_dir() and (direct / "model.bin").exists():
        return str(direct)
    aliases = {
        "distil-small.en": "models--Systran--faster-distil-whisper-small.en",
        "distil-medium.en": "models--Systran--faster-distil-whisper-medium.en",
    }
    cache_name = aliases.get(model_size, f"models--Systran--faster-whisper-{model_size}")
    repository = root / cache_name
    ref = repository / "refs" / "main"
    revisions = []
    if ref.exists():
        revisions.append(ref.read_text(encoding="utf-8").strip())
    snapshots = repository / "snapshots"
    if snapshots.exists():
        revisions.extend(path.name for path in snapshots.iterdir() if path.is_dir())
    for revision in dict.fromkeys(revisions):
        candidate = snapshots / revision
        if (candidate / "model.bin").exists() and (candidate / "config.json").exists():
            return str(candidate)
    raise FileNotFoundError(
        f"El modelo Whisper '{model_size}' no está instalado localmente en {root}. "
        "El traductor no intentará descargarlo porque funciona en modo offline."
    )


def deduplicate_repetitions(text: str) -> str:
    """Elimina bucles patológicos de repetición de palabras y frases consecutivas generadas por ASR."""
    if not text:
        return ""
    # 1. Eliminar repeticiones inmediatas de una misma palabra: 'palabra palabra palabra' -> 'palabra'
    cleaned = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    # 2. Eliminar repetición consecutiva de frases (ej. 'iba a suceder cuando iba a suceder cuando')
    pattern = r'(\b(?:\w+\s+){1,6}\w+)\s+\1\b'
    for _ in range(3):
        cleaned = re.sub(pattern, r'\1', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


class WhisperEngine(ASREngine):
    """Envoltorio optimizado sobre faster-whisper para transcripción en tiempo real."""

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        download_root: Optional[str] = None,
    ):
        self.model_size = model_size
        self.download_root = download_root or str(MODELS_DIR)
        self.model = None
        self.actual_device = device
        self.actual_compute_type = compute_type

        self._load_model(device, compute_type)

    def _load_model(self, requested_device: str, requested_compute_type: str) -> None:
        """Carga el modelo faster-whisper con soporte de fallback automático a CPU."""
        from faster_whisper import WhisperModel
        local_model_path = _resolve_local_model(self.model_size, self.download_root)

        target_device = requested_device
        target_compute = requested_compute_type
        if requested_device != 'auto' and requested_compute_type == 'auto':
            target_compute = 'int8' if requested_device == 'cpu' else 'float16'

        # Si se especificó 'auto', intentar GPU primero
        if requested_device == "auto":
            try:
                import ctranslate2
                cuda_available = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                cuda_available = False

            if cuda_available:
                target_device = "cuda"
                target_compute = "float16" if requested_compute_type == "auto" else requested_compute_type
            else:
                target_device = "cpu"
                target_compute = "int8" if requested_compute_type == "auto" else requested_compute_type

        logger.info(
            "Cargando modelo faster-whisper '%s' en [%s - %s] (Directorio: %s)...",
            self.model_size,
            target_device,
            target_compute,
            self.download_root,
        )

        try:
            self.model = WhisperModel(
                local_model_path,
                device=target_device,
                compute_type=target_compute,
                download_root=self.download_root,
            )
            # Prueba de inferencia rápida para validar disponibilidad de librerías CUDA
            dummy_test = np.zeros(1600, dtype=np.float32)
            list(self.model.transcribe(dummy_test, beam_size=1)[0])

            self.actual_device = target_device
            self.actual_compute_type = target_compute
            logger.info("Modelo cargado exitosamente en %s.", target_device)

        except Exception as e:
            if target_device == "cuda":
                logger.warning(
                    "CUDA no disponible o faltan librerías nativas (%s). Ejecutando en CPU con int8...",
                    str(e),
                )
                self.model = WhisperModel(
                    local_model_path,
                    device="cpu",
                    compute_type="int8",
                    download_root=self.download_root,
                )
                self.actual_device = "cpu"
                self.actual_compute_type = "int8"
                logger.info("Modelo cargado exitosamente en CPU (int8).")
            else:
                raise e

        # Calentamiento inicial rápido (warm-up) para evitar latencia en la primera frase
        self._warmup()

    def _warmup(self) -> None:
        """Realiza una transcripción en blanco de prueba para compilar caches de ejecución."""
        try:
            dummy_audio = np.zeros(16000, dtype=np.float32)
            list(self.model.transcribe(dummy_audio, beam_size=1)[0])
        except Exception as e:
            logger.debug("Warmup finalizado: %s", e)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> str:
        return self.actual_device

    @property
    def compute_type(self) -> str:
        return self.actual_compute_type

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        beam_size: int = BEAM_SIZE,
        initial_prompt: Optional[str] = None,
        vad_filter: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe un array de audio (16kHz float32 mono).

        Retorna un diccionario con:
          - text: texto transcripto consolidado
          - language: idioma detectado o utilizado
          - probability: probabilidad del idioma detectado
          - elapsed_time: tiempo en segundos que tomó la transcripción
          - segments: lista de segmentos detallados con marcas de tiempo
        """
        if self.model is None:
            raise RuntimeError("El modelo faster-whisper no está cargado.")

        start_time = time.perf_counter()

        # Doble VAD protection: si vad_filter es explícitamente False (por segmentación externa),
        # no usamos vad_filter en Whisper para evitar corte de palabras y latencia innecesaria.
        use_vad = True if vad_filter is None else bool(vad_filter)

        # Normalización preventiva de audio tenue (micrófono físico o voz lejana)
        if len(audio) > 0:
            peak = float(np.max(np.abs(audio)))
            if 0.001 < peak < 0.25:
                gain = min(5.0, 0.60 / peak)
                audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

        # Configuración de transcripción optimizada para tiempo real y prevención de alucinaciones
        segments_gen, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            temperature=0.0,
            vad_filter=use_vad,
            condition_on_previous_text=False,  # CRÍTICO: False para evitar bucles infinitos de repetición
            initial_prompt=initial_prompt,
            repetition_penalty=1.2,  # Penaliza la repetición de palabras
            no_repeat_ngram_size=3,  # Prohíbe repetición idéntica de 3-gramas
            compression_ratio_threshold=2.4,  # Descarta alucinaciones repetitivas
            hallucination_silence_threshold=2.0,  # Suprime alucinaciones en silencio
            word_timestamps=True,
            **kwargs,
        )

        segments_list = []
        full_text_parts: List[str] = []
        all_words: List[Dict[str, Any]] = []
        avg_logprobs: List[float] = []
        no_speech_probs: List[float] = []

        for seg in segments_gen:
            clean_text = seg.text.strip()
            if clean_text:
                full_text_parts.append(clean_text)
                seg_words = []
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        w_info = {
                            "word": getattr(w, "word", ""),
                            "start": getattr(w, "start", 0.0),
                            "end": getattr(w, "end", 0.0),
                            "probability": getattr(w, "probability", 1.0),
                        }
                        seg_words.append(w_info)
                        all_words.append(w_info)

                logprob = getattr(seg, "avg_logprob", 0.0)
                no_sp = getattr(seg, "no_speech_prob", 0.0)
                avg_logprobs.append(logprob)
                no_speech_probs.append(no_sp)

                segments_list.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": clean_text,
                    "avg_logprob": logprob,
                    "no_speech_prob": no_sp,
                    "words": seg_words,
                })

        elapsed = time.perf_counter() - start_time
        full_text = " ".join(full_text_parts).strip()
        full_text = deduplicate_repetitions(full_text)

        word_conf = float(np.mean([w["probability"] for w in all_words])) if all_words else 1.0
        avg_lp = float(np.mean(avg_logprobs)) if avg_logprobs else 0.0
        # Mapeo a confianza [0.0, 1.0]: exp(avg_lp) * word_conf
        try:
            exp_lp = math.exp(avg_lp) if avg_lp <= 0.0 else 1.0
        except OverflowError:
            exp_lp = 1.0
        confidence = float(max(0.0, min(1.0, exp_lp * word_conf))) if avg_logprobs else 1.0

        return {
            "text": full_text,
            "language": info.language if info else language,
            "probability": info.language_probability if info else 1.0,
            "elapsed_time": elapsed,
            "segments": segments_list,
            "words": all_words,
            "avg_logprob": avg_lp,
            "no_speech_prob": float(np.mean(no_speech_probs)) if no_speech_probs else 0.0,
            "confidence": confidence,
        }
