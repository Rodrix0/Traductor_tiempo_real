"""
audio_capture.py
Captura continua desde el micrófono con VAD calibrado para baja latencia en tiempo real.
Corta frases tan pronto hay una pausa natural (~400 ms) y evita acumular retraso.
"""

import os
import wave
import queue
import logging
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
import math

logger = logging.getLogger(__name__)

EXCLUDED_DEVICE_KEYWORDS = [
    "altavoz",
    "speaker",
    "mezcla estéreo",
    "stereo mix",
    "output",
    "salida",
    "line out",
    "line-out",
    "mapper - output",
]


class AudioCapture:
    """Captura audio del micrófono y detecta fin de oraciones rápidamente sin acumular retraso."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        target_sample_rate: int = 16000,
        energy_threshold: float = 0.012,
        pre_buffer_ms: int = 250,
        silence_duration_ms: int = 400,       # 400ms: corta la frase inmediatamente tras una pausa natural
        min_speech_duration_ms: int = 400,    # 400ms: descarta carraspeos y chasquidos
        max_speech_duration_ms: int = 4000,   # 4.0s máximo: nunca acumula más de 4s de habla continua
    ):
        self.target_sample_rate = target_sample_rate
        self.energy_threshold = energy_threshold
        self.pre_buffer_ms = pre_buffer_ms
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_ms = max_speech_duration_ms

        # Resolver información del dispositivo seleccionado
        self.device_info = self.resolve_microphone(device_index)
        self.device_id = self.device_info["id"]
        self.device_name = self.device_info["name"]
        self.native_sample_rate = int(self.device_info["default_samplerate"])
        self.input_channels = min(2, max(1, self.device_info.get("channels", 1)))

        # Factor de remuestreo (simplificación de fracciones nativa -> 16000)
        gcd = math.gcd(self.target_sample_rate, self.native_sample_rate)
        self.resample_up = self.target_sample_rate // gcd
        self.resample_down = self.native_sample_rate // gcd

        # Tamaño de bloque de lectura: ~30 ms a la frecuencia nativa
        self.frame_duration_ms = 30.0
        self.native_blocksize = int(self.native_sample_rate * (self.frame_duration_ms / 1000.0))

        # Cola thread-safe para fragmentos de voz listos
        # (audio_16k, duracion_seg, nivel_rms)
        self.speech_queue: queue.Queue[Tuple[np.ndarray, float, float]] = queue.Queue()
        self.capture_error = None

        # Búfer circular de pre-voz en frames nativos (250 ms)
        pre_frames_count = max(1, int(self.pre_buffer_ms / self.frame_duration_ms))
        self.pre_buffer: deque = deque(maxlen=pre_frames_count)

        self._stream: Optional[sd.InputStream] = None
        self._is_recording = False

        # Estado del VAD
        self._is_speaking = False
        self._current_frames: List[np.ndarray] = []
        self._silence_frames = 0
        self._total_native_samples = 0

    @staticmethod
    def list_microphones() -> List[Dict[str, Any]]:
        """Devuelve la lista filtrada de micrófonos reales disponibles."""
        try:
            devices = sd.query_devices()
            default_in = sd.default.device[0]
            mics = []
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) <= 0:
                    continue
                name = dev.get("name", "")
                name_lower = name.lower()

                if any(kw in name_lower for kw in EXCLUDED_DEVICE_KEYWORDS):
                    continue

                mics.append({
                    "id": idx,
                    "name": name,
                    "channels": dev.get("max_input_channels", 1),
                    "default_samplerate": dev.get("default_samplerate", 44100.0),
                    "is_default": (idx == default_in),
                })
            return mics
        except Exception as e:
            logger.error("Error al listar dispositivos: %s", e)
            return []

    @classmethod
    def resolve_microphone(cls, requested_id: Optional[int]) -> Dict[str, Any]:
        """Resuelve el ID y nombre real del micrófono a usar."""
        all_devices = sd.query_devices()

        if requested_id is not None:
            if 0 <= requested_id < len(all_devices):
                dev = all_devices[requested_id]
                if dev.get("max_input_channels", 0) > 0:
                    return {
                        "id": requested_id,
                        "name": dev.get("name", f"Dispositivo {requested_id}"),
                        "channels": dev.get("max_input_channels", 1),
                        "default_samplerate": dev.get("default_samplerate", 44100.0),
                    }
                else:
                    raise ValueError(f"El dispositivo ID {requested_id} ('{dev.get('name')}') no tiene canales de entrada.")
            else:
                raise ValueError(f"ID de micrófono inválido: {requested_id}. El sistema tiene {len(all_devices)} dispositivos.")

        default_idx = sd.default.device[0]
        if default_idx >= 0 and default_idx < len(all_devices):
            dev = all_devices[default_idx]
            return {
                "id": default_idx,
                "name": dev.get("name", "Predeterminado"),
                "channels": dev.get("max_input_channels", 1),
                "default_samplerate": dev.get("default_samplerate", 44100.0),
            }

        valid_mics = cls.list_microphones()
        if valid_mics:
            return valid_mics[0]

        raise RuntimeError("No se encontró ningún micrófono de entrada disponible en el sistema.")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback ultraligero ejecutado por sounddevice a frecuencia nativa."""
        # 1. Promediar canales estéreo a mono 1D
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono_native = np.mean(indata, axis=1)
        else:
            mono_native = indata.flatten()

        # 2. Calcular nivel RMS en el bloque nativo
        rms = float(np.sqrt(np.mean(mono_native**2))) if len(mono_native) > 0 else 0.0
        is_speech = rms >= self.energy_threshold

        # 3. Máquina de estados de detección de frases
        if not self._is_speaking:
            if is_speech:
                # Comienza a hablar: rescatar el pre-búfer (250 ms)
                self._is_speaking = True
                self._current_frames = list(self.pre_buffer)
                self._current_frames.append(mono_native)
                self._total_native_samples = sum(len(f) for f in self._current_frames)
                self._silence_frames = 0
                self.pre_buffer.clear()
            else:
                self.pre_buffer.append(mono_native)
        else:
            self._current_frames.append(mono_native)
            self._total_native_samples += len(mono_native)

            if is_speech:
                self._silence_frames = 0
            else:
                self._silence_frames += 1

            silence_ms = self._silence_frames * self.frame_duration_ms
            speech_ms = (self._total_native_samples / self.native_sample_rate) * 1000.0

            # Condición de corte: pausa natural (400ms) o techo de duración (4s)
            if silence_ms >= self.silence_duration_ms or speech_ms >= self.max_speech_duration_ms:
                if speech_ms >= self.min_speech_duration_ms and self._current_frames:
                    raw_segment = np.concatenate(self._current_frames)

                    # Remuestrear el segmento consolidado a 16000 Hz fuera del ciclo frame-a-frame
                    if self.native_sample_rate != self.target_sample_rate:
                        segment_16k = resample_poly(
                            raw_segment, self.resample_up, self.resample_down
                        ).astype(np.float32)
                    else:
                        segment_16k = raw_segment.astype(np.float32)

                    seg_rms = float(np.sqrt(np.mean(segment_16k**2)))
                    duration_sec = len(segment_16k) / self.target_sample_rate

                    # Normalización suave si el volumen es bajo
                    peak = float(np.max(np.abs(segment_16k)))
                    if peak > 0.005 and peak < 0.7:
                        gain = min(0.7 / peak, 4.0)
                        segment_16k = segment_16k * gain

                    with self.speech_queue.mutex:
                        pending = sum(item[1] for item in self.speech_queue.queue)
                    if pending + duration_sec > 180:
                        self.capture_error = RuntimeError('Búfer lleno (3 minutos). Captura detenida; se conservan las frases anteriores.')
                        raise sd.CallbackStop
                    self.speech_queue.put((segment_16k, duration_sec, seg_rms))

                # Resetear estado
                self._is_speaking = False
                self._current_frames = []
                self._silence_frames = 0
                self._total_native_samples = 0
                self.pre_buffer.clear()

    def start(self) -> None:
        """Inicia la captura desde el micrófono."""
        if self._is_recording:
            return

        try:
            self._is_recording = True
            self._stream = sd.InputStream(
                samplerate=self.native_sample_rate,
                channels=self.input_channels,
                dtype="float32",
                blocksize=self.native_blocksize,
                device=self.device_id,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self._is_recording = False
            raise RuntimeError(
                f"Error al iniciar captura en [ID {self.device_id}] '{self.device_name}': {e}"
            ) from e

    def stop(self) -> None:
        """Detiene y libera el micrófono."""
        self._is_recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None

    def get_speech_segment(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float, float]]:
        """Retorna el siguiente fragmento de voz listo (audio_16k, duracion_seg, rms)."""
        try:
            return self.speech_queue.get(timeout=timeout)
        except queue.Empty:
            if self.capture_error:
                raise self.capture_error
            return None

    @staticmethod
    def save_wav(audio_16k: np.ndarray, file_path: str, sample_rate: int = 16000) -> None:
        """Guarda un array float32 de audio como archivo WAV PCM de 16 bits."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(audio_16k, -1.0, 1.0)
        pcm16 = (clipped * 32767).astype(np.int16)

        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

    @property
    def is_recording(self) -> bool:
        return self._is_recording
