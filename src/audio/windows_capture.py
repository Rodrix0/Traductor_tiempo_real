"""
windows_capture.py
Captura profesional de audio WASAPI en Windows con soporte para:
1. Audio del sistema (WASAPI Loopback).
2. Micrófono físico.
3. Modo combinado: Sistema + Micrófono (mezcla digital nativa sin cables virtuales).
4. Resampling continuo con StatefulResampler (sin distorsión de borde).
5. Desacoplamiento de streaming hacia AudioQueue para el pipeline asíncrono.
6. Manejo seguro de desconexiones y cambio de dispositivos.
"""

import math
import queue
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
import numpy as np

from src.audio.resampler import StatefulResampler, to_mono_16k
from src.audio.vad import VoiceActivityDetector
from src.audio.ring_buffer import AudioRingBuffer
from src.pipeline.models import AudioChunk, AudioSourceMode, VADProfile

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    audio: np.ndarray
    start: float
    end: float
    captured_at: float = field(default_factory=time.monotonic)
    end_reason: str = "silence"
    trailing_silence_ms: int = 0

    @property
    def start_time(self) -> float:
        return self.start

    @property
    def end_time(self) -> float:
        return self.end


def list_sources(include_combined: bool = True) -> Tuple[List[Tuple[int, str]], Optional[int]]:
    """
    Lista todos los dispositivos WASAPI disponibles en Windows:
    - Altavoces/Auriculares en Loopback (Audio PC).
    - Micrófonos de entrada.
    - Opción combinada (Audio PC + Micrófono).
    """
    import pyaudiowpatch as pa
    with pa.PyAudio() as audio:
        api = audio.get_host_api_info_by_type(pa.paWASAPI)
        result: List[Tuple[int, str]] = []
        loopback_index: Optional[int] = None
        mic_index: Optional[int] = None

        for device in audio.get_device_info_generator():
            if device["hostApi"] == api["index"] and device["maxInputChannels"] > 0:
                is_loopback = device.get("isLoopbackDevice", False)
                kind = "Audio PC" if is_loopback else "Micrófono"
                dev_idx = int(device["index"])
                result.append((dev_idx, f"{kind} · {device['name']}"))
                if is_loopback and loopback_index is None:
                    loopback_index = dev_idx
                elif not is_loopback and mic_index is None:
                    mic_index = dev_idx

        # Intentar obtener el loopback predeterminado del sistema
        try:
            default = int(audio.get_default_wasapi_loopback()["index"])
        except (OSError, LookupError):
            default = loopback_index if loopback_index is not None else (result[0][0] if result else None)

        # Si se solicita modo combinado y existen ambos dispositivos
        if include_combined and loopback_index is not None and mic_index is not None:
            result.append((-999, "Sistema + Micrófono (Mezcla en vivo)"))

        return result, default


class WindowsCapture:
    """Captura continua de audio WASAPI con soporte loopback, mic, combinado y pipeline asíncrono."""

    def __init__(
        self,
        device: int,
        threshold: float = 0.008,
        chunk_seconds: float = 8.0,
        buffer_seconds: float = 180.0,
        source_mode: Union[AudioSourceMode, str] = AudioSourceMode.SYSTEM,
        audio_queue: Optional[queue.Queue] = None,
        vad_profile: Union[VADProfile, str] = VADProfile.NATURAL,
        silence_duration_ms: Optional[int] = None,
    ):
        self.device = device
        self.threshold = threshold
        self.chunk_seconds = chunk_seconds
        self.buffer_seconds = buffer_seconds
        self.source_mode = AudioSourceMode(source_mode) if isinstance(source_mode, str) else source_mode
        self.audio_queue = audio_queue
        self.vad_profile = VADProfile(vad_profile) if isinstance(vad_profile, str) else vad_profile
        self.silence_duration_ms = silence_duration_ms
        self.ring_buffer = AudioRingBuffer(capacity_seconds=5.0, sample_rate=16000)

        # Cola de segmentos de voz procesados (para compatibilidad hacia atrás)
        self.segments = queue.Queue()
        self.stopped = threading.Event()
        self.error: Optional[Exception] = None
        self.level: float = 0.0
        self.dropped: int = 0
        self.thread: Optional[threading.Thread] = None
        self.ready = threading.Event()
        self._vad = None

    @property
    def acoustic_progress(self):
        return self._vad.acoustic_progress if self._vad is not None else None

    def start(self):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("La captura ya está activa.")
        self.stopped.clear()
        self.ready.clear()
        self.error = None
        self.thread = threading.Thread(target=self._record, daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            self.stopped.set()
            raise RuntimeError("El dispositivo de audio no respondió. Actualizá la lista y volvé a intentar.")
        if self.error:
            raise RuntimeError(f"No se pudo abrir el dispositivo: {self.error}") from self.error

    def stream_to_queue(self, target_queue: queue.Queue, stop_event: Optional[threading.Event] = None):
        """Run capture as a raw continuous source for PipelineController."""
        self.audio_queue = target_queue
        self.start()

    def enqueue(self, segment: AudioSegment):
        """Encola un segmento de audio asegurando control de desbordamiento de memoria."""
        with self.segments.mutex:
            duration = sum(len(s.audio) / 16000 for s in self.segments.queue)
        if duration + len(segment.audio) / 16000 > self.buffer_seconds:
            self.error = RuntimeError(
                f"Se llenó el búfer de audio ({self.buffer_seconds:g} segundos). "
                f"La captura se detuvo; no se descartaron las frases anteriores."
            )
            self.stopped.set()
            return
        self.segments.put_nowait(segment)

    @property
    def pending_seconds(self) -> float:
        with self.segments.mutex:
            return sum(len(s.audio) / 16000 for s in self.segments.queue)

    def _record(self):
        try:
            import pyaudiowpatch as pa
            with pa.PyAudio() as audio:
                # Caso: Modo Combinado (-999 o BOTH)
                if self.device == -999 or self.source_mode == AudioSourceMode.BOTH:
                    self._record_combined(audio, pa)
                    return

                # Caso estándar: Dispositivo individual (Loopback o Micrófono)
                device_info = audio.get_device_info_by_index(self.device)
                is_loopback = bool(device_info.get("isLoopbackDevice", False))
                rate = int(device_info["defaultSampleRate"])
                channels = int(device_info["maxInputChannels"])
                frames = round(rate * 0.03)

                resampler = StatefulResampler(in_rate=rate, out_rate=16000, channels=channels)

                # Para micrófono físico, un silencio de 450-500 ms es ideal para no acumular ruido de sala
                if not is_loopback and self.silence_duration_ms is None and self.vad_profile == VADProfile.NATURAL:
                    silence_ms = 450
                else:
                    silence_ms = self.silence_duration_ms or self.vad_profile.silence_duration_ms

                vad = VoiceActivityDetector(
                    sample_rate=16000,
                    energy_threshold=self.threshold,
                    silence_duration_ms=silence_ms,
                    min_speech_duration_ms=180,
                    max_speech_duration_ms=int(self.chunk_seconds * 1000),
                    pre_speech_padding_ms=300,
                )

                self._vad = vad
                stream = audio.open(
                    format=pa.paFloat32,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=self.device,
                    frames_per_buffer=frames,
                )
                self.ready.set()
                elapsed_samples = 0

                try:
                    while not self.stopped.is_set():
                        try:
                            raw = stream.read(frames, exception_on_overflow=False)
                        except OSError as e:
                            logger.warning("Error de lectura en dispositivo WASAPI: %s", e)
                            self.error = e
                            break

                        frame_16k = resampler.resample(raw)
                        if len(frame_16k) == 0:
                            continue

                        # Si es micrófono, remover DC offset para estabilizar el VAD y la energía
                        if not is_loopback:
                            frame_16k = frame_16k - np.mean(frame_16k)

                        self.ring_buffer.write(frame_16k)
                        elapsed_samples += len(frame_16k)
                        self.level = vad.calculate_energy(frame_16k)

                        # Enviar al pipeline asíncrono si está activo
                        if self.audio_queue is not None:
                            try:
                                chunk = AudioChunk(
                                    data=frame_16k,
                                    timestamp=time.monotonic(),
                                    sample_rate=16000,
                                    source_mode=self.source_mode,
                                )
                                self.audio_queue.put_nowait(chunk)
                            except queue.Full:
                                self.dropped += 1
                            continue

                        # Procesamiento VAD integrado para cola síncrona
                        segment = vad.process_frame(frame_16k)
                        if segment is not None:
                            # Para micrófono, normalización suave de ganancia si la señal fue muy tenue (< 0.35 pico)
                            if not is_loopback and len(segment) > 0:
                                peak = float(np.max(np.abs(segment)))
                                if 0.001 < peak < 0.35:
                                    gain = min(6.0, 0.65 / peak)
                                    segment = np.clip(segment * gain, -1.0, 1.0).astype(np.float32)

                            end_s = (elapsed_samples - vad.speech_samples_count) / 16000.0
                            start_s = max(0.0, end_s - len(segment) / 16000.0)
                            self.enqueue(AudioSegment(segment, start_s, end_s, end_reason=vad.last_finalize_reason or "silence", trailing_silence_ms=vad.last_trailing_silence_ms))

                finally:
                    stream.stop_stream()
                    stream.close()

        except Exception as exc:
            self.error = exc
        finally:
            self.ready.set()

    def _record_combined(self, audio, pa):
        """Captura y mezcla en tiempo real el audio del sistema y del micrófono."""
        api = audio.get_host_api_info_by_type(pa.paWASAPI)
        loopback_dev = None
        mic_dev = None

        for dev in audio.get_device_info_generator():
            if dev["hostApi"] == api["index"] and dev["maxInputChannels"] > 0:
                if dev.get("isLoopbackDevice") and loopback_dev is None:
                    loopback_dev = dev
                elif not dev.get("isLoopbackDevice") and mic_dev is None:
                    mic_dev = dev

        if loopback_dev is None or mic_dev is None:
            raise RuntimeError("Se requieren tanto altavoces como micrófono para el modo combinado.")

        sys_rate = int(loopback_dev["defaultSampleRate"])
        sys_ch = int(loopback_dev["maxInputChannels"])
        sys_frames = round(sys_rate * 0.03)

        mic_rate = int(mic_dev["defaultSampleRate"])
        mic_ch = int(mic_dev["maxInputChannels"])
        mic_frames = round(mic_rate * 0.03)

        sys_resampler = StatefulResampler(in_rate=sys_rate, out_rate=16000, channels=sys_ch)
        mic_resampler = StatefulResampler(in_rate=mic_rate, out_rate=16000, channels=mic_ch)

        sys_stream = audio.open(
            format=pa.paFloat32,
            channels=sys_ch,
            rate=sys_rate,
            input=True,
            input_device_index=int(loopback_dev["index"]),
            frames_per_buffer=sys_frames,
        )
        mic_stream = audio.open(
            format=pa.paFloat32,
            channels=mic_ch,
            rate=mic_rate,
            input=True,
            input_device_index=int(mic_dev["index"]),
            frames_per_buffer=mic_frames,
        )

        silence_ms = self.silence_duration_ms or self.vad_profile.silence_duration_ms
        vad = VoiceActivityDetector(
            sample_rate=16000,
            energy_threshold=self.threshold,
            silence_duration_ms=silence_ms,
            min_speech_duration_ms=180,
            max_speech_duration_ms=int(self.chunk_seconds * 1000),
            pre_speech_padding_ms=300,
        )

        self._vad = vad
        self.ready.set()
        elapsed_samples = 0

        try:
            while not self.stopped.is_set():
                raw_sys = sys_stream.read(sys_frames, exception_on_overflow=False)
                raw_mic = mic_stream.read(mic_frames, exception_on_overflow=False)

                frame_sys = sys_resampler.resample(raw_sys)
                frame_mic = mic_resampler.resample(raw_mic)

                # Alinear longitudes si hay una discrepancia de 1 muestra por redondeo
                min_len = min(len(frame_sys), len(frame_mic))
                if min_len == 0:
                    continue

                # Mezcla digital ponderada: sistema 100% + micrófono 90% (evita saturación)
                mixed_frame = np.clip(frame_sys[:min_len] + frame_mic[:min_len] * 0.9, -1.0, 1.0).astype(np.float32)

                self.ring_buffer.write(mixed_frame)
                elapsed_samples += min_len
                self.level = vad.calculate_energy(mixed_frame)

                if self.audio_queue is not None:
                    try:
                        chunk = AudioChunk(
                            data=mixed_frame,
                            timestamp=time.monotonic(),
                            sample_rate=16000,
                            source_mode=AudioSourceMode.BOTH,
                        )
                        self.audio_queue.put_nowait(chunk)
                    except queue.Full:
                        self.dropped += 1
                    continue

                segment = vad.process_frame(mixed_frame)
                if segment is not None:
                    end_s = (elapsed_samples - vad.speech_samples_count) / 16000.0
                    start_s = max(0.0, end_s - len(segment) / 16000.0)
                    self.enqueue(AudioSegment(segment, start_s, end_s, end_reason=vad.last_finalize_reason or "silence", trailing_silence_ms=vad.last_trailing_silence_ms))

        finally:
            sys_stream.stop_stream()
            sys_stream.close()
            mic_stream.stop_stream()
            mic_stream.close()

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=2)


WASAPICapture = WindowsCapture
