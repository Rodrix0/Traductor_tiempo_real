"""WASAPI loopback/microphone capture, resampling and bounded speech queue."""
import math
import queue
import threading
from dataclasses import dataclass
import numpy as np
from scipy.signal import resample_poly
from src.audio.vad import VoiceActivityDetector


@dataclass
class AudioSegment:
    audio: np.ndarray
    start: float
    end: float


def list_sources():
    import pyaudiowpatch as pa
    with pa.PyAudio() as audio:
        api = audio.get_host_api_info_by_type(pa.paWASAPI)
        result = []
        for device in audio.get_device_info_generator():
            if device["hostApi"] == api["index"] and device["maxInputChannels"] > 0:
                kind = "Audio PC" if device.get("isLoopbackDevice") else "Micrófono"
                result.append((int(device["index"]), f"{kind} · {device['name']}"))
        try:
            default = int(audio.get_default_wasapi_loopback()["index"])
        except (OSError, LookupError):
            default = result[0][0] if result else None
        return result, default


def to_mono_16k(raw, channels, rate):
    samples = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).mean(axis=1)
    divisor = math.gcd(rate, 16000)
    return resample_poly(samples, 16000 // divisor, rate // divisor).astype(np.float32)


class WindowsCapture:
    def __init__(self, device, threshold=0.008, chunk_seconds=4):
        self.device = device
        self.threshold = threshold
        self.chunk_seconds = chunk_seconds
        self.segments = queue.Queue(maxsize=3)
        self.stopped = threading.Event()
        self.error = None
        self.level = 0.0
        self.dropped = 0
        self.thread = None
        self.ready = threading.Event()

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

    def enqueue(self, segment):
        try:
            self.segments.put_nowait(segment)
        except queue.Full:
            try:
                self.segments.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            self.segments.put_nowait(segment)

    def _record(self):
        try:
            import pyaudiowpatch as pa
            with pa.PyAudio() as audio:
                device = audio.get_device_info_by_index(self.device)
                rate = int(device["defaultSampleRate"])
                channels = int(device["maxInputChannels"])
                frames = round(rate * 0.03)
                vad = VoiceActivityDetector(energy_threshold=self.threshold,
                    silence_duration_ms=450, min_speech_duration_ms=350,
                    max_speech_duration_ms=int(self.chunk_seconds * 1000))
                stream = audio.open(format=pa.paFloat32, channels=channels,
                    rate=rate, input=True, input_device_index=self.device,
                    frames_per_buffer=frames)
                self.ready.set()
                elapsed = 0
                try:
                    while not self.stopped.is_set():
                        raw = stream.read(frames, exception_on_overflow=True)
                        frame = to_mono_16k(raw, channels, rate)
                        elapsed += len(frame)
                        self.level = vad.calculate_energy(frame)
                        segment = vad.process_frame(frame)
                        if segment is not None:
                            self.enqueue(AudioSegment(segment, max(0, (elapsed - len(segment)) / 16000), elapsed / 16000))
                finally:
                    stream.close()
        except Exception as exc:
            self.error = exc
        finally:
            self.ready.set()

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=2)
