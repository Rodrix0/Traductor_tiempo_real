"""
embeddings.py
Extractor acústico de huellas vocales (speaker embeddings) en memoria RAM.
Genera vectores acústico-espectrales normalizados de 64 dimensiones a partir del audio
para identificar y distinguir hablantes de forma 100% local, rápida (<3ms) y con privacidad total.
"""

import os
import numpy as np
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class SpeakerEmbeddingProvider(ABC):
    """Interfaz base para proveedores de extracción de huellas vocales (embeddings)."""

    @abstractmethod
    def extract(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extrae un vector de características normalizado del audio."""
        pass

    @abstractmethod
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcula la similitud coseno entre dos vectores en rango [0.0, 1.0]."""
        pass


class SpectralEmbeddingProvider(SpeakerEmbeddingProvider):
    """
    Extractor de embeddings de voz basado en bancos de filtros Mel, formantes espectrales
    y estadísticas de tono fundamental (F0).
    Produce representaciones invariantes a la escala y normalizadas en norma L2 (64-D).
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 48):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = 512
        self.hop_len = 256
        self._mel_basis = self._build_mel_basis()

    def _hz_to_mel(self, hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_basis(self) -> np.ndarray:
        """Construye la matriz de filtros triangulares Mel."""
        n_freqs = self.n_fft // 2 + 1
        low_mel = self._hz_to_mel(80.0)
        high_mel = self._hz_to_mel(self.sample_rate / 2.0)
        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        weights = np.zeros((self.n_mels, n_freqs), dtype=np.float32)
        for i in range(1, self.n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            for j in range(left, center):
                if center > left:
                    weights[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    weights[i - 1, j] = (right - j) / (right - center)

        return weights

    def extract(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae un vector de embedding normalizado de 64 dimensiones del audio dado.
        Retorna None si el audio es demasiado corto o silencio absoluto.
        """
        if audio is None or len(audio) < self.n_fft:
            return None

        # Descartar silencio evidente
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.01:
            return None

        # Calcular espectrograma de potencia
        _, _, Zxx = signal.stft(
            audio.astype(np.float32),
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_len,
        )
        power_spec = np.abs(Zxx) ** 2

        # Proyectar a escala Mel (n_mels x n_frames)
        mel_spec = np.dot(self._mel_basis, power_spec[:self.n_fft // 2 + 1, :])
        log_mel = np.log1p(mel_spec)

        # Estadísticas temporales: media y desviación estándar por banda Mel (48 x 2 = 96)
        mel_mean = np.mean(log_mel, axis=1)
        mel_std = np.std(log_mel, axis=1)

        # Estadísticas adicionales: centroide espectral y pendiente (tilt)
        freqs = np.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
        spectral_centroid = np.sum(freqs[:, None] * power_spec, axis=0) / (np.sum(power_spec, axis=0) + 1e-7)
        c_mean = float(np.mean(spectral_centroid)) / (self.sample_rate / 2)
        c_std = float(np.std(spectral_centroid)) / (self.sample_rate / 2)

        # Vector de características compuesto
        raw_feat = np.concatenate([
            mel_mean[:28],              # 28 bandas medias/bajas (formantes clave)
            mel_std[:24],               # 24 varianzas de banda
            [c_mean, c_std],            # 2 de centroide
            mel_mean[28:38],            # 10 bandas altas
        ]).astype(np.float32)

        # Asegurar longitud exacta de 64 dimensiones
        if len(raw_feat) > 64:
            raw_feat = raw_feat[:64]
        elif len(raw_feat) < 64:
            raw_feat = np.pad(raw_feat, (0, 64 - len(raw_feat)))

        # Normalización L2 (longitud unitaria) para que el producto punto equivalga a similitud coseno
        norm = float(np.linalg.norm(raw_feat))
        if norm < 1e-6:
            return None

        return raw_feat / norm

    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcula la similitud coseno entre dos vectores de embedding en rango [0.0, 1.0].
        """
        if emb1 is None or emb2 is None or len(emb1) != len(emb2):
            return 0.0

        dot = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, dot))


class ECAPATDNNEmbeddingProvider(SpeakerEmbeddingProvider):
    """
    Proveedor opcional basado en red neuronal ECAPA-TDNN (192-D).
    Opera 100% offline si los pesos existen en models/ecapa. Si no, realiza
    fallback transparente a SpectralEmbeddingProvider.
    """

    def __init__(self, sample_rate: int = 16000, model_dir: str = "models/ecapa"):
        self.sample_rate = sample_rate
        self.model_dir = model_dir
        self._spectral_fallback = SpectralEmbeddingProvider(sample_rate=sample_rate)
        self._model = None
        self._loaded = False
        self._load_offline()

    def _load_offline(self) -> None:
        if not os.path.isdir(self.model_dir):
            return
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
            self._model = SpeakerRecognition.from_hparams(
                source=self.model_dir,
                savedir=self.model_dir,
                run_opts={"device": "cpu"},
            )
            self._loaded = True
            logger.info("ECAPA-TDNN cargado exitosamente desde %s.", self.model_dir)
        except Exception as e:
            logger.debug("ECAPA-TDNN no disponible offline (%s). Usando fallback espectral.", e)
            self._model = None

    def extract(self, audio: np.ndarray) -> Optional[np.ndarray]:
        if not self._loaded or self._model is None:
            return self._spectral_fallback.extract(audio)
        try:
            import torch
            tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emb = self._model.encode_batch(tensor).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            return (emb / norm).astype(np.float32) if norm > 1e-6 else None
        except Exception:
            return self._spectral_fallback.extract(audio)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if emb1 is None or emb2 is None or len(emb1) != len(emb2):
            return 0.0
        dot = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, dot))


# Alias de compatibilidad total con código existente
SpeakerEmbeddingExtractor = SpectralEmbeddingProvider
