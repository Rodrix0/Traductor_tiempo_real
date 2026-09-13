"""
Configuración centralizada para la aplicación de traducción en tiempo real.
Todos los parámetros ajustables de audio, modelos y ejecución se definen aquí.
"""

from pathlib import Path
import os
import sys

# Deshabilitar advertencia de symlinks en Windows para HuggingFace Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Si se ejecuta como ejecutable empaquetado (PyInstaller) o desde script
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).parent
else:
    APP_DIR = BASE_DIR

MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
# Configuración de Audio
# -------------------------------------------------------------
SAMPLE_RATE: int = 16000  # Frecuencia nativa esperada por Whisper
CHANNELS: int = 1        # Mono
DTYPE: str = "float32"   # Formato normalizado [-1.0, 1.0]

# Intervalo de captura de cada frame en milisegundos (ej. 30ms = 480 muestras a 16kHz)
FRAME_DURATION_MS: int = 30
FRAME_SIZE: int = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0))

# -------------------------------------------------------------
# Configuración de Detección de Actividad de Voz (VAD)
# -------------------------------------------------------------
# Umbral de energía RMS para clasificar un frame como voz
# Rango típico: 0.005 a 0.03 según la sensibilidad del micrófono
ENERGY_THRESHOLD: float = 0.015

# Milisegundos consecutivos de silencio para considerar que una frase terminó
SILENCE_DURATION_MS: int = 700

# Duración mínima de audio para enviar a transcribir (180ms permite capturar respuestas cortas: "Yeah", "Right")
MIN_SPEECH_DURATION_MS: int = 180

# Duración máxima de un bloque de audio antes de forzar una transcripción parcial
MAX_SPEECH_DURATION_MS: int = 12000

# Milisegundos de audio previo (pre-roll buffer) para no cortar la primera sílaba ("I've done", "Well")
PRE_SPEECH_PADDING_MS: int = 300

# -------------------------------------------------------------
# Configuración de Reconocimiento de Voz (faster-whisper)
# -------------------------------------------------------------
# Tamaños disponibles: "tiny", "base", "small", "medium", "large-v3"
WHISPER_MODEL_SIZE: str = "base"

# Dispositivo: "auto", "cuda", "cpu"
WHISPER_DEVICE: str = "auto"

# Tipo de cálculo: "auto", "float16", "int8_float16", "int8"
# En GPU (CUDA), "float16" es óptimo. En CPU, "int8" es el más rápido.
WHISPER_COMPUTE_TYPE: str = "auto"

# Idioma de entrada: "es", "en", etc. O None para autodetección continua.
INPUT_LANGUAGE: str = "es"

# Beam size (1 es más rápido para tiempo real, 5 es más preciso)
BEAM_SIZE: int = 1

# -------------------------------------------------------------
# Configuración de Diagnóstico y Benchmark Real
# -------------------------------------------------------------
REAL_AUDIO_DEBUG: bool = os.environ.get("REAL_AUDIO_DEBUG", "1").lower() in ("1", "true", "yes")
DEBUG_AUDIO_DIR: Path = APP_DIR / "debug_audio"
DEBUG_AUDIO_DIR.mkdir(exist_ok=True)

