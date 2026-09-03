# Traductor en Tiempo Real (100% Local y Offline)

Aplicación de escritorio para Windows diseñada para capturar la voz del micrófono, transcribirla localmente con **faster-whisper**, traducirla de forma offline y mostrar los subtítulos en una ventana flotante transparente tipo overlay.

---

## Estructura del Proyecto

```
Traductor_tiempo_real/
│
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuración central (modelos, sample rate, umbrales de VAD, GPU/CPU)
│
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── recorder.py      # Captura continua con sounddevice mediante cola no bloqueante
│   │   └── vad.py           # Detección de actividad vocal (VAD) y corte automático por silencios
│   ├── asr/
│   │   ├── __init__.py
│   │   └── whisper_engine.py# Motor faster-whisper con gestión de modelos offline y GPU (CUDA) / CPU
│   ├── translation/         # (Módulo preparado para la Etapa 2: NLLB-200 / MarianMT)
│   └── ui/                  # (Módulo preparado para la Etapa 3: Subtítulos flotantes con PySide6)
│
├── models/                  # Directorio local donde se almacenan los modelos descargados (persistentes offline)
├── main.py                  # Punto de entrada ejecutable de la Etapa 1 (consola en tiempo real)
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Documentación del proyecto
```

---

## Explicación de los Archivos de la Etapa 1

- **`config/settings.py`**: Contiene todas las constantes y configuraciones ajustables: tamaño del modelo (`tiny`, `base`, `small`), idioma de entrada, umbral de sensibilidad de voz (`ENERGY_THRESHOLD`), tiempo de silencio para considerar una frase completa (`SILENCE_DURATION_MS`), etc.
- **`src/audio/recorder.py`**: Gestiona la interacción con el micrófono mediante `sounddevice`. Captura audio en un hilo secundario sin congelar la aplicación y deposita frames pequeños (30ms) en una cola thread-safe.
- **`src/audio/vad.py`**: Detector de voz y segmentador. Evalúa la energía RMS del audio continuo, detecta cuándo una persona empieza a hablar, conserva un búfer previo para no perder la primera sílaba, y al detectar una pausa de silencio natural emite el segmento completo de audio.
- **`src/asr/whisper_engine.py`**: Wrapper de `faster-whisper` (CTranslate2). Detecta automáticamente tu GPU NVIDIA RTX 4070 (`device="cuda"`, `compute_type="float16"`) para una inferencia instantánea o recurre a CPU (`int8`) en caso necesario. Guarda los modelos en `./models` para que funcionen sin conexión.
- **`main.py`**: Orquesta el flujo completo de la Etapa 1. Lista los micrófonos, carga el modelo Whisper, escucha el micrófono y muestra en pantalla las frases transcritas con su duración y latencia en milisegundos.

---

## Requisitos e Instalación

### 1. Requisitos Previos
- Windows 10 o Windows 11.
- Python 3.10 recomendado (por compatibilidad de ruedas nativas de CTranslate2 y PyAudio/sounddevice).
- Micrófono funcional.

### 2. Creación del Entorno Virtual e Instalación de Dependencias

Ejecuta en PowerShell dentro de la carpeta del proyecto:

```powershell
# 1. Crear entorno virtual con Python 3.10
py -3.10 -m venv .venv

# 2. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# 3. Instalar las dependencias
pip install -r requirements.txt
```

---

## Ejecución de la Etapa 1

Con el entorno virtual activado, ejecuta:

```powershell
python main.py
```

### ¿Qué esperar al iniciar?
1. En la primera ejecución, `faster-whisper` descargará el modelo `base` (aproximadamente 145 MB) a la carpeta `./models/`. En las ejecuciones siguientes funcionará **100% offline**.
2. Mostrará los micrófonos disponibles en el sistema.
3. Se iniciará la escucha activa. Cuando hables al micrófono y hagas una pausa natural, verás la transcripción en consola junto al tiempo de procesamiento.
4. Para salir, presiona `Ctrl + C`.

---

## Ejecución de Pruebas (Tests)

Para verificar que cada componente (captura de audio, algoritmo VAD y motor Whisper) funcione de manera aislada o integral:

### 1. Suite Completa de Pruebas Automatizadas (Unit & Integration)
```powershell
python run_tests.py
```
Ejecuta los 9 tests unitarios y de integración de VAD, Sounddevice y WhisperEngine.

### 2. Prueba Diagnóstica de Micrófono en Vivo (5 Segundos)
```powershell
python tests\test_microphone_live.py
```
Te dará una cuenta regresiva de 3 segundos, grabará durante 5 segundos lo que digas y medirá el nivel de señal RMS y la transcripción inmediata en pantalla.

---

## Próximas Etapas

- **Etapa 2**: Integración del motor de traducción local offline (MarianMT o NLLB-200) para traducir el texto de entrada al idioma deseado (ej. Español -> Inglés).
- **Etapa 3**: Interfaz gráfica flotante (Overlay) con PySide6 (fondo semitransparente, siempre encima, texto grande, posicionamiento inferior).
- **Etapa 4**: Empaquetado final en un archivo `.exe` con PyInstaller.
