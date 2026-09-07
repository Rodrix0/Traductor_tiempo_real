# Traductor Studio · Español / English / Português

La nueva aplicación se abre con **`Iniciar_traductor.cmd`** (doble clic) o con
`.\.venv\Scripts\python.exe traductor.py`.

## Subtítulos traducidos para videos y llamadas

1. Instalá las dependencias: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
2. Abrí `Iniciar_traductor.cmd` y pulsá **Preparar idiomas**. La primera vez descarga
   cuatro modelos de traducción y necesita Internet y espacio en disco.
3. Elegí **Audio PC** con los altavoces o auriculares donde escuchás el video o la
   llamada. También podés elegir un micrófono para traducir voz cercana.
4. Elegí el idioma del audio (**Automático**, español, inglés o portugués) y el idioma
   de destino. Se admiten las seis direcciones de traducción; español ↔ portugués
   usa inglés como idioma intermedio. Si origen y destino coinciden, muestra la transcripción.
5. Pulsá **Iniciar** y reproducí el contenido. Whisper puede descargar su modelo la
   primera vez. **base** prioriza velocidad; **small**, precisión.
6. La ventana flotante se abre al iniciar. Podés ajustar tamaño, opacidad y visibilidad
   del original. Se puede mover y redimensionar; usá el reproductor en ventana o pantalla
   completa sin exclusividad. Escape cierra solamente los subtítulos, no la captura.
7. Pulsá **Detener** al terminar. Para cambiar idiomas o dispositivo, detené e iniciá otra vez.

Después de preparar los modelos, el audio y la traducción se procesan localmente.
La aplicación no guarda el audio capturado. Guarda automáticamente las transcripciones
y traducciones en `data/history.sqlite3`, incluyendo resultados parciales de sesiones
canceladas. El modo de diagnóstico antiguo descrito abajo sí puede guardar WAV.

## Archivos

1. Entrá a **Archivos** y elegí un archivo de audio o video (por ejemplo WAV, MP3, MP4 o MKV).
2. Elegí idioma de origen, destino y modelo. Pulsá **Generar subtítulos**.
3. Se procesa la primera pista de audio en bloques de 30 segundos para no cargar
   películas completas en memoria. Los subtítulos conservan los tiempos del archivo.
4. Exportá **SRT**, **VTT** o **TXT**. SRT/VTT contienen la traducción; TXT incluye también el original.
5. **Cancelar** detiene el trabajo después de la operación en curso y permite exportar
   lo ya procesado. Los errores también conservan las frases guardadas en Historial.

Los subtítulos se exportan como archivos separados; no se incrustan en el video.
La decodificación usa PyAV incluido por faster-whisper; no necesita un ejecutable FFmpeg separado.

## Historial

Cada sesión guarda nombre, fecha, origen, destino, modelo, estado y frases. En **Historial**
podés buscar por nombre o contenido del original/traducción, seleccionar una sesión y
exportarla en **TXT, SRT, VTT o JSON**. JSON conserva los tiempos, idioma, original y traducción.
La lista muestra hasta 200 coincidencias; la vista previa muestra hasta 1000 frases.
La exportación incluye todas las frases de la sesión seleccionada.
En vivo, los tiempos son relativos al inicio de la captura; no al inicio de la película.

## Modelos y ajustes

Elegí los idiomas iniciales, **base** o **small**, procesamiento automático o **CPU**,
dispositivo de audio, tamaño/opacidad de los subtítulos y sensibilidad/duración del
fragmento de audio. **Guardar preferencias** aplica los valores a ambas secciones y
los conserva en `data/preferences.json` para el próximo inicio.

**Preparar modelo e idiomas** descarga/verifica el modelo elegido y las cuatro
traducciones. El catálogo muestra qué modelos están descargados y qué procesador se
usa cuando hay un motor cargado. La preparación puede tardar; Detener/Cerrar espera
la descarga o inferencia en curso antes de terminar. Los ajustes se cambian con el trabajo detenido.

## Verificación

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p 'test_*.py' -v
```

La suite cubre audio, colas, errores, subtítulos flotantes, importación por bloques,
cancelación, formatos de exportación, historial, preferencias, interfaz y reconocimiento.
Las pruebas existentes de audio y Whisper requieren dispositivo y modelo local.

Integraciones optativas con una frase sintética en `models/speech_smoke.wav`:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tests.integration_local
.\.venv\Scripts\python.exe -X utf8 -m tests.integration_app
```

La primera genera un MP4 de prueba y verifica WAV/MP4 → traducción → SRT/VTT.
La segunda prueba el trabajo desde la interfaz, historial, cuatro exportaciones y
reapertura con preferencias EN → PT / CPU. Usan archivos temporales para sus resultados.
También se verificó manualmente la captura WASAPI con una frase sintética reproducida
por los altavoces y su traducción al español. No se probaron llamadas o servicios de
streaming específicos ni audio protegido.

La captura usa [WASAPI loopback / PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch)
y la traducción usa [Argos Translate](https://argos-translate.readthedocs.io/en/stable/).
Captura todo el sonido del dispositivo seleccionado, incluidas notificaciones y
otras aplicaciones. En llamadas captura a quienes escuchás; tu micrófono se puede
seleccionar por separado, pero esta versión no mezcla ambos dispositivos.

Los subtítulos aparecen por fragmentos de hasta unos cuatro segundos más el tiempo
de reconocimiento y traducción: no son instantáneos. La música, voces superpuestas
y frases muy cortas pueden reducir la precisión. Si la computadora no alcanza a
procesar, descarta fragmentos antiguos y muestra el contador de omisiones. Elegí
el idioma de origen manualmente si la detección automática falla. Algunos contenidos
protegidos o dispositivos pueden impedir la captura del audio.

Si el entorno `.venv` no arranca, necesitás una instalación funcional de Python 3.10+
con Tkinter en Windows. Podés crear un entorno nuevo sin borrar el anterior:

```powershell
py -3.10 -m venv .venv-traductor
.\.venv-traductor\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-traductor\Scripts\python.exe traductor.py
```

## Modo anterior: reconocimiento de micrófono en consola

Esta es la **Etapa 1** del proyecto de traducción en tiempo real. Su único objetivo es escuchar el micrófono y transcribir la voz a texto de forma local y continua usando `faster-whisper`, sin APIs externas ni servicios en la nube.

Flujo:
```
MICRÓFONO ──> AUDIO (sounddevice) ──> FASTER-WHISPER ──> TEXTO EN CONSOLA
```

---

## Estructura del Proyecto

```
TraductorTiempoReal/
│
├── main.py               # Punto de entrada y bucle principal continuo
├── audio_capture.py      # Captura de micrófono con detección de fragmentos de voz (VAD)
├── speech_to_text.py     # Reconocimiento de voz local con faster-whisper (GPU/CPU)
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Esta guía de uso
```

---

## 1. Cómo Crear el Entorno Virtual

Recomendamos usar **Python 3.10** (que ya tienes instalado en tu sistema) para garantizar la máxima compatibilidad con las librerías de audio y CTranslate2 en Windows.

Abre PowerShell en la carpeta del proyecto y ejecuta:

```powershell
py -3.10 -m venv .venv
```

---

## 2. Cómo Activar el Entorno Virtual en Windows

En PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

*(Si Windows muestra un error sobre políticas de ejecución de scripts, puedes habilitarlo en tu usuario con: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*

---

## 3. Cómo Instalar las Dependencias

Con el entorno virtual activado:

```powershell
pip install -r requirements.txt
```

---

## 4. Cómo Ejecutar el Programa

Con el entorno virtual activado:

```powershell
python main.py
```

### ¿Qué verás al ejecutarlo?
1. La lista de micrófonos detectados en tu sistema (con dispositivos virtuales y altavoces excluidos automáticamente).
2. El micrófono seleccionado (por defecto configurado en `ID 1: Microphone Array (AMD Audio Device)`).
3. La carga del modelo local Whisper **`small`** en CPU con cuantización `int8`.
4. El panel de prueba guiada:
   ```
   PRUEBA:
   Di claramente:
     "Hola, esto es una prueba de reconocimiento de voz."
   ```
5. En cada frase hablada, verás la duración capturada, el nivel RMS, la ruta del archivo WAV guardado en `debug_audio/` y el texto transcripto:
   ```
   Duración capturada: 2.85 s
   Nivel RMS: 0.04312
   [Audio guardado en: debug_audio/capture_001.wav]
   Texto detectado: Hola, esto es una prueba de reconocimiento de voz.
   ```
6. Para detener el programa de forma segura, presiona **`Ctrl + C`**.

---

## 5. Diagnóstico de Audio con Archivos WAV

Para verificar si el micrófono físico está entregando audio claro y sin distorsión, cada frase detectada se guarda automáticamente en:

```
debug_audio/capture_001.wav
debug_audio/capture_002.wav
...
```

Puedes abrir y reproducir esos archivos WAV con el Reproductor de Windows para escuchar exactamente qué sonido está recibiendo Whisper.

Para desactivar el guardado de WAV una vez verificado, cambia en `main.py`:
```python
DEBUG_SAVE_AUDIO = False
```

---

## 6. Cómo Cambiar el Micrófono

En `main.py`, modifica `MIC_DEVICE_ID`:
```python
# Ejemplo para usar Microphone Array (AMD Audio Device):
MIC_DEVICE_ID = 1

# O para usar otro ID de la lista:
MIC_DEVICE_ID = 5
```
