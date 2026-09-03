# Reconocimiento de Voz en Tiempo Real (100% Local y Offline)

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
1. La lista de micrófonos detectados en tu sistema con su respectivo ID.
2. La carga del modelo local Whisper (en la primera ejecución descargará el modelo a la carpeta local `./models/` y luego funcionará **100% offline**).
3. El programa empezará a escuchar continuamente.
4. Cada vez que hables una frase y hagas una pausa natural, verás en pantalla:
   ```
   Texto detectado: Hola, esto es una prueba.
   ```
5. Para detener el programa de forma segura, presiona **`Ctrl + C`**.

---

## 5. Cómo Cambiar el Micrófono

Al iniciar el programa por primera vez con `python main.py`, verás en la consola la lista de micrófonos disponibles, por ejemplo:

```
Micrófonos de entrada disponibles:
  [ID 1] Micrófono (Realtek High Definition Audio) (Predeterminado)
  [ID 3] Micrófono (HyperX SoloCast)
```

Si deseas usar un micrófono específico en lugar del predeterminado de Windows:
1. Abre el archivo **`main.py`**.
2. En las primeras líneas, cambia `MIC_DEVICE_ID`:
   ```python
   # Cambia None por el ID de tu micrófono deseado:
   MIC_DEVICE_ID = 3
   ```
3. Guarda el archivo y vuelve a ejecutar `python main.py`.

---

## Selección de Modelo Whisper

En `main.py` puedes ajustar la variable `MODEL_SIZE`:
- `"base"` *(predeterminado)*: Muy rápido, liviano (~145 MB), ideal para baja latencia.
- `"small"`: Mayor precisión (~460 MB), recomendado si necesitas mejor reconocimiento de términos técnicos.
