# Traductor_tiempo_real

## Objetivo
Aplicación Windows 100% local/offline para capturar audio del sistema
y micrófono, transcribirlo, traducirlo y mostrar subtítulos en tiempo real.

## Arquitectura principal
- audio/: captura WASAPI, micrófono y procesamiento
- VAD: detección de voz y silencios
- faster-whisper: speech-to-text
- traducción local
- overlay: subtítulos flotantes
- exportación: SRT/VTT/TXT

## Reglas importantes
- No usar APIs de pago ni servicios cloud.
- Todo debe funcionar localmente.
- No eliminar funcionalidades existentes.
- Mantener captura WASAPI y micrófono.
- Mantener soporte GPU CUDA y fallback CPU.
- No hacer commits ni push sin autorización.
- Antes de modificar código, revisar los archivos relacionados.
- Ejecutar las pruebas relevantes después de los cambios.
- Priorizar baja latencia sin sacrificar frases completas.

## Objetivo actual
La transcripción debe:
1. Escuchar la oración completa antes de cerrarla.
2. Evitar perder palabras al principio o al final.
3. Separar correctamente hablantes.
4. Traducir únicamente cuando la oración esté completa.
5. Continuar inmediatamente con la siguiente oración.
6. Conservar el máximo contenido posible del audio original.

## Flujo esperado
audio -> resampling -> VAD -> segmentación ->
speech-to-text -> separación de hablantes ->
frase completa -> traducción -> subtítulos

## Antes de cambiar algo
- Analizar la implementación existente.
- Reutilizar componentes actuales siempre que sea posible.
- Evitar reescrituras innecesarias.
- Explicar qué archivos se modificarán.