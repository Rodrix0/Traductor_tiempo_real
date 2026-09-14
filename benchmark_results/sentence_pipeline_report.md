# Informe del pipeline de oraciones confirmadas

## Estado

La ruta en vivo iniciada por `Iniciar_traductor.cmd` usa ahora un único pipeline asíncrono que sólo entrega `ConfirmedUtterance` al traductor. La suite completa pasa y el benchmark continuo etiquetado recupera los 20 turnos sin mezclar hablantes ni descartar frames.

La validación continua usa una sola forma de onda de 65,737 segundos, transmitida en frames consecutivos de 30 ms, construida repitiendo diez veces el fixture acústico etiquetado `intra_segment_two_speakers.wav`. El pipeline no recibe cortes de oración ni de speaker. Esta prueba valida continuidad, colas, ASR, diarización, respuestas cortas, pasada final y traducción bajo carga sostenida. El repositorio no contiene una grabación natural única de 60–120 segundos con transcripción humana; por eso esta evidencia no se presenta como sustituto de ese benchmark adicional.

## Arquitectura anterior

`WindowsCapture -> VAD segment -> Whisper -> TranscriptResult/SegmentReassembler -> traducción síncrona -> UI`

Había dos rutas que traducían unidades previas a la confirmación final:

- `TranslationWorker` consumía directamente `TranscriptResult` (`src/pipeline/workers.py` anterior: lectura en línea 313 y traducción en 352–354).
- La GUI traducía resultados temporales, resultados separados y salidas de `SegmentReassembler` dentro del mismo bucle (`traductor.py` anterior: 403, 487, 587 y 617).

El bloque heredado de la GUI fue retirado; la llamada restante de `traductor.py` es el calentamiento explícito de modelos, no audio STT.

## Arquitectura nueva

`captura continua -> resampler -> ring buffer -> VAD -> ASR con word timestamps -> IntraSegmentDiarizer -> SentenceTurnAssembler -> ConfirmedUtterance queue -> TranslationWorker -> Subtitle queue -> UI/historial`

Las etapas se ejecutan en workers independientes. La captura continúa mientras ASR, pasada final o traducción procesan la oración anterior. El cierre drena en orden las colas de audio, separación, ASR, confirmación y traducción antes de finalizar.

## Confirmación de oraciones

`SentenceTurnAssembler` conserva estado independiente por `speaker_id`, texto parcial, timestamps, audio, confianza y marcas temporales. Su `completion_score` combina:

| Evidencia | Peso |
|---|---:|
| Pausa acústica | 0,32 |
| Puntuación | 0,22 |
| Estructura sintáctica | 0,28 |
| Cambio de speaker | 0,12 |
| Duración | 0,06 |

Thresholds configurables:

| Parámetro | Valor |
|---|---:|
| `MIN_CONTINUATION_PAUSE_MS` | 250 ms |
| `SENTENCE_END_PAUSE_MS` | 650 ms |
| `TURN_END_PAUSE_MS` | 900 ms |
| `SENTENCE_COMPLETION_THRESHOLD` | 0,58 |
| `MAX_SENTENCE_DURATION_SECONDS` | 18 s |
| `FINAL_ASR_PRE_ROLL_MS` | 280 ms |
| `FINAL_ASR_POST_ROLL_MS` | 300 ms |

La heurística sintáctica contempla conectores, auxiliares, patrones abiertos y ausencia de verbo finito en inglés, español y portugués. Una coma o una micropausa no confirman por sí solas. Los límites internos con timestamps permiten separar varias oraciones del mismo monólogo.

Ante un cambio de speaker, el buffer anterior se cierra en su identidad; una cláusula abierta se marca como interrupción y jamás se concatena con el speaker nuevo. `IntraSegmentDiarizer` corre antes del assembler y ahora aplica divisiones recursivas, por lo que un único bloque VAD puede producir A -> B -> A.

## Pasada ASR final y doble VAD

Al confirmar, se reconstruye el audio completo en su línea temporal, promediando únicamente zonas solapadas para no duplicar pre-roll. Se agrega el padding configurado y se ejecuta Whisper sobre toda la unidad. Ese resultado tiene prioridad sobre la concatenación parcial. Si el detector de anomalías solicita retry, recibe el mismo audio completo.

ASR principal, retry y pasada final fuerzan `vad_filter=False` cuando el VAD externo ya delimitó la voz. Whisper resuelve directamente el snapshot instalado en `models`; no consulta Hugging Face durante la carga.

## Pruebas

- Suite completa: **141/141** pruebas aprobadas.
- Los ocho escenarios obligatorios están cubiertos en `tests/test_sentence_turn_assembler.py`.
- Se verifican además timeout combinado, heurística sintáctica, rechazo de `TranscriptResult` crudo por el traductor y las tres métricas nuevas.
- Audio real existente: 6/6 pruebas de ASR/VAD, respuestas cortas, inicio/final tenue, alternancia, diarización intrasegmento y bloqueo de red aprobadas.
- Stream continuo etiquetado: **65,737 s**, 2.192 frames de 30 ms, **0 descartados**, 20 turnos de referencia y 20 utterances recuperadas.

## Métricas del stream continuo

| Métrica | Antes | Nueva ruta |
|---|---:|---:|
| WER | 0,00% | 0,00% |
| CER | 0,00% | 4,42% |
| DER | 78,39% | 25,55% |
| Missed Word Rate | 0,00% | 0,00% |
| Missed Utterance Rate | 50,00% | 0,00% |
| Short Utterance Recall | 50,00% | 100,00% |
| Speaker Turn Precision | 0,00% | 100,00% |
| Speaker Turn Recall | 0,00% | 100,00% |
| Sentence Completeness Rate | 50,00% | 100,00% |
| Cross-Speaker Merge Rate | 100,00% | 0,00% |
| SOURCE_DIALOGUE_COVERAGE | 100,00% | 100,00% |

El baseline conserva todas las palabras pero las agrupa de a dos speakers; por eso su WER y cobertura son 0%/100% aunque pierde el 50% de las unidades y mezcla el 100%. El CER nuevo refleja la coma que Whisper produce en `Yes, I am`; WER normaliza puntuación. El DER restante se debe principalmente a que la referencia existente anota intervalos amplios que incluyen silencio, mientras la salida usa los límites de palabras de Whisper.

Latencia medida desde el final acústico hasta el subtítulo traducido, con Whisper `base` y traducción Marian en CPU:

- promedio: **1.635,3 ms**
- p95: **1.762,2 ms**
- máximo: **2.297,0 ms**

La máquina intentó CUDA pero no encontró `cublas64_12.dll` y pasó correctamente a CPU `int8`. La precisión y la cobertura se conservaron; la latencia queda por encima del objetivo orientativo de 300–1.200 ms para este hardware.

Los datos completos, cada utterance y las traducciones están en `benchmark_results/sentence_pipeline_continuous.json`.
