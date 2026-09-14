"""
src/diagnostics/ground_truth_evaluator.py
Evaluador automático de fidelidad Ground Truth vs Salida del Sistema.
Calcula métricas objetivas:
  - WER (Word Error Rate)
  - CER (Character Error Rate)
  - DER (Diarization Error Rate)
  - Missed Word Rate
  - Missed Utterance Rate
  - Short Utterance Recall
  - Speaker Turn Precision & Recall
  - Sentence Completeness Rate
  - Cross-Speaker Merge Rate
  - Source Dialogue Coverage
Y genera reportes detallados por cada discrepancia (ERROR #XX).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.utils.metrics import compute_wer, compute_cer, compute_der, DERMetrics
from src.diagnostics.loss_detector import LossDetector, LossClassification

logger = logging.getLogger(__name__)


@dataclass
class ErrorReportItem:
    error_index: int
    time_range: str
    ground_truth: str
    system_text: str
    missing_words: str
    audio_file: str
    raw_whisper: str
    retry_whisper: str
    reassembled: str
    speaker_gt: str
    speaker_sys: str
    loss_class: str
    reason: str


@dataclass
class BenchmarkEvaluationResult:
    wer: float
    cer: float
    der: float
    missed_word_rate: float
    missed_utterance_rate: float
    short_utterance_recall: float
    speaker_turn_precision: float
    speaker_turn_recall: float
    sentence_completeness_rate: float
    cross_speaker_merge_rate: float
    source_dialogue_coverage: float
    total_reference_words: int
    total_hypothesis_words: int
    total_utterances: int
    missed_utterances_count: int
    short_utterances_count: int
    short_utterances_captured: int
    errors: List[ErrorReportItem] = field(default_factory=list)
    der_metrics: Optional[DERMetrics] = None


class GroundTruthEvaluator:
    """
    Compara de forma estricta las transcripciones y turnos de hablante producidos por el sistema
    contra las anotaciones de referencia manual (ground_truth.json).
    """

    def __init__(self, loss_detector: Optional[LossDetector] = None):
        self.loss_detector = loss_detector or LossDetector()

    def evaluate(
        self,
        ground_truth_turns: List[Dict[str, Any]],
        system_outputs: List[Dict[str, Any]],
    ) -> BenchmarkEvaluationResult:
        """
        Ejecuta la evaluación completa alineando turnos de referencia con salidas del sistema.
        """
        total_ref_words = 0
        total_hyp_words = 0
        total_utterances = len(ground_truth_turns)
        missed_utterances_count = 0
        short_utterances_count = 0
        short_utterances_captured = 0

        ref_full_text = []
        hyp_full_text = []

        ref_der_turns = []
        hyp_der_turns = []

        errors: List[ErrorReportItem] = []
        error_counter = 0
        complete_utterances = 0
        matched_system_indexes = set()

        # Mapeo temporal para asociar ground truth con salidas del sistema
        for gt_turn in ground_truth_turns:
            gt_text = gt_turn.get("text", "").strip()
            gt_spk = gt_turn.get("speaker_id", gt_turn.get("speaker", "SPEAKER_01"))
            gt_start = float(gt_turn.get("start", 0.0))
            gt_end = float(gt_turn.get("end", 0.0))

            ref_full_text.append(gt_text)
            ref_der_turns.append((gt_start, gt_end, gt_spk))

            words_gt = gt_text.split()
            total_ref_words += len(words_gt)
            is_short = len(words_gt) <= 3 or (gt_end - gt_start <= 1.2)
            if is_short:
                short_utterances_count += 1

            # Buscar salida del sistema que solape temporalmente con este turno
            matching_sys = None
            best_overlap = 0.0

            matching_index = None
            for s_index, s_out in enumerate(system_outputs):
                if s_index in matched_system_indexes:
                    continue
                s_start = float(s_out.get("start", s_out.get("start_time", 0.0)))
                s_end = float(s_out.get("end", s_out.get("end_time", 0.0)))

                overlap_start = max(gt_start, s_start)
                overlap_end = min(gt_end, s_end)
                overlap_dur = max(0.0, overlap_end - overlap_start)

                if overlap_dur > best_overlap:
                    best_overlap = overlap_dur
                    matching_sys = s_out
                    matching_index = s_index

            # Si no hubo overlap pero hay salida disponible por secuencia
            if matching_sys is None and len(system_outputs) == 1 and total_utterances == 1:
                matching_sys = system_outputs[0]
                matching_index = 0

            if matching_index is not None:
                matched_system_indexes.add(matching_index)

            sys_text = (matching_sys.get("text", matching_sys.get("final_stt", "")) if matching_sys else "").strip()
            sys_spk = (matching_sys.get("speaker_id", matching_sys.get("speaker", "SPEAKER_01")) if matching_sys else "")
            audio_file = matching_sys.get("audio_file", "") if matching_sys else ""
            raw_whisper = matching_sys.get("raw_stt", sys_text) if matching_sys else ""
            retry_whisper = matching_sys.get("retry_stt", raw_whisper) if matching_sys else ""
            reassembled = matching_sys.get("reassembled_stt", sys_text) if matching_sys else ""

            if sys_text:
                hyp_full_text.append(sys_text)
                total_hyp_words += len(sys_text.split())
                if is_short:
                    short_utterances_captured += 1
                if self._is_sentence_complete(gt_text, sys_text):
                    complete_utterances += 1
            else:
                missed_utterances_count += 1

            # Clasificar pérdida si hay discrepancia
            diff_words = self._extract_missing_words(gt_text, sys_text)
            has_error = (len(diff_words) > 0) or (gt_spk != sys_spk and sys_text) or not sys_text

            if has_error:
                error_counter += 1
                audio_arr = matching_sys.get("audio_array") if matching_sys else None
                loss_info: LossClassification = self.loss_detector.classify(
                    audio=audio_arr,
                    raw_whisper_text=raw_whisper,
                    final_source_text=sys_text,
                    ground_truth_text=gt_text,
                )

                item = ErrorReportItem(
                    error_index=error_counter,
                    time_range=f"{gt_start:.2f}s - {gt_end:.2f}s",
                    ground_truth=gt_text,
                    system_text=sys_text or "(vacío / no detectado)",
                    missing_words=" ".join(diff_words) if diff_words else "(ninguna - posible sustitución o hablante)",
                    audio_file=audio_file or "(no registrado)",
                    raw_whisper=raw_whisper,
                    retry_whisper=retry_whisper,
                    reassembled=reassembled,
                    speaker_gt=gt_spk,
                    speaker_sys=sys_spk or "(none)",
                    loss_class=loss_info.loss_type,
                    reason=loss_info.reason,
                )
                errors.append(item)

        hyp_full_text = []
        total_hyp_words = 0
        for s_out in sorted(system_outputs, key=lambda item: float(item.get("start", item.get("start_time", 0.0)))):
            s_start = float(s_out.get("start", s_out.get("start_time", 0.0)))
            s_end = float(s_out.get("end", s_out.get("end_time", 0.0)))
            s_spk = s_out.get("speaker_id", s_out.get("speaker", "SPEAKER_01"))
            hyp_der_turns.append((s_start, s_end, s_spk))
            output_text = str(s_out.get("text", s_out.get("final_stt", ""))).strip()
            if output_text:
                hyp_full_text.append(output_text)
                total_hyp_words += len(output_text.split())

        # Métricas globales
        full_ref = " ".join(ref_full_text)
        full_hyp = " ".join(hyp_full_text)

        wer = compute_wer(full_ref, full_hyp)
        cer = compute_cer(full_ref, full_hyp)
        der_metrics = compute_der(ref_der_turns, hyp_der_turns)

        source_coverage = self._coverage(full_ref, full_hyp)
        missed_word_rate = 1.0 - source_coverage
        missed_utterance_rate = (missed_utterances_count / max(1, total_utterances)) * 100.0
        short_recall = (
            (short_utterances_captured / max(1, short_utterances_count)) * 100.0
            if short_utterances_count > 0
            else 100.0
        )

        # Precisión y Recall de cambios de turno
        turn_precision, turn_recall = self._compute_turn_metrics(ref_der_turns, hyp_der_turns)
        sentence_completeness = complete_utterances / max(1, total_utterances)
        cross_speaker_merge = self._cross_speaker_merge_rate(ground_truth_turns, system_outputs)

        return BenchmarkEvaluationResult(
            wer=round(wer * 100.0, 2),
            cer=round(cer * 100.0, 2),
            der=round(der_metrics.der_percentage, 2),
            missed_word_rate=round(missed_word_rate * 100.0, 2),
            missed_utterance_rate=round(missed_utterance_rate, 2),
            short_utterance_recall=round(short_recall, 2),
            speaker_turn_precision=round(turn_precision * 100.0, 2),
            speaker_turn_recall=round(turn_recall * 100.0, 2),
            sentence_completeness_rate=round(sentence_completeness * 100.0, 2),
            cross_speaker_merge_rate=round(cross_speaker_merge * 100.0, 2),
            source_dialogue_coverage=round(source_coverage * 100.0, 2),
            total_reference_words=total_ref_words,
            total_hypothesis_words=total_hyp_words,
            total_utterances=total_utterances,
            missed_utterances_count=missed_utterances_count,
            short_utterances_count=short_utterances_count,
            short_utterances_captured=short_utterances_captured,
            errors=errors,
            der_metrics=der_metrics,
        )

    def format_error_report(self, errors: List[ErrorReportItem]) -> str:
        """Formatea los errores detectados exactamente según la estructura solicitada."""
        if not errors:
            return "No se detectaron discrepancias entre Ground Truth y Salida del Sistema."

        lines = []
        for e in errors:
            lines.append(f"ERROR #{e.error_index}")
            lines.append(f"Time: {e.time_range}")
            lines.append(f"Ground truth: \"{e.ground_truth}\"")
            lines.append(f"System:       \"{e.system_text}\"")
            lines.append(f"Missing:      \"{e.missing_words}\"")
            lines.append(f"Audio file:   {e.audio_file}")
            lines.append(f"Raw Whisper:  \"{e.raw_whisper}\"")
            lines.append(f"Retry:        \"{e.retry_whisper}\"")
            lines.append(f"Reassembled:  \"{e.reassembled}\"")
            lines.append(f"Speaker:      GT=[{e.speaker_gt}] vs SYS=[{e.speaker_sys}]")
            lines.append(f"Clase Pérdida: {e.loss_class} ({e.reason})")
            lines.append("-" * 60)
        return "\n".join(lines)

    @staticmethod
    def _extract_missing_words(reference: str, hypothesis: str) -> List[str]:
        """Extrae palabras de referencia que no aparecen en la hipótesis (en orden)."""
        import re
        ref_tokens = re.findall(r"\b\w+(?:'\w+)?\b", reference.lower())
        hyp_tokens = set(re.findall(r"\b\w+(?:'\w+)?\b", hypothesis.lower()))
        return [w for w in ref_tokens if w not in hyp_tokens]

    @classmethod
    def _coverage(cls, reference: str, hypothesis: str) -> float:
        ref = cls._tokens(reference)
        hyp = cls._tokens(hypothesis)
        return cls._lcs_length(ref, hyp) / max(1, len(ref))

    @classmethod
    def _is_sentence_complete(cls, reference: str, hypothesis: str) -> bool:
        ref = cls._tokens(reference)
        hyp = cls._tokens(hypothesis)
        if not ref or not hyp:
            return False
        if cls._lcs_length(ref, hyp) / len(ref) < 0.90:
            return False
        return ref[0] in hyp and ref[-1] in hyp and hyp.index(ref[0]) <= len(hyp) - 1 - hyp[::-1].index(ref[-1])

    @classmethod
    def _cross_speaker_merge_rate(cls, ground_truth_turns, system_outputs) -> float:
        merged = 0
        nonempty = 0
        for output in system_outputs:
            text = str(output.get("text", output.get("final_stt", ""))).strip()
            if not text:
                continue
            nonempty += 1
            start = float(output.get("start", output.get("start_time", 0.0)))
            end = float(output.get("end", output.get("end_time", 0.0)))
            supported_speakers = set()
            for turn in ground_truth_turns:
                turn_start = float(turn.get("start", 0.0))
                turn_end = float(turn.get("end", 0.0))
                if min(end, turn_end) <= max(start, turn_start):
                    continue
                reference_tokens = cls._tokens(str(turn.get("text", "")))
                matches = cls._lcs_length(reference_tokens, cls._tokens(text))
                minimum = 1 if len(reference_tokens) <= 3 else 2
                if matches >= minimum:
                    supported_speakers.add(turn.get("speaker_id", turn.get("speaker", "SPEAKER_01")))
            if len(supported_speakers) > 1:
                merged += 1
        return merged / max(1, nonempty)

    @staticmethod
    def _tokens(text: str) -> List[str]:
        import re
        return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())

    @staticmethod
    def _lcs_length(left: List[str], right: List[str]) -> int:
        previous = [0] * (len(right) + 1)
        for left_token in left:
            current = [0]
            for index, right_token in enumerate(right, start=1):
                if left_token == right_token:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(max(current[-1], previous[index]))
            previous = current
        return previous[-1]

    @staticmethod
    def _compute_turn_metrics(ref_turns, hyp_turns) -> Tuple[float, float]:
        """Calcula precisión y recall de detección de transiciones de hablante."""
        ref_transitions = [
            t2[0] for t1, t2 in zip(ref_turns[:-1], ref_turns[1:]) if t1[2] != t2[2]
        ]
        hyp_transitions = [
            t2[0] for t1, t2 in zip(hyp_turns[:-1], hyp_turns[1:]) if t1[2] != t2[2]
        ]

        if not ref_transitions:
            return (1.0, 1.0) if not hyp_transitions else (0.0, 1.0)

        # Tolerancia temporal de 0.5s para transiciones
        tolerance = 0.5
        matched_ref = 0
        matched_hyp = 0

        for r_t in ref_transitions:
            if any(abs(r_t - h_t) <= tolerance for h_t in hyp_transitions):
                matched_ref += 1

        for h_t in hyp_transitions:
            if any(abs(h_t - r_t) <= tolerance for r_t in ref_transitions):
                matched_hyp += 1

        precision = matched_hyp / max(1, len(hyp_transitions))
        recall = matched_ref / max(1, len(ref_transitions))
        return precision, recall
