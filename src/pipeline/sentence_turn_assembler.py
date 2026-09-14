"""Speaker-isolated sentence assembly and whole-utterance ASR confirmation."""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config.settings import (
    FINAL_ASR_POST_ROLL_MS,
    FINAL_ASR_PRE_ROLL_MS,
    MAX_SENTENCE_DURATION_SECONDS,
    MIN_CONTINUATION_PAUSE_MS,
    SENTENCE_COMPLETION_THRESHOLD,
    SENTENCE_END_PAUSE_MS,
    TURN_END_PAUSE_MS,
)
from src.pipeline.models import ConfirmedUtterance

logger = logging.getLogger(__name__)

_CONNECTORS = {
    "and", "but", "because", "so", "if", "when", "that", "which", "who", "to", "for", "with", "of", "from", "about", "in", "on", "at", "by", "as", "than", "while", "although", "unless", "or",
    "y", "pero", "porque", "si", "cuando", "que", "cual", "quien", "para", "con", "de", "desde", "sobre", "en", "por", "como", "aunque", "mientras", "o",
    "e", "mas", "pois", "se", "quando", "qual", "quem", "com", "de", "desde", "sobre", "em", "por", "como", "embora", "enquanto", "ou",
}
_AUXILIARIES = {
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must",
    "es", "son", "era", "eran", "ser", "estar", "estoy", "está", "estan", "he", "ha", "han", "puede", "podría", "debe",
    "é", "são", "era", "ser", "estar", "estou", "está", "tenho", "tem", "pode", "deve",
}
_OPEN_PATTERNS = [
    re.compile(r"\b(?:want(?:ed)?|need(?:ed)?|try(?:ing|ied)?|going|used)\s+to\s*$", re.I),
    re.compile(r"\b(?:because|although|unless|while|when|if)\s+[^.!?]{0,80}$", re.I),
    re.compile(r"\b(?:quiero|quería|necesito|trato|voy)\s+(?:a|de)\s*$", re.I),
    re.compile(r"\b(?:porque|aunque|mientras|cuando|si)\s+[^.!?]{0,80}$", re.I),
    re.compile(r"\b(?:quero|queria|preciso|tento|vou)\s+(?:a|de)\s*$", re.I),
]
_FINITE_HINT = re.compile(
    r"\b(?:am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|should|want|need|think|feel|know|mean|went|said|made|works?|matters?|"
    r"soy|es|somos|son|era|fui|fue|tengo|tiene|quiero|necesito|pienso|siento|sé|dijo|hice|funciona|importa|"
    r"sou|é|somos|são|era|fui|foi|tenho|tem|quero|preciso|penso|sinto|sei|disse|funciona|importa)\b",
    re.I,
)


def is_likely_incomplete(text: str) -> bool:
    """Estimate syntactic openness without treating one example phrase as a rule."""
    clean = " ".join(str(text or "").strip().split())
    if not clean:
        return True
    if clean.endswith(("...", "…", "—", "-", ",", ":", ";")):
        return True
    if clean[-1:] in ".?!":
        return False
    if any(pattern.search(clean) for pattern in _OPEN_PATTERNS):
        return True
    tokens = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", clean.lower(), re.UNICODE)
    if not tokens:
        return True
    if tokens[-1] in _CONNECTORS or tokens[-1] in _AUXILIARIES:
        return True
    # A short noun phrase or an utterance without a finite predicate is weak evidence
    # of completion; longer clauses with a predicate may close after acoustic silence.
    return len(tokens) >= 2 and _FINITE_HINT.search(clean) is None


@dataclass
class CompletionEvidence:
    pause_score: float
    punctuation_score: float
    syntax_score: float
    speaker_change_score: float
    duration_score: float
    total: float


@dataclass
class _Fragment:
    sequence_id: int
    session_id: Optional[str]
    speaker_id: str
    text: str
    language: str
    start_time: float
    end_time: float
    confidence: float
    audio: np.ndarray
    words: List[Dict[str, Any]]
    captured_at: float
    asr_duration: float
    is_partial: bool
    is_overlap: bool
    trailing_silence_ms: int
    received_at: float = field(default_factory=time.monotonic)


@dataclass
class _TurnState:
    speaker_id: str
    fragments: List[_Fragment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return SentenceTurnAssembler.join_text([part.text for part in self.fragments])

    @property
    def start_time(self) -> float:
        return self.fragments[0].start_time

    @property
    def end_time(self) -> float:
        return self.fragments[-1].end_time


class SentenceTurnAssembler:
    """Accumulates fragments per speaker and emits only ConfirmedUtterance objects."""

    def __init__(
        self,
        asr_engine: Optional[Any] = None,
        retry_engine: Optional[Any] = None,
        sample_rate: int = 16000,
        min_continuation_pause_ms: int = MIN_CONTINUATION_PAUSE_MS,
        sentence_end_pause_ms: int = SENTENCE_END_PAUSE_MS,
        turn_end_pause_ms: int = TURN_END_PAUSE_MS,
        completion_threshold: float = SENTENCE_COMPLETION_THRESHOLD,
        max_turn_seconds: float = MAX_SENTENCE_DURATION_SECONDS,
        pre_roll_ms: int = FINAL_ASR_PRE_ROLL_MS,
        post_roll_ms: int = FINAL_ASR_POST_ROLL_MS,
        initial_prompt: Optional[str] = None,
        on_debug: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ):
        self.asr_engine = asr_engine
        self.retry_engine = retry_engine
        self.sample_rate = sample_rate
        self.min_continuation_pause_ms = min_continuation_pause_ms
        self.sentence_end_pause_ms = sentence_end_pause_ms
        self.turn_end_pause_ms = turn_end_pause_ms
        self.completion_threshold = completion_threshold
        self.max_turn_seconds = max_turn_seconds
        self.pre_roll_ms = pre_roll_ms
        self.post_roll_ms = post_roll_ms
        self.initial_prompt = initial_prompt
        self.on_debug = on_debug
        self.current_turn: Dict[str, _TurnState] = {}
        self.last_active_speaker: Optional[str] = None

    def reset(self) -> None:
        self.current_turn.clear()
        self.last_active_speaker = None

    def add_event(
        self,
        *,
        speaker_id: str,
        text: str,
        word_timestamps: Optional[List[Dict[str, Any]]],
        audio: np.ndarray,
        audio_start: float,
        audio_end: float,
        is_partial: bool,
        confidence: float,
        source_language: str = "en",
        sequence_id: int = 0,
        session_id: Optional[str] = None,
        captured_at: Optional[float] = None,
        asr_duration: float = 0.0,
        trailing_silence_ms: int = 0,
        is_overlap: bool = False,
    ) -> List[ConfirmedUtterance]:
        clean = " ".join(str(text or "").strip().split())
        if not clean:
            return []
        emitted: List[ConfirmedUtterance] = []
        if self.last_active_speaker and self.last_active_speaker != speaker_id:
            previous = self._finalize_speaker(
                self.last_active_speaker,
                reason="speaker_change",
                interrupted=is_likely_incomplete(self.current_turn[self.last_active_speaker].text),
                speaker_change=True,
            )
            if previous:
                emitted.append(previous)
        self.last_active_speaker = speaker_id
        state = self.current_turn.get(speaker_id)
        if state and not state.fragments[-1].is_partial and not is_likely_incomplete(state.text):
            pause_ms = state.fragments[-1].trailing_silence_ms + max(
                0, int(round((audio_start - state.end_time) * 1000))
            )
            evidence = self.completion_score(state.text, pause_ms, False, state.end_time - state.start_time)
            if pause_ms >= self.turn_end_pause_ms or (pause_ms >= self.sentence_end_pause_ms and evidence.total >= self.completion_threshold):
                previous = self._finalize_speaker(speaker_id, "acoustic_gap", evidence=evidence)
                if previous:
                    emitted.append(previous)

        event = _Fragment(
            sequence_id=sequence_id,
            session_id=session_id,
            speaker_id=speaker_id,
            text=clean,
            language=source_language or "en",
            start_time=float(audio_start),
            end_time=float(audio_end),
            confidence=float(confidence),
            audio=np.asarray(audio, dtype=np.float32).reshape(-1),
            words=list(word_timestamps or []),
            captured_at=captured_at if captured_at is not None else time.monotonic(),
            asr_duration=float(asr_duration),
            is_partial=bool(is_partial),
            is_overlap=bool(is_overlap),
            trailing_silence_ms=max(0, int(trailing_silence_ms)),
        )
        pieces = self._split_internal_sentences(event)
        for index, piece in enumerate(pieces):
            state = self.current_turn.setdefault(speaker_id, _TurnState(speaker_id))
            self.last_active_speaker = speaker_id
            state.fragments.append(piece)
            self._debug("PARTIAL", speaker_id, {"text": state.text, "fragments": len(state.fragments)})
            internal_boundary = index < len(pieces) - 1
            evidence = self.completion_score(
                state.text,
                pause_ms=piece.trailing_silence_ms,
                speaker_change=False,
                duration_seconds=state.end_time - state.start_time,
            )
            if internal_boundary or (not piece.is_partial and evidence.total >= self.completion_threshold):
                utterance = self._finalize_speaker(speaker_id, "internal_sentence" if internal_boundary else "completion_score", evidence=evidence)
                if utterance:
                    emitted.append(utterance)
            elif state.end_time - state.start_time >= self.max_turn_seconds:
                utterance = self._finalize_speaker(speaker_id, "max_turn", interrupted=is_likely_incomplete(state.text), evidence=evidence)
                if utterance:
                    emitted.append(utterance)
        return emitted

    def check_timeouts(
        self,
        now: Optional[float] = None,
        acoustic_progress: Optional[tuple] = None,
    ) -> List[ConfirmedUtterance]:
        # `now` remains accepted for callers, but wall-clock waiting is not silence.
        # The capture/VAD snapshot is (processed audio end, last voiced audio end).
        emitted: List[ConfirmedUtterance] = []
        for speaker_id, state in list(self.current_turn.items()):
            last = state.fragments[-1]
            total_pause = last.trailing_silence_ms
            if acoustic_progress is not None:
                processed_end, voice_end = acoustic_progress
                # Later speech may still be in VAD, separation, or ASR. Never close
                # the old fragment while its continuation is being processed.
                if processed_end < state.end_time or voice_end > state.end_time + 0.001:
                    continue
                speech_end = max(voice_end, state.end_time - total_pause / 1000.0)
                total_pause = max(total_pause, int(round((processed_end - speech_end) * 1000)))
            elif last.is_partial:
                continue
            if is_likely_incomplete(state.text):
                continue
            evidence = self.completion_score(state.text, total_pause, False, state.end_time - state.start_time)
            if total_pause >= self.turn_end_pause_ms or (total_pause >= self.sentence_end_pause_ms and evidence.total >= self.completion_threshold):
                item = self._finalize_speaker(speaker_id, "silence_timeout", evidence=evidence)
                if item:
                    emitted.append(item)
        return emitted

    def flush_all(self, reason: str = "shutdown") -> List[ConfirmedUtterance]:
        output = []
        for speaker_id in list(self.current_turn):
            state = self.current_turn[speaker_id]
            item = self._finalize_speaker(speaker_id, reason, interrupted=is_likely_incomplete(state.text))
            if item:
                output.append(item)
        self.last_active_speaker = None
        return output

    def completion_score(self, text: str, pause_ms: int, speaker_change: bool, duration_seconds: float) -> CompletionEvidence:
        if pause_ms < self.min_continuation_pause_ms:
            pause = 0.0
        elif pause_ms < self.sentence_end_pause_ms:
            pause = 0.25 + 0.35 * (pause_ms - self.min_continuation_pause_ms) / max(1, self.sentence_end_pause_ms - self.min_continuation_pause_ms)
        elif pause_ms < self.turn_end_pause_ms:
            pause = 0.80 + 0.20 * (pause_ms - self.sentence_end_pause_ms) / max(1, self.turn_end_pause_ms - self.sentence_end_pause_ms)
        else:
            pause = 1.0
        clean = text.rstrip()
        punctuation = 0.0 if clean.endswith((",", ":", ";", "...", "…")) else 1.0 if clean.endswith((".", "?", "!")) else 0.25
        syntax = 0.0 if is_likely_incomplete(clean) else 0.85
        speaker = 1.0 if speaker_change else 0.0
        duration = min(1.0, max(0.0, duration_seconds / self.max_turn_seconds))
        total = 0.32 * pause + 0.22 * punctuation + 0.28 * syntax + 0.12 * speaker + 0.06 * duration
        return CompletionEvidence(pause, punctuation, syntax, speaker, duration, round(total, 4))

    def _finalize_speaker(
        self,
        speaker_id: str,
        reason: str,
        interrupted: bool = False,
        speaker_change: bool = False,
        evidence: Optional[CompletionEvidence] = None,
    ) -> Optional[ConfirmedUtterance]:
        state = self.current_turn.pop(speaker_id, None)
        if not state or not state.fragments:
            return None
        if self.last_active_speaker == speaker_id:
            self.last_active_speaker = None
        evidence = evidence or self.completion_score(state.text, state.fragments[-1].trailing_silence_ms, speaker_change, state.end_time - state.start_time)
        self._debug("FINALIZING", speaker_id, {"text": state.text, "reason": reason, "score": evidence.total})
        merged_audio = self._merge_audio(state.fragments)
        partial_text = state.text
        language = state.fragments[-1].language
        final_text = partial_text
        final_words = self._merge_words(state.fragments)
        final_confidence = float(np.mean([part.confidence for part in state.fragments]))
        final_pass_used = False
        total_asr_duration = sum(part.asr_duration for part in state.fragments)
        if self.asr_engine is not None and len(merged_audio):
            padded = np.concatenate([
                np.zeros(int(self.pre_roll_ms * self.sample_rate / 1000), dtype=np.float32),
                merged_audio,
                np.zeros(int(self.post_roll_ms * self.sample_rate / 1000), dtype=np.float32),
            ])
            started = time.monotonic()
            try:
                result = self.asr_engine.transcribe(
                    padded,
                    language=language,
                    beam_size=2,
                    initial_prompt=self.initial_prompt,
                    vad_filter=False,
                )
                total_asr_duration += time.monotonic() - started
                if self.retry_engine is not None:
                    result = self.retry_engine.maybe_retry(
                        audio=padded,
                        stt_result=result,
                        context_prompt=self.initial_prompt,
                        language=language,
                        vad_filter=False,
                    )
                candidate = " ".join(str(result.get("text", "")).strip().split())
                if candidate:
                    final_text = candidate
                    language = result.get("language") or language
                    final_words = self._absolute_words(result.get("words", []), state.start_time - self.pre_roll_ms / 1000.0)
                    final_confidence = float(result.get("confidence", result.get("probability", final_confidence)))
                    final_pass_used = True
            except Exception as exc:
                logger.warning("Final ASR pass failed for %s: %s; retaining confirmed partial text.", speaker_id, exc)
        complete = not interrupted and not is_likely_incomplete(final_text)
        utterance = ConfirmedUtterance(
            utterance_id=f"utt_{uuid.uuid4().hex[:16]}",
            sequence_id=state.fragments[0].sequence_id,
            session_id=state.fragments[0].session_id,
            speaker_id=speaker_id,
            start_time=state.start_time,
            end_time=state.end_time,
            source_text=final_text,
            source_language=language,
            word_timestamps=final_words,
            stt_confidence=max(0.0, min(1.0, final_confidence)),
            audio_reference=None,
            is_interrupted=bool(interrupted),
            is_complete=complete,
            final_asr_pass_used=final_pass_used,
            captured_at=max(part.captured_at for part in state.fragments),
            asr_duration=total_asr_duration,
            completion_score=evidence.total,
            completion_reason=reason,
            partial_fragments=[part.text for part in state.fragments],
            audio=merged_audio,
            is_overlap=any(part.is_overlap for part in state.fragments),
        )
        self._debug("CONFIRMED", speaker_id, {"text": utterance.source_text, "reason": reason, "final_asr_pass_used": final_pass_used})
        return utterance

    def _split_internal_sentences(self, event: _Fragment) -> List[_Fragment]:
        words = event.words
        if len(words) < 2:
            return [event]
        boundaries = [
            index for index, word in enumerate(words[:-1])
            if str(word.get("word", "")).rstrip().endswith((".", "?", "!"))
            and not str(word.get("word", "")).rstrip().endswith(("...", "…"))
        ]
        if not boundaries:
            return [event]
        # Word timestamps are estimates. Partition the original audio contiguously
        # rather than discarding pre-roll, post-roll and inter-word samples.
        cuts = []
        for boundary in boundaries:
            left = float(words[boundary].get("end", 0.0))
            right = float(words[boundary + 1].get("start", left))
            if not np.isfinite(left) or not np.isfinite(right):
                return [event]
            cut = int(round((left + right) * 0.5 * self.sample_rate))
            if cut <= (cuts[-1] if cuts else 0) or cut >= len(event.audio):
                return [event]
            cuts.append(cut)
        pieces: List[_Fragment] = []
        first = 0
        sample_start = 0
        for boundary, sample_end in zip(boundaries + [len(words) - 1], cuts + [len(event.audio)]):
            group = words[first:boundary + 1]
            rel_start = sample_start / self.sample_rate
            rel_end = sample_end / self.sample_rate
            text = " ".join(str(word.get("word", "")).strip() for word in group).strip()
            if not text:
                return [event]
            pieces.append(_Fragment(
                **{**event.__dict__, "text": text, "audio": event.audio[sample_start:sample_end],
                   "start_time": event.start_time + rel_start, "end_time": event.start_time + rel_end,
                   "words": [{**word, "start": float(word.get("start", 0.0)) - rel_start, "end": float(word.get("end", 0.0)) - rel_start} for word in group],
                   "is_partial": False if boundary < len(words) - 1 else event.is_partial,
                   "trailing_silence_ms": self.sentence_end_pause_ms if boundary < len(words) - 1 else event.trailing_silence_ms}))
            first = boundary + 1
            sample_start = sample_end
        return pieces

    def _merge_audio(self, fragments: List[_Fragment]) -> np.ndarray:
        start = min(part.start_time for part in fragments)
        end = max(part.end_time for part in fragments)
        size = max(1, int(round((end - start) * self.sample_rate)))
        mixed = np.zeros(size, dtype=np.float32)
        counts = np.zeros(size, dtype=np.float32)
        for part in fragments:
            offset = max(0, int(round((part.start_time - start) * self.sample_rate)))
            length = min(len(part.audio), size - offset)
            if length > 0:
                mixed[offset:offset + length] += part.audio[:length]
                counts[offset:offset + length] += 1.0
        populated = counts > 0
        mixed[populated] /= counts[populated]
        return mixed

    def _merge_words(self, fragments: List[_Fragment]) -> List[Dict[str, Any]]:
        output = []
        for part in fragments:
            output.extend(self._absolute_words(part.words, part.start_time))
        return output

    @staticmethod
    def _absolute_words(words: List[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
        return [{**word, "start": max(0.0, offset + float(word.get("start", 0.0))), "end": max(0.0, offset + float(word.get("end", 0.0)))} for word in words]

    @staticmethod
    def join_text(parts: List[str]) -> str:
        text = ""
        for part in parts:
            clean = str(part).strip()
            if not clean:
                continue
            if text and text.endswith(("-", "—")):
                text = text[:-1].rstrip() + clean
            else:
                text = f"{text} {clean}".strip()
        return re.sub(r"\s+([,.?!;:])", r"\1", text)

    def _debug(self, stage: str, speaker_id: str, data: Dict[str, Any]) -> None:
        logger.debug("[%s] %s: %s", stage, speaker_id, data.get("text", ""))
        if self.on_debug:
            self.on_debug(stage, speaker_id, data)
