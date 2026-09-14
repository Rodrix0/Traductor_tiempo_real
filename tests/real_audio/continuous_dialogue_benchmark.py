"""Run the real asynchronous pipeline on one uninterrupted 65.7-second stream."""

import json
import queue
import statistics
import sys
import threading
import time
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.asr.whisper_engine import WhisperEngine
from src.audio.vad import VoiceActivityDetector
from src.diagnostics.ground_truth_evaluator import GroundTruthEvaluator
from src.pipeline.controller import PipelineController
from src.pipeline.models import AudioChunk, AudioSourceMode, VADProfile
from src.speakers.tracker import SpeakerTracker
from src.translation.context_aware_translator import ContextAwareTranslator
from src.translation.context_manager import ConversationContextManager


SOURCE_AUDIO = Path(__file__).with_name("intra_segment_two_speakers.wav")
REPETITIONS = 10
FRAME_SAMPLES = 480


def load_wave(path):
    with wave.open(str(path), "rb") as stream:
        sample_rate = stream.getframerate()
        channels = stream.getnchannels()
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16000:
        raise RuntimeError(f"Fixture must be 16 kHz; found {sample_rate}")
    return audio


class ContinuousArrayCapture:
    """Feeds one waveform in 30 ms frames; no utterance boundaries are provided."""

    def __init__(self, audio, realtime=True):
        self.audio = audio
        self.realtime = realtime
        self.source_mode = AudioSourceMode.SYSTEM
        self.ring_buffer = None
        self.done = threading.Event()
        self.stop_requested = threading.Event()
        self.thread = None
        self.dropped = 0
        self.frame_intervals = []

    def stream_to_queue(self, target_queue, stop_event=None):
        def produce():
            previous = None
            target_time = time.monotonic()
            try:
                for offset in range(0, len(self.audio), FRAME_SAMPLES):
                    if self.stop_requested.is_set():
                        break
                    now = time.monotonic()
                    if previous is not None:
                        self.frame_intervals.append(now - previous)
                    previous = now
                    frame = self.audio[offset:offset + FRAME_SAMPLES]
                    target_queue.put(AudioChunk(frame, timestamp=now, source_mode=self.source_mode))
                    if self.realtime:
                        target_time += len(frame) / 16000.0
                        delay = target_time - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
            finally:
                self.done.set()

        self.thread = threading.Thread(target=produce, name="ContinuousGroundTruthCapture", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_requested.set()
        if self.thread:
            self.thread.join(timeout=2.0)


def ground_truth(cycle_seconds):
    turns = []
    for cycle in range(REPETITIONS):
        base = cycle * cycle_seconds
        turns.extend([
            {"speaker_id": "SPEAKER_01", "start": base, "end": base + 1.50, "text": "Are you ready?"},
            {"speaker_id": "SPEAKER_02", "start": base + 1.62, "end": base + 6.57, "text": "Yes I am."},
        ])
    return turns


def baseline_outputs(cycle_seconds):
    return [
        {
            "speaker_id": "SPEAKER_01",
            "start": cycle * cycle_seconds,
            "end": (cycle + 1) * cycle_seconds,
            "text": "Are you ready? Yes I am.",
        }
        for cycle in range(REPETITIONS)
    ]


def run(realtime=True):
    source = load_wave(SOURCE_AUDIO)
    cycle_seconds = len(source) / 16000.0
    continuous_audio = np.concatenate([source] * REPETITIONS)
    duration = len(continuous_audio) / 16000.0
    if not 60.0 <= duration <= 120.0:
        raise AssertionError(f"Continuous benchmark must last 60-120 seconds, got {duration:.2f}")

    outputs = []
    latencies = []
    debug_stages = []
    capture = ContinuousArrayCapture(continuous_audio, realtime=realtime)
    engine = WhisperEngine(model_size="base")
    translator = ContextAwareTranslator(context_manager=ConversationContextManager())
    vad = VoiceActivityDetector(sample_rate=16000, profile=VADProfile.NATURAL)

    def on_subtitle(item):
        outputs.append({
            "utterance_id": item.utterance_id,
            "speaker_id": item.speaker_id,
            "start": item.start_time,
            "end": item.end_time,
            "text": item.original,
            "translation": " ".join(item.translated.split()),
            "is_complete": item.is_complete,
            "is_interrupted": item.is_interrupted,
            "final_asr_pass_used": item.final_asr_pass_used,
            "completion_reason": item.completion_reason,
        })
        latencies.append(item.total_latency * 1000.0)

    controller = PipelineController(
        audio_capture=capture,
        vad=vad,
        asr_engine=engine,
        translator=translator,
        source_language="en",
        target_language="es",
        on_subtitle=on_subtitle,
        session_id="continuous-real-65s",
        max_queue_size=240,
        speaker_tracker=SpeakerTracker(sample_rate=16000, similarity_threshold=0.91),
        on_pipeline_debug=lambda stage, speaker, data: debug_stages.append(stage),
    )

    started = time.monotonic()
    controller.start()
    if not capture.done.wait(duration + 15.0 if realtime else 30.0):
        raise TimeoutError("Continuous capture did not finish")
    controller.stop()
    elapsed = time.monotonic() - started

    evaluator = GroundTruthEvaluator()
    truth = ground_truth(cycle_seconds)
    before = evaluator.evaluate(truth, baseline_outputs(cycle_seconds))
    after = evaluator.evaluate(truth, outputs)
    latency_summary = {
        "average_end_to_translation_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "p95_end_to_translation_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
        "maximum_end_to_translation_ms": round(max(latencies), 1) if latencies else None,
    }
    report = {
        "fixture": str(SOURCE_AUDIO.relative_to(ROOT)),
        "continuous_duration_seconds": round(duration, 3),
        "repetitions": REPETITIONS,
        "stream_frame_ms": 30,
        "realtime_playback": realtime,
        "wall_time_seconds": round(elapsed, 3),
        "ground_truth_turns": len(truth),
        "system_utterances": len(outputs),
        "dropped_frames": capture.dropped,
        "capture_interval_p95_ms": round(float(np.percentile(capture.frame_intervals, 95)) * 1000.0, 2),
        "debug_stages_seen": sorted(set(debug_stages)),
        "latency": latency_summary,
        "before": asdict(before),
        "after": asdict(after),
        "outputs": outputs,
    }
    report["before"].pop("errors", None)
    report["before"].pop("der_metrics", None)
    report["after"].pop("errors", None)
    report["after"].pop("der_metrics", None)

    destination = ROOT / "benchmark_results" / "sentence_pipeline_continuous.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "duration_seconds": report["continuous_duration_seconds"],
        "utterances": len(outputs),
        "dropped_frames": capture.dropped,
        "before": report["before"],
        "after": report["after"],
        "latency": latency_summary,
        "report": str(destination),
    }, ensure_ascii=False, indent=2))

    if capture.dropped:
        raise AssertionError(f"Capture dropped {capture.dropped} frames")
    if after.cross_speaker_merge_rate != 0.0:
        raise AssertionError(f"Cross-speaker merge rate is {after.cross_speaker_merge_rate}%")
    if after.source_dialogue_coverage < 95.0:
        raise AssertionError(f"Source dialogue coverage is {after.source_dialogue_coverage}%")
    return report


if __name__ == "__main__":
    run(realtime=True)
