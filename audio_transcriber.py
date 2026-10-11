from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".wav", ".wma"}
MODEL_CACHE = {}


class TranscriptionDependencyError(RuntimeError):
    pass


def list_audio_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.casefold() in AUDIO_EXTENSIONS else []

    if not path.exists():
        return []

    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.casefold() in AUDIO_EXTENSIONS
    )


def transcribe_audio_path(
    audio_path: Path,
    output_dir: Path,
    model_name: str = "tiny",
    language: str = "de",
) -> list[Path]:
    audio_files = list_audio_files(audio_path)
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in: {audio_path}")

    return transcribe_audio_files(audio_files, output_dir, model_name=model_name, language=language)


def transcribe_audio_files(
    audio_files: Iterable[Path],
    output_dir: Path,
    model_name: str = "tiny",
    language: str = "de",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_whisper_model(model_name)

    generated_paths: list[Path] = []
    for audio_file in audio_files:
        generated_paths.append(
            transcribe_audio_file_with_model(audio_file, output_dir, model, language=language)
        )

    return generated_paths


def load_whisper_model(model_name: str):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise TranscriptionDependencyError(
            "Audio transcription needs faster-whisper. Install it with: "
            f'"{sys.executable}" -m pip install faster-whisper'
        ) from exc

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    MODEL_CACHE[model_name] = model
    return model


def transcribe_audio_file_with_model(
    audio_file: Path,
    output_dir: Path,
    model,
    language: str = "de",
) -> Path:
    patch_pyav_metadata_errors()
    language_code = None if language == "auto" else language
    try:
        segments, _info = model.transcribe(
            str(audio_file),
            language=language_code,
            vad_filter=True,
            beam_size=1,
        )
    except Exception as exc:
        if "silero_vad" not in str(exc) and "NO_SUCHFILE" not in str(exc):
            raise
        segments, _info = model.transcribe(
            str(audio_file),
            language=language_code,
            vad_filter=False,
            beam_size=1,
        )

    transcript_blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        if not text:
            continue
        transcript_blocks.append(
            f"T{index:03d} - Audio\n"
            f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] {text}"
        )

    if not transcript_blocks:
        transcript_blocks.append("T001 - Audio\n[No speech detected]")

    output_path = output_dir / f"{audio_file.stem}_audio_transcript.txt"
    output_path.write_text("\n\n".join(transcript_blocks), encoding="utf-8")
    return output_path


def patch_pyav_metadata_errors() -> None:
    """Keep faster-whisper compatible with PyAV versions that removed metadata_errors."""
    try:
        import av
    except ModuleNotFoundError:
        return

    current_open = av.open
    if getattr(current_open, "_metadata_errors_patched", False):
        return

    def open_without_metadata_errors(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return current_open(*args, **kwargs)

    open_without_metadata_errors._metadata_errors_patched = True
    av.open = open_without_metadata_errors


def format_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    milliseconds = total_milliseconds % 1000
    total_seconds = total_milliseconds // 1000
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"
