from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from audio_transcriber import TranscriptionDependencyError, transcribe_audio_path


TURN_HEADER_RE = re.compile(r"^(T\d{3})\b\s*(?:[^\w\s]+\s*)?(.*)$", re.IGNORECASE)
GENERATED_POINTS_SUFFIX = "_points"

@dataclass(frozen=True)
class KeywordTerm:
    keyword_id: str
    label: str
    family: str
    family_label: str
    language: str
    phrase: str
    case_sensitive: bool
    match_unit: str
    partial_words_count_as_full_hit: bool


@dataclass(frozen=True)
class TranscriptLine:
    file: str
    line_number: int
    turn_id: str
    speaker: str
    text: str


@dataclass(frozen=True)
class KeywordHit:
    evidence_type: str
    confidence: float
    file: str
    line_number: int
    turn_id: str
    speaker: str
    keyword_id: str
    keyword_label: str
    family: str
    family_label: str
    language: str
    phrase: str
    matched_text: str
    start: int
    end: int
    reason: str
    text: str


def fold_text(value: str) -> str:
    """Make text easier to compare across umlaut/accent transcription variants."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).casefold()


def load_keyword_data(keyword_file: Path) -> dict:
    return json.loads(keyword_file.read_text(encoding="utf-8-sig"))


def load_synonym_data(synonym_file: Path | None) -> dict[str, list[str]]:
    if synonym_file is None:
        return {}
    return normalize_synonym_data(json.loads(synonym_file.read_text(encoding="utf-8-sig")))


def build_keyword_terms(
    keyword_file: Path,
    languages: set[str],
    synonym_file: Path | None = None,
) -> list[KeywordTerm]:
    return build_keyword_terms_from_data(
        load_keyword_data(keyword_file),
        languages,
        synonym_data=load_synonym_data(synonym_file),
    )


def build_keyword_terms_from_data(
    data: dict,
    languages: set[str],
    synonym_data: dict[str, list[str]] | None = None,
) -> list[KeywordTerm]:
    families_by_keyword = build_family_lookup(data)
    synonym_data = synonym_data or {}
    terms: list[KeywordTerm] = []

    for keyword in data.get("keywords", []):
        if not keyword.get("enabled", True):
            continue

        keyword_id = str(keyword["id"])
        family_id, family_label = families_by_keyword.get(keyword_id, ("unassigned", "Unassigned"))
        phrases: dict[str, tuple[str, set[str]]] = {}

        for language in languages:
            for phrase in keyword.get(language, []):
                phrase = str(phrase).strip()
                if not phrase:
                    continue
                stored_phrase, stored_languages = phrases.setdefault(phrase.casefold(), (phrase, set()))
                stored_languages.add(language)

        for synonym in synonyms_for_keyword(keyword, synonym_data):
            synonym = str(synonym).strip()
            if synonym:
                phrases.setdefault(synonym.casefold(), (synonym, {"synonym"}))

        for variant in spelling_variants_for_keyword(keyword, languages):
            variant = variant.strip()
            if variant:
                phrases.setdefault(variant.casefold(), (variant, {"variant"}))

        for phrase, phrase_languages in phrases.values():
            terms.append(
                KeywordTerm(
                    keyword_id=keyword_id,
                    label=str(keyword.get("label", "")),
                    family=family_id,
                    family_label=family_label,
                    language=",".join(sorted(phrase_languages)),
                    phrase=phrase,
                    case_sensitive=bool(keyword.get("case_sensitive", False)),
                    match_unit=str(keyword.get("match_unit", "whole_phrase")),
                    partial_words_count_as_full_hit=bool(
                        keyword.get("partial_words_count_as_full_hit", False)
                    ),
                )
            )

    return terms


def build_family_lookup(data: dict) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    for family_id, family in data.get("families", {}).items():
        family_label = str(family.get("label", family_id))
        for keyword_id in family.get("keywords", []):
            lookup[str(keyword_id)] = (str(family_id), family_label)
    return lookup


def normalize_synonym_data(data) -> dict[str, list[str]]:
    """Accept flexible synonym JSON formats and return {keyword_id_or_label: [terms]}."""
    if not data:
        return {}

    if isinstance(data, dict) and isinstance(data.get("synonyms"), dict):
        data = data["synonyms"]

    if isinstance(data, dict):
        return {
            str(key): [str(value).strip() for value in values if str(value).strip()]
            for key, values in data.items()
            if isinstance(values, list)
        }

    if isinstance(data, list):
        normalized: dict[str, list[str]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or item.get("keyword_id") or item.get("label") or "").strip()
            values = item.get("synonyms") or item.get("terms") or item.get("phrases") or []
            if key and isinstance(values, list):
                normalized[key] = [str(value).strip() for value in values if str(value).strip()]
        return normalized

    return {}


def synonyms_for_keyword(keyword: dict, synonym_data: dict[str, list[str]]) -> list[str]:
    keyword_id = str(keyword.get("id", ""))
    label = str(keyword.get("label", ""))
    configured = synonym_data.get(keyword_id, []) + synonym_data.get(label, [])

    inline = []
    for field_name in ("synonyms", "synonyme", "aliases", "related"):
        values = keyword.get(field_name, [])
        if isinstance(values, list):
            inline.extend(str(value).strip() for value in values if str(value).strip())

    return dedupe_phrases([*configured, *inline])


def spelling_variants_for_keyword(keyword: dict, languages: set[str]) -> list[str]:
    variants: list[str] = []
    for language in languages:
        for phrase in keyword.get(language, []):
            phrase = str(phrase).strip()
            if phrase:
                variants.extend(make_spelling_variants(phrase))
    return dedupe_phrases(variants)


def make_spelling_variants(phrase: str) -> list[str]:
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
    }
    expanded = phrase
    for source, replacement in replacements.items():
        expanded = expanded.replace(source, replacement)

    stripped = fold_text(phrase)
    return [variant for variant in (expanded, stripped) if variant and variant != phrase]


def dedupe_phrases(phrases: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        key = phrase.casefold()
        if key not in seen:
            seen.add(key)
            result.append(phrase)
    return result


def iter_transcript_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    yield from sorted(
        transcript_path
        for transcript_path in path.glob("*.txt")
        if not transcript_path.stem.endswith(GENERATED_POINTS_SUFFIX)
    )


def prepare_transcript_for_evaluation(path: Path) -> Path:
    """Create a sentence-point transcript only for unstructured paragraph files."""
    text = path.read_text(encoding="utf-8-sig")
    if has_structured_turns(text) or path.stem.endswith(GENERATED_POINTS_SUFFIX):
        return path

    sentences = split_paragraph_into_points(text)
    if not sentences:
        return path

    prepared_path = path.with_name(f"{path.stem}{GENERATED_POINTS_SUFFIX}{path.suffix}")
    prepared_text = "\n\n".join(
        f"T{index:03d} · Text\n{sentence}" for index, sentence in enumerate(sentences, start=1)
    )

    if not prepared_path.exists() or prepared_path.read_text(encoding="utf-8-sig") != prepared_text:
        prepared_path.write_text(prepared_text, encoding="utf-8")

    return prepared_path


def has_structured_turns(text: str) -> bool:
    return any(TURN_HEADER_RE.match(line.strip()) for line in text.splitlines())


def split_paragraph_into_points(text: str) -> list[str]:
    compact_text = re.sub(r"\s+", " ", text.strip())
    if not compact_text:
        return []

    sentence_matches = re.findall(r"[^.!?]+[.!?]+[\"'»”’]*|[^.!?]+$", compact_text)
    sentences = [sentence.strip() for sentence in sentence_matches if sentence.strip()]
    return sentences or [compact_text]


def parse_transcript(path: Path) -> list[TranscriptLine]:
    lines: list[TranscriptLine] = []
    current_turn = ""
    current_speaker = ""

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        header_match = TURN_HEADER_RE.match(line)
        if header_match:
            current_turn = header_match.group(1).upper()
            current_speaker = header_match.group(2).strip()
            continue

        lines.append(
            TranscriptLine(
                file=str(path),
                line_number=line_number,
                turn_id=current_turn,
                speaker=current_speaker,
                text=line,
            )
        )

    return lines


def make_pattern(term: KeywordTerm) -> re.Pattern[str]:
    escaped = phrase_to_regex(term.phrase)
    flags = 0 if term.case_sensitive else re.IGNORECASE

    if term.match_unit == "whole_phrase" and not term.partial_words_count_as_full_hit:
        escaped = rf"(?<!\w){escaped}(?!\w)"

    return re.compile(escaped, flags)


def phrase_to_regex(phrase: str) -> str:
    """Match phrase words even if ASR inserts punctuation or variable spacing."""
    parts = [part for part in re.split(r"[\s\-_./]+", phrase.strip()) if part]
    if not parts:
        return re.escape(phrase)
    return r"[\W_]*".join(re.escape(part) for part in parts)


def find_hits(
    transcript_paths: Iterable[Path],
    terms: list[KeywordTerm],
    accent_insensitive: bool = False,
    include_context: bool = True,
    include_fuzzy: bool = True,
    context_window: int = 2,
) -> list[KeywordHit]:
    hits: list[KeywordHit] = []
    paths = [prepare_transcript_for_evaluation(path) for path in transcript_paths]

    if accent_insensitive:
        hits.extend(find_hits_accent_insensitive(paths, terms))
    else:
        patterns = [(term, make_pattern(term)) for term in terms]

        for path in paths:
            for line in parse_transcript(path):
                for term, pattern in patterns:
                    for match in pattern.finditer(line.text):
                        hits.append(hit_from_match(line, term, match.group(0), match.start(), match.end()))

    hits = dedupe_direct_hits(hits)
    if include_fuzzy:
        hits.extend(find_fuzzy_hits(paths, terms, hits))
        hits = dedupe_direct_hits(hits)

    if include_context:
        hits.extend(find_context_hits(hits, terms, context_window=context_window))

    return sorted(hits, key=lambda hit: (hit.file, hit.line_number, hit.keyword_id, hit.evidence_type))


def find_fuzzy_hits(
    transcript_paths: Iterable[Path],
    terms: list[KeywordTerm],
    existing_hits: list[KeywordHit],
) -> list[KeywordHit]:
    """Recover likely ASR near-misses without counting them as exact matches."""
    hits: list[KeywordHit] = []
    existing_keys = {(hit.file, hit.line_number, hit.keyword_id) for hit in existing_hits}
    fuzzy_terms = [
        (term, tokenize_for_fuzzy(term.phrase))
        for term in terms
        if term_is_direct(term)
    ]
    fuzzy_terms = [(term, tokens) for term, tokens in fuzzy_terms if tokens]

    for path in transcript_paths:
        for line in parse_transcript(path):
            line_tokens = tokenize_for_fuzzy(line.text)
            if not line_tokens:
                continue

            for term, phrase_tokens in fuzzy_terms:
                if (line.file, line.line_number, term.keyword_id) in existing_keys:
                    continue

                match_text, score = best_fuzzy_phrase_match(phrase_tokens, line_tokens)
                if not match_text:
                    continue

                confidence = 0.72 if len(phrase_tokens) == 1 else 0.78
                hits.append(
                    hit_from_match(
                        line,
                        term,
                        match_text,
                        0,
                        0,
                        evidence_type="synonym",
                        confidence=confidence,
                        reason=f"approximate ASR phrase match for '{term.phrase}' ({score:.0%} similar)",
                    )
                )
                existing_keys.add((line.file, line.line_number, term.keyword_id))

    return hits


def tokenize_for_fuzzy(text: str) -> list[str]:
    return re.findall(r"\w+", fold_text(text))


def best_fuzzy_phrase_match(phrase_tokens: list[str], line_tokens: list[str]) -> tuple[str, float]:
    phrase = " ".join(phrase_tokens)
    phrase_len = len(phrase_tokens)
    best_text = ""
    best_score = 0.0

    if phrase_len == 1 and len(phrase_tokens[0]) < 5:
        return "", 0.0

    window_sizes = {phrase_len}
    if phrase_len > 1:
        window_sizes.add(phrase_len + 1)

    for window_size in sorted(size for size in window_sizes if size > 0):
        if window_size > len(line_tokens):
            continue
        for index in range(0, len(line_tokens) - window_size + 1):
            window_tokens = line_tokens[index : index + window_size]
            if not plausible_fuzzy_candidate(phrase_tokens, window_tokens):
                continue
            candidate = " ".join(window_tokens)
            score = SequenceMatcher(None, phrase, candidate).ratio()
            if score > best_score:
                best_score = score
                best_text = candidate

    threshold = 0.90 if phrase_len == 1 else 0.86
    if best_score >= threshold:
        return best_text, best_score
    return "", best_score


def plausible_fuzzy_candidate(phrase_tokens: list[str], window_tokens: list[str]) -> bool:
    phrase_first = phrase_tokens[0]
    window_first = window_tokens[0]

    if len(phrase_tokens) == 1:
        candidate = window_tokens[0]
        if abs(len(candidate) - len(phrase_first)) > max(2, len(phrase_first) // 3):
            return False
        return candidate[:1] == phrase_first[:1] or candidate[:2] == phrase_first[:2]

    phrase_initials = {token[:1] for token in phrase_tokens if token}
    window_initials = {token[:1] for token in window_tokens if token}
    return bool(phrase_initials & window_initials)


def find_hits_accent_insensitive(
    transcript_paths: Iterable[Path],
    terms: list[KeywordTerm],
) -> list[KeywordHit]:
    folded_terms = []
    for term in terms:
        phrase = fold_text(term.phrase) if not term.case_sensitive else term.phrase
        escaped = phrase_to_regex(phrase)
        if term.match_unit == "whole_phrase" and not term.partial_words_count_as_full_hit:
            escaped = rf"(?<!\w){escaped}(?!\w)"
        folded_terms.append((term, re.compile(escaped)))

    for path in transcript_paths:
        for line in parse_transcript(path):
            folded_line = fold_text(line.text)
            for term, pattern in folded_terms:
                for match in pattern.finditer(folded_line):
                    yield hit_from_match(line, term, term.phrase, match.start(), match.end())


def dedupe_direct_hits(hits: list[KeywordHit]) -> list[KeywordHit]:
    priority = {"exact": 0, "synonym": 1, "context": 2}
    best_hits: dict[tuple[str, int, str, str], KeywordHit] = {}

    for hit in hits:
        key = (hit.file, hit.line_number, hit.keyword_id, fold_text(hit.matched_text))
        current = best_hits.get(key)
        if current is None or priority.get(hit.evidence_type, 9) < priority.get(current.evidence_type, 9):
            best_hits[key] = hit

    return sorted(best_hits.values(), key=lambda hit: (hit.file, hit.line_number, hit.keyword_id, hit.evidence_type))


def find_context_hits(
    direct_hits: list[KeywordHit],
    terms: list[KeywordTerm],
    context_window: int = 2,
    min_keywords: int = 2,
) -> list[KeywordHit]:
    """Generate context findings from JSON families, not predefined domain rules."""
    hits: list[KeywordHit] = []
    family_sizes = build_family_sizes(terms)
    seen: set[tuple[str, str, tuple[str, ...], tuple[int, ...]]] = set()
    eligible_hits = [
        hit
        for hit in direct_hits
        if hit.evidence_type != "context"
        and hit.family != "unassigned"
        and family_sizes.get(hit.family, 0) >= min_keywords
    ]

    for file_path in sorted({hit.file for hit in eligible_hits}):
        file_hits = [hit for hit in eligible_hits if hit.file == file_path]
        for family in sorted({hit.family for hit in file_hits}):
            family_hits = sorted(
                [hit for hit in file_hits if hit.family == family],
                key=lambda hit: (turn_number(hit), hit.line_number, hit.keyword_id),
            )
            for focal in family_hits:
                focal_turn = turn_number(focal)
                window_hits = [
                    hit
                    for hit in family_hits
                    if abs(turn_number(hit) - focal_turn) <= context_window
                ]
                keyword_ids = sorted({hit.keyword_id for hit in window_hits})
                if len(keyword_ids) < min_keywords:
                    continue

                evidence_lines = tuple(sorted({hit.line_number for hit in window_hits}))
                key = (file_path, family, tuple(keyword_ids), evidence_lines)
                if key in seen:
                    continue
                seen.add(key)

                first_hit = min(window_hits, key=lambda hit: (turn_number(hit), hit.line_number))
                representative_hits = first_hit_per_keyword(window_hits)
                matched_text = "; ".join(
                    f"{hit.keyword_id}={hit.matched_text}" for hit in representative_hits
                )
                reason = (
                    f"JSON family context: {len(keyword_ids)} different keyword IDs "
                    f"from family '{first_hit.family_label}' found within +/-{context_window} turns: "
                    + ", ".join(keyword_ids)
                )
                evidence_text = " / ".join(
                    f"{hit.turn_id}: {hit.text}" for hit in unique_hits_by_line(window_hits)
                )
                confidence = min(0.95, 0.55 + 0.1 * len(keyword_ids) + 0.03 * len(evidence_lines))

                hits.append(
                    KeywordHit(
                        evidence_type="context",
                        confidence=round(confidence, 2),
                        file=file_path,
                        line_number=first_hit.line_number,
                        turn_id=first_hit.turn_id,
                        speaker=first_hit.speaker,
                        keyword_id=f"CTX_{safe_context_id(family)}",
                        keyword_label=f"Context from JSON family: {first_hit.family_label}",
                        family=family,
                        family_label=first_hit.family_label,
                        language="context",
                        phrase=matched_text,
                        matched_text=matched_text,
                        start=0,
                        end=0,
                        reason=reason,
                        text=evidence_text,
                    )
                )

    return sorted(hits, key=lambda hit: (hit.file, hit.line_number, hit.keyword_id))


def build_family_sizes(terms: list[KeywordTerm]) -> dict[str, int]:
    family_keywords: dict[str, set[str]] = {}
    for term in terms:
        if term.family == "unassigned":
            continue
        family_keywords.setdefault(term.family, set()).add(term.keyword_id)
    return {family: len(keyword_ids) for family, keyword_ids in family_keywords.items()}


def turn_number(hit: KeywordHit) -> int:
    match = re.search(r"\d+", hit.turn_id or "")
    if match:
        return int(match.group(0))
    return hit.line_number


def first_hit_per_keyword(hits: list[KeywordHit]) -> list[KeywordHit]:
    first_by_keyword: dict[str, KeywordHit] = {}
    for hit in sorted(hits, key=lambda item: (turn_number(item), item.line_number, item.keyword_id)):
        first_by_keyword.setdefault(hit.keyword_id, hit)
    return list(first_by_keyword.values())


def unique_hits_by_line(hits: list[KeywordHit]) -> list[KeywordHit]:
    first_by_line: dict[int, KeywordHit] = {}
    for hit in sorted(hits, key=lambda item: (turn_number(item), item.line_number)):
        first_by_line.setdefault(hit.line_number, hit)
    return list(first_by_line.values())


def safe_context_id(family: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", family).strip("_")
    return cleaned.upper() or "FAMILY"


def reduce_context_hits(hits: list[KeywordHit]) -> list[KeywordHit]:
    best_by_conversation: dict[tuple[str, str], KeywordHit] = {}
    for hit in hits:
        key = (hit.file, hit.keyword_id)
        current = best_by_conversation.get(key)
        if current is None or (hit.confidence, -hit.line_number) > (current.confidence, -current.line_number):
            best_by_conversation[key] = hit

    return sorted(best_by_conversation.values(), key=lambda hit: (hit.file, hit.line_number, hit.keyword_id))


def hit_from_match(
    line: TranscriptLine,
    term: KeywordTerm,
    matched_text: str,
    start: int,
    end: int,
    evidence_type: str | None = None,
    confidence: float | None = None,
    reason: str | None = None,
) -> KeywordHit:
    hit_type = evidence_type or evidence_type_for_term(term)
    default_confidence = 1.0
    languages = set(term.language.split(","))
    if "synonym" in languages:
        default_confidence = 0.75
    elif "variant" in languages:
        default_confidence = 0.7 if len(term.phrase.split()) == 1 else 0.8
    return KeywordHit(
        evidence_type=hit_type,
        confidence=confidence if confidence is not None else default_confidence,
        file=line.file,
        line_number=line.line_number,
        turn_id=line.turn_id,
        speaker=line.speaker,
        keyword_id=term.keyword_id,
        keyword_label=term.label,
        family=term.family,
        family_label=term.family_label,
        language=term.language,
        phrase=term.phrase,
        matched_text=matched_text,
        start=start,
        end=end,
        reason=reason or f"{hit_type} phrase match",
        text=line.text,
    )


def evidence_type_for_term(term: KeywordTerm) -> str:
    languages = set(term.language.split(","))
    if languages & {"de", "gsw"}:
        return "exact"
    if languages & {"synonym", "variant"}:
        return "synonym"
    return "exact"


def term_is_direct(term: KeywordTerm) -> bool:
    return bool(set(term.language.split(",")) & {"de", "gsw"})


def write_json(hits: list[KeywordHit], output_path: Path) -> None:
    output_path.write_text(
        json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(hits: list[KeywordHit], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(hits[0]).keys()) if hits else list(KeywordHit.__dataclass_fields__))
        writer.writeheader()
        for hit in hits:
            writer.writerow(asdict(hit))


def print_human_report(hits: list[KeywordHit]) -> None:
    if not hits:
        print("No keyword hits found.")
        return

    current_file = None
    for hit in hits:
        if hit.file != current_file:
            current_file = hit.file
            print(f"\n{current_file}")

        speaker = f" - {hit.speaker}" if hit.speaker else ""
        print(
            f"  line {hit.line_number}, {hit.turn_id}{speaker}: "
            f"[{hit.evidence_type}, {hit.confidence:.2f}] "
            f"{hit.keyword_id} ({hit.keyword_label}) matched '{hit.matched_text}'"
        )
        print(f"    reason: {hit.reason}")
        print(f"    {hit.text}")

    print(f"\nTotal hits: {len(hits)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find Swiss German/German compliance keywords in transcript files."
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=Path("Transkript"),
        help="Transcript .txt file or folder to scan. Default: Transkript",
    )
    parser.add_argument(
        "--keywords",
        type=Path,
        default=Path("Stichwortliste.json"),
        help="Keyword JSON file. Default: Stichwortliste.json",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional audio file or folder to transcribe before scanning.",
    )
    parser.add_argument(
        "--audio-output",
        type=Path,
        default=Path("Transkript"),
        help="Folder where generated audio transcripts are saved. Default: Transkript",
    )
    parser.add_argument(
        "--audio-model",
        choices=["tiny", "base", "small", "medium"],
        default="small",
        help="Whisper model used for audio transcription. Default: small",
    )
    parser.add_argument(
        "--audio-language",
        choices=["de", "auto"],
        default="de",
        help="Audio transcription language. Use de for Swiss German/German or auto for detection. Default: de",
    )
    parser.add_argument(
        "--synonyms",
        type=Path,
        help="Optional synonym JSON keyed by keyword id or label.",
    )
    parser.add_argument(
        "--language",
        choices=["all", "de", "gsw"],
        default="all",
        help="Keyword language variants to use. Default: all",
    )
    parser.add_argument(
        "--accent-insensitive",
        action="store_true",
        help="Also tolerate umlaut/accent differences such as oe/o. Positions are approximate in this mode.",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Disable synonym/context review signals and report only the original keyword phrases.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Number of neighboring transcript turns used for context signals. Default: 2",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only print/export hits at or above this confidence. Example: 0.7",
    )
    parser.add_argument("--json-out", type=Path, help="Optional path for JSON result output.")
    parser.add_argument("--csv-out", type=Path, help="Optional path for CSV result output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.keywords.exists():
        print(f"Keyword file not found: {args.keywords}", file=sys.stderr)
        return 2
    if not args.transcripts.exists() and not args.audio:
        print(f"Transcript path not found: {args.transcripts}", file=sys.stderr)
        return 2
    if args.audio and not args.audio.exists():
        print(f"Audio path not found: {args.audio}", file=sys.stderr)
        return 2

    languages = {"de", "gsw"} if args.language == "all" else {args.language}
    terms = build_keyword_terms(args.keywords, languages, synonym_file=args.synonyms)
    if args.direct_only:
        terms = [term for term in terms if term_is_direct(term)]

    transcript_paths = list(iter_transcript_paths(args.transcripts)) if args.transcripts.exists() else []

    if args.audio:
        try:
            generated_paths = transcribe_audio_path(
                args.audio,
                args.audio_output,
                model_name=args.audio_model,
                language=args.audio_language,
            )
        except TranscriptionDependencyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        transcript_paths.extend(generated_paths)

    transcript_paths = list(dict.fromkeys(transcript_paths))
    if not transcript_paths:
        print("No transcript files found to scan.", file=sys.stderr)
        return 2

    hits = find_hits(
        transcript_paths,
        terms,
        accent_insensitive=args.accent_insensitive,
        include_context=not args.direct_only,
        include_fuzzy=not args.direct_only,
        context_window=max(0, args.context_window),
    )
    hits = [hit for hit in hits if hit.confidence >= args.min_confidence]

    print_human_report(hits)

    if args.json_out:
        write_json(hits, args.json_out)
    if args.csv_out:
        write_csv(hits, args.csv_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
