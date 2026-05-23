from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


@dataclass
class ManifestRecord:
    segment_id: str
    image_id: str
    source: str
    lat: float
    lng: float
    heading: int
    pitch: int
    fov: int
    captured_at: str | None
    file_path: str


@dataclass
class AnnotationRecord:
    segment_id: str
    image_id: str
    scene_id: str
    environment_type: str
    overall_queer_visibility_score: int
    overall_confidence: float
    commercial_queer_signal_count: int
    symbolic_density_score: int
    pedestrian_activity_level: int
    rainbow_signal_present: bool
    inclusive_signage_present: bool
    active_frontage_score: int
    facade_openness_score: int
    queer_visibility_score: float
    confidence: float
    rationale_short: str
    raw_model_response: str


def load_env_file(path: Path | None = None) -> None:
    candidate_paths: list[Path] = []
    if path is not None:
        candidate_paths.append(path)
    candidate_paths.append(Path.cwd() / ".env")
    candidate_paths.append(Path(__file__).resolve().parents[2] / ".env")

    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break


def load_manifest(path: Path) -> list[ManifestRecord]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_manifest_jsonl(path)
    if suffix == ".csv":
        return load_manifest_csv(path)
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def load_manifest_jsonl(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append(
                ManifestRecord(
                    segment_id=str(item["segment_id"]),
                    image_id=str(item["image_id"]),
                    source=str(item["source"]),
                    lat=float(item["lat"]),
                    lng=float(item["lng"]),
                    heading=int(item["heading"]),
                    pitch=int(item["pitch"]),
                    fov=int(item["fov"]),
                    captured_at=item.get("captured_at"),
                    file_path=str(item["file_path"]),
                )
            )
    return records


def load_manifest_csv(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                ManifestRecord(
                    segment_id=str(row["segment_id"]),
                    image_id=str(row["image_id"]),
                    source=str(row["source"]),
                    lat=float(row["lat"]),
                    lng=float(row["lng"]),
                    heading=int(row["heading"]),
                    pitch=int(row["pitch"]),
                    fov=int(row["fov"]),
                    captured_at=row.get("captured_at") or None,
                    file_path=str(row["file_path"]),
                )
            )
    return records


def load_prompt_template(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    instruction = str(payload.get("instruction", ""))
    output_format = payload.get("output_format", {})
    json_keys = payload.get("json_keys", [])
    prompt_lines = [instruction.strip()]
    prompt_lines.append("Return structured JSON only.")
    prompt_lines.append(f"Required keys: {json.dumps(json_keys, ensure_ascii=False)}")
    prompt_lines.append(f"Output format reference: {json.dumps(output_format, ensure_ascii=False)}")
    return "\n\n".join(line for line in prompt_lines if line)


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_batches(items: list[ManifestRecord], batch_size: int) -> list[list[ManifestRecord]]:
    if batch_size <= 0:
        return [items]
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def resolve_image_path(image_dir: Path, record: ManifestRecord) -> Path:
    file_path = Path(record.file_path)
    if file_path.is_absolute():
        return file_path
    candidate = image_dir / file_path
    if candidate.exists():
        return candidate
    return image_dir.parent / file_path


def compose_prompt(prompt_template: str, record: ManifestRecord) -> str:
    metadata = {
        "segment_id": record.segment_id,
        "image_id": record.image_id,
        "scene_id": record.image_id,
        "source": record.source,
        "heading": record.heading,
        "pitch": record.pitch,
        "fov": record.fov,
        "captured_at": record.captured_at,
    }
    return "\n\n".join(
        [
            prompt_template,
            f"Scene metadata: {json.dumps(metadata, ensure_ascii=False)}",
            "Return JSON without markdown fences.",
        ]
    )


def build_gemini_request(api_key: str, model_name: str, prompt_text: str, image_bytes: bytes, mime_type: str, temperature: float) -> urllib.request.Request:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model_name, safe='')}:generateContent?key={urllib.parse.quote(api_key)}"
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("ascii")}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    return urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")


def call_gemini(api_key: str, model_name: str, prompt_text: str, image_bytes: bytes, mime_type: str, temperature: float) -> dict[str, Any]:
    request = build_gemini_request(api_key, model_name, prompt_text, image_bytes, mime_type, temperature)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        message = f"Gemini API request failed with HTTP {exc.code}"
        if body:
            message = f"{message}: {body[:1000]}"
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc


def extract_text_response(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response did not contain candidates.")
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    if not text.strip():
        raise ValueError("Gemini response did not contain text content.")
    return text


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Model response did not contain a JSON object.")
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(cleaned[start:])
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON was not an object.")
    return payload


def validate_payload(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    required = schema.get("required", [])
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing required key: {key}")


def summarize_evidence(evidence_summary: list[Any]) -> str:
    pieces = [str(item).strip() for item in evidence_summary if str(item).strip()]
    return "; ".join(pieces[:3])


def normalize_record(record: ManifestRecord, payload: dict[str, Any]) -> AnnotationRecord:
    inclusive_infrastructure = payload.get("inclusive_infrastructure", {}) or {}
    commercial = payload.get("commercial_queer_signals", {}) or {}
    symbolic = payload.get("symbolic_visibility", {}) or {}
    context = payload.get("streetscape_context", {}) or {}
    overall_score = int(payload.get("overall_queer_visibility_score", 0))
    overall_confidence = float(payload.get("overall_confidence", 0.0))
    rainbow_signal_present = any(
        bool(inclusive_infrastructure.get(key))
        for key in ["rainbow_flag", "progress_pride_flag", "trans_flag"]
    )
    inclusive_signage_present = any(bool(value) for value in inclusive_infrastructure.values())
    active_frontage_score = int(context.get("pedestrian_activity_level", 0))
    facade_openness_score = int(context.get("perceived_social_openness", 0))
    evidence_summary = payload.get("evidence_summary", []) or []
    return AnnotationRecord(
        segment_id=record.segment_id,
        image_id=record.image_id,
        scene_id=str(payload.get("scene_id", record.image_id)),
        environment_type=str(payload.get("environment_type", "unknown")),
        overall_queer_visibility_score=overall_score,
        overall_confidence=overall_confidence,
        commercial_queer_signal_count=int(commercial.get("count", 0)),
        symbolic_density_score=int(symbolic.get("symbolic_density_score", 0)),
        pedestrian_activity_level=int(context.get("pedestrian_activity_level", 0)),
        rainbow_signal_present=rainbow_signal_present,
        inclusive_signage_present=inclusive_signage_present,
        active_frontage_score=active_frontage_score,
        facade_openness_score=facade_openness_score,
        queer_visibility_score=float(overall_score),
        confidence=float(overall_confidence),
        rationale_short=summarize_evidence(evidence_summary),
        raw_model_response=json.dumps(payload, ensure_ascii=False),
    )


def build_annotation_records(
    manifest_records: list[ManifestRecord],
    image_dir: Path,
    prompt_template: str,
    schema: dict[str, Any],
    model_name: str,
    api_key: str,
    temperature: float,
    batch_size: int,
    logger: logging.Logger,
) -> list[AnnotationRecord]:
    annotations: list[AnnotationRecord] = []
    batches = iter_batches(manifest_records, batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        logger.info("Processing batch %s/%s with %s images", batch_index, len(batches), len(batch))
        for record in batch:
            image_path = resolve_image_path(image_dir, record)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            prompt_text = compose_prompt(prompt_template, record)
            image_bytes = image_path.read_bytes()
            response_payload = call_gemini(
                api_key=api_key,
                model_name=model_name,
                prompt_text=prompt_text,
                image_bytes=image_bytes,
                mime_type=guess_mime_type(image_path),
                temperature=temperature,
            )
            response_text = extract_text_response(response_payload)
            parsed_payload = parse_json_object(response_text)
            validate_payload(parsed_payload, schema)
            annotations.append(normalize_record(record, parsed_payload))
    return annotations


def process_annotation_batch(
    batch: list[ManifestRecord],
    image_dir: Path,
    prompt_template: str,
    schema: dict[str, Any],
    model_name: str,
    api_key: str,
    temperature: float,
) -> list[AnnotationRecord]:
    batch_annotations: list[AnnotationRecord] = []
    for record in batch:
        image_path = resolve_image_path(image_dir, record)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        prompt_text = compose_prompt(prompt_template, record)
        image_bytes = image_path.read_bytes()
        response_payload = call_gemini(
            api_key=api_key,
            model_name=model_name,
            prompt_text=prompt_text,
            image_bytes=image_bytes,
            mime_type=guess_mime_type(image_path),
            temperature=temperature,
        )
        response_text = extract_text_response(response_payload)
        parsed_payload = parse_json_object(response_text)
        validate_payload(parsed_payload, schema)
        batch_annotations.append(normalize_record(record, parsed_payload))
    return batch_annotations


def write_jsonl(records: list[AnnotationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False))
            handle.write("\n")


def append_jsonl(records: list[AnnotationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False))
            handle.write("\n")


def load_annotations_jsonl(path: Path) -> list[AnnotationRecord]:
    if not path.exists():
        return []
    records: list[AnnotationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append(
                AnnotationRecord(
                    segment_id=str(item["segment_id"]),
                    image_id=str(item["image_id"]),
                    scene_id=str(item.get("scene_id", item["image_id"])),
                    environment_type=str(item.get("environment_type", "unknown")),
                    overall_queer_visibility_score=int(item.get("overall_queer_visibility_score", 0)),
                    overall_confidence=float(item.get("overall_confidence", 0.0)),
                    commercial_queer_signal_count=int(item.get("commercial_queer_signal_count", 0)),
                    symbolic_density_score=int(item.get("symbolic_density_score", 0)),
                    pedestrian_activity_level=int(item.get("pedestrian_activity_level", 0)),
                    rainbow_signal_present=bool(item.get("rainbow_signal_present", False)),
                    inclusive_signage_present=bool(item.get("inclusive_signage_present", False)),
                    active_frontage_score=int(item.get("active_frontage_score", 0)),
                    facade_openness_score=int(item.get("facade_openness_score", 0)),
                    queer_visibility_score=float(item.get("queer_visibility_score", 0.0)),
                    confidence=float(item.get("confidence", 0.0)),
                    rationale_short=str(item.get("rationale_short", "")),
                    raw_model_response=str(item.get("raw_model_response", "")),
                )
            )
    return records


def write_summary_table(records: list[AnnotationRecord], path: Path) -> None:
    grouped: dict[str, list[AnnotationRecord]] = {}
    for record in records:
        grouped.setdefault(record.segment_id, []).append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "segment_id",
                "image_count",
                "mean_queer_visibility_score",
                "mean_confidence",
            ],
        )
        writer.writeheader()
        for segment_id, items in grouped.items():
            writer.writerow(
                {
                    "segment_id": segment_id,
                    "image_count": len(items),
                    "mean_queer_visibility_score": round(mean(item.queer_visibility_score for item in items), 3),
                    "mean_confidence": round(mean(item.confidence for item in items), 3),
                }
            )


def run_annotator(args: argparse.Namespace) -> int:
    load_env_file()
    manifest_path = Path(args.manifest_path)
    image_dir = Path(args.image_dir)
    prompt_template_path = Path(args.prompt_template_path)
    schema_path = Path(args.schema_path)
    output_jsonl = Path(args.output_jsonl)
    output_table = Path(args.output_table)
    api_key = (
        args.api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_STREET_VIEW_API_KEY")
    )
    if not api_key:
        raise ValueError("Missing API key. Set GEMINI_API_KEY, GOOGLE_GENAI_API_KEY, GOOGLE_API_KEY, or GOOGLE_STREET_VIEW_API_KEY in .env, or pass --api-key.")
    prompt_template = load_prompt_template(prompt_template_path)
    schema = load_schema(schema_path)
    manifest_records = load_manifest(manifest_path)
    logger = logging.getLogger("annotator.run")
    output_jsonl.unlink(missing_ok=True)
    output_table.unlink(missing_ok=True)
    batches = iter_batches(manifest_records, args.batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        logger.info("Processing batch %s/%s with %s images", batch_index, len(batches), len(batch))
        batch_annotations = process_annotation_batch(
            batch=batch,
            image_dir=image_dir,
            prompt_template=prompt_template,
            schema=schema,
            model_name=args.model_name,
            api_key=api_key,
            temperature=args.temperature,
        )
        append_jsonl(batch_annotations, output_jsonl)
        current_annotations = load_annotations_jsonl(output_jsonl)
        write_summary_table(current_annotations, output_table)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotator")
    parser.add_argument("--image-dir", required=True, help="Directory containing the collected street imagery.")
    parser.add_argument("--manifest-path", required=True, help="JSONL or CSV manifest produced by the collector stage.")
    parser.add_argument("--prompt-template-path", required=True, help="Prompt template used to structure Gemini output.")
    parser.add_argument("--model-name", required=True, help="Gemini model name, for example gemini-2.0-flash.")
    parser.add_argument("--schema-path", required=True, help="JSON schema that constrains the model output.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of images or image groups sent per batch.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature for the LM call.")
    parser.add_argument("--output-jsonl", required=True, help="Path for raw image-level QVI annotations in JSONL format.")
    parser.add_argument("--output-table", required=True, help="Path for the segment-level summary table.")
    parser.add_argument("--api-key", help="Gemini API key. Defaults to GEMINI_API_KEY, GOOGLE_GENAI_API_KEY, GOOGLE_API_KEY, or GOOGLE_STREET_VIEW_API_KEY in .env.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_annotator(args)


if __name__ == "__main__":
    raise SystemExit(main())
