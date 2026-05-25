from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class SegmentRecord:
    segment_id: str
    lat: float
    lng: float


@dataclass
class ImageManifestRecord:
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
    status: str = "pending"
    error_message: str | None = None
    image_url: str | None = None
    scale: int = 2


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


def parse_heading_list(raw_value: str) -> list[int]:
    values = [part.strip() for part in raw_value.split(",")]
    return [int(value) for value in values if value]


def load_segments(path: Path) -> list[SegmentRecord]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return deduplicate_segments(load_segments_from_csv(path))
    if suffix in {".json", ".geojson"}:
        return deduplicate_segments(load_segments_from_json(path))
    raise ValueError(f"Unsupported segment file format: {path.suffix}")


def load_segments_from_csv(path: Path) -> list[SegmentRecord]:
    records: list[SegmentRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            segment_id = str(
                row.get("segment_id")
                or row.get("segment_id_num")
                or row.get("segment_name")
                or row.get("Full Name")
                or row.get("Street Name")
                or f"segment_{index}"
            )
            lat_value = row.get("lat") or row.get("seg_lat") or row.get("latitude")
            lng_value = row.get("lng") or row.get("seg_lon") or row.get("lon") or row.get("longitude")
            if lat_value is None or lng_value is None:
                raise ValueError(f"Missing coordinate columns in {path.name} at row {index}")
            records.append(
                SegmentRecord(
                    segment_id=segment_id,
                    lat=float(lat_value),
                    lng=float(lng_value),
                )
            )
    return records


def load_segments_from_json(path: Path) -> list[SegmentRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("segments", [])
    else:
        items = payload
    records: list[SegmentRecord] = []
    for item in items:
        records.append(
            SegmentRecord(
                segment_id=str(item["segment_id"]),
                lat=float(item["lat"]),
                lng=float(item["lng"]),
            )
        )
    return records


def deduplicate_segments(records: list[SegmentRecord]) -> list[SegmentRecord]:
    seen: set[tuple[str, float, float]] = set()
    unique_records: list[SegmentRecord] = []
    for record in records:
        key = (record.segment_id, round(record.lat, 6), round(record.lng, 6))
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(record)
    return unique_records


def build_manifest_records(
    segments: Iterable[SegmentRecord],
    provider: str,
    headings: list[int],
    pitch: int,
    fov: int,
    output_dir: Path,
    captured_at: str | None,
    image_size: str,
    scale: int = 1,
) -> list[ImageManifestRecord]:
    records: list[ImageManifestRecord] = []
    for segment in segments:
        segment_dir = output_dir / segment.segment_id
        for heading in headings:
            image_id = f"{segment.segment_id}_{heading}"
            file_path = segment_dir / f"{image_id}.jpg"
            image_url = build_street_view_url(
                lat=segment.lat,
                lng=segment.lng,
                heading=heading,
                pitch=pitch,
                fov=fov,
                size=image_size,
                scale=scale,
            )
            records.append(
                ImageManifestRecord(
                    segment_id=segment.segment_id,
                    image_id=image_id,
                    source=provider,
                    lat=segment.lat,
                    lng=segment.lng,
                    heading=heading,
                    pitch=pitch,
                    fov=fov,
                    scale=scale,
                    captured_at=captured_at,
                    file_path=str(file_path),
                    image_url=image_url,
                )
            )
    return records


def build_street_view_url(lat: float, lng: float, heading: int, pitch: int, fov: int, size: str, scale: int = 1) -> str:
    params = {
        "location": f"{lat},{lng}",
        "heading": str(heading),
        "pitch": str(pitch),
        "fov": str(fov),
        "size": size,
        "scale": str(scale),
    }
    query = urllib.parse.urlencode(params)
    return f"https://maps.googleapis.com/maps/api/streetview?{query}"


def download_street_view_image(url: str, api_key: str) -> bytes:
    separator = "&" if "?" in url else "?"
    request_url = f"{url}{separator}key={urllib.parse.quote(api_key)}"
    request = urllib.request.Request(request_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type.lower():
            body = response.read().decode("utf-8", errors="ignore")
            raise ValueError(f"Street View response was not an image: {body[:200]}")
        return response.read()


def write_image_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def collect_images(records: list[ImageManifestRecord], api_key: str, provider: str, logger: object) -> list[ImageManifestRecord]:
    completed: list[ImageManifestRecord] = []
    for record in records:
        try:
            if provider != "google_street_view":
                raise NotImplementedError(f"Unsupported provider: {provider}")
            image_bytes = download_street_view_image(record.image_url or "", api_key)
            write_image_file(Path(record.file_path), image_bytes)
            completed.append(
                ImageManifestRecord(
                    **{**asdict(record), "status": "downloaded", "error_message": None}
                )
            )
            logger.info("Downloaded %s", record.image_id)
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, NotImplementedError, OSError) as exc:
            logger.exception("Failed to download %s: %s", record.image_id, exc)
            completed.append(
                ImageManifestRecord(
                    **{**asdict(record), "status": "failed", "error_message": str(exc)}
                )
            )
    return completed


def write_manifest_jsonl(records: Iterable[ImageManifestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False))
            handle.write("\n")


def run_collector(args: argparse.Namespace) -> int:
    load_env_file()
    segment_path = Path(args.segment_file)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)
    api_key = args.api_key or os.getenv("GSV_API_KEY") or os.getenv("GOOGLE_STREET_VIEW_API_KEY") or os.getenv("GSV_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing API key. Set GSV_API_KEY, GOOGLE_STREET_VIEW_API_KEY, or GS_API_KEY in .env, or pass --api-key.")
    segments = load_segments(segment_path)
    headings = parse_heading_list(args.headings)
    records = build_manifest_records(
        segments=segments,
        provider=args.provider,
        headings=headings,
        pitch=args.pitch,
        fov=args.fov,
        output_dir=output_dir,
        captured_at=args.captured_at,
        image_size=args.image_size,
        scale=args.scale,
    )
    _ = api_key
    logger = logging.getLogger("collector.run")
    completed_records = collect_images(records, api_key=api_key, provider=args.provider, logger=logger)
    write_manifest_jsonl(completed_records, manifest_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="collector")
    parser.add_argument("--segment-file", required=True, help="CSV/JSON segment table with segment_id, lat, lng columns.")
    parser.add_argument("--provider", required=True, help="Image source adapter name such as google_street_view or mapillary.")
    parser.add_argument("--api-key", help="API key for the image provider. Defaults to GSV_API_KEY, GOOGLE_STREET_VIEW_API_KEY, or GEMINI_API_KEY in .env.")
    parser.add_argument("--headings", default="0,90,180,270", help="Comma-separated camera headings to sample around each segment midpoint.")
    parser.add_argument("--pitch", type=int, default=0, help="Camera pitch in degrees.")
    parser.add_argument("--fov", type=int, default=90, help="Field of view in degrees.")
    parser.add_argument("--image-size", default="640x640", help="Street View image size, for example 640x640.")
    parser.add_argument("--scale", type=int, choices=[1, 2], default=1, help="Image scale multiplier for Street View (`1` or `2`). Use with awareness of Google API limits.")
    parser.add_argument("--output-dir", required=True, help="Directory where raw images will be written.")
    parser.add_argument("--manifest-path", required=True, help="Path for the JSONL manifest that records image metadata.")
    parser.add_argument("--captured-at", help="Optional capture date string to store in the manifest.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_collector(args)


if __name__ == "__main__":
    raise SystemExit(main())
