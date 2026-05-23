# Queer-Signal-Extraction-Framework
## Goal
Build a two-stage pipeline for extracting a segment-level Queer Visibility Index (QVI) from street imagery.

## Stage 1: GVI / Panorama Collector
### Purpose
Collect 360 panorama images or nearby street-level imagery for each road segment or segment midpoint.

### Input arguments
- `segment_id` or segment table
- `lat`, `lng` or segment centroid geometry
- `provider` = `google_street_view` | `mapillary` | `local_cache` | custom adapter
- `api_key` or access token, preferably from `.env` as `GVI_API_KEY` or `GOOGLE_STREET_VIEW_API_KEY`
- `heading_list` = list of camera directions, e.g. `0,90,180,270`
- `pitch` = camera pitch
- `fov` = field of view
- `radius` = search radius around the segment midpoint
- `zoom` or image size
- `output_dir`
- `manifest_path`
- `year_filter` or nearest-date rule if metadata is available

### Expected behavior
1. Read segment geometry or centroid.
2. Generate panorama requests for the requested headings.
3. Save images to a segment-specific directory.
4. Write a manifest with image metadata.
5. Keep raw imagery separate from derived analysis outputs.

### Suggested manifest schema
```json
{
  "segment_id": "S001",
  "image_id": "S001_0",
  "source": "google_street_view",
  "lat": 34.0001,
  "lng": -118.0002,
  "heading": 0,
  "pitch": 0,
  "fov": 90,
  "captured_at": "2021-06-01",
  "file_path": "images/S001/S001_0.jpg"
}
```

### Output
- Image files under `images/`
- JSON or CSV manifest
- Optional download log

## Stage 2: LM-based QVI Raw Data Generator
### Purpose
Run Gemini or another multimodal LM over the collected images and convert visual cues into structured JSON annotations.

### Input arguments
- `image_dir`
- `manifest_path`
- `prompt_template_path`
- `model_name` = `gemini-*`
- `schema_path` for JSON output control
- `api_key`, preferably from `.env` as `GEMINI_API_KEY`
- `batch_size`
- `temperature`
- `output_jsonl`
- `output_table`

### Expected behavior
1. Read the manifest and discover images.
2. Load a fixed prompt template.
3. Send each image, or image group per segment, to Gemini.
4. Force structured output as JSON.
5. Validate the JSON against the schema.
6. Aggregate image-level outputs into segment-level raw QVI records.

### Suggested JSON fields
- `segment_id`
- `image_id`
- `rainbow_signal_present`
- `inclusive_signage_present`
- `active_frontage_score`
- `façade_openness_score`
- `queer_visibility_score`
- `confidence`
- `rationale_short`
- `raw_model_response`

### Example output row
```json
{
  "segment_id": "S001",
  "image_id": "S001_0",
  "queer_visibility_score": 0.72,
  "rainbow_signal_present": true,
  "inclusive_signage_present": true,
  "active_frontage_score": 4,
  "façade_openness_score": 3,
  "confidence": 0.81
}
```

## Recommended Folder Layout
```text
QueerSignalExtraction/
  data/
    raw/
    interim/
    processed/
  prompts/
    qvi_prompt.md
  schemas/
    qvi_annotation.schema.json
  src/
    collector/
    annotator/
    utils/
  outputs/
    manifest/
    annotations/
    qvi_raw/
```

## Implementation Notes
- Treat the panorama collector as a pluggable adapter so a Google Street View version can be replaced later.
- Keep raw images and derived annotations separate.
- Use JSONL for model outputs and convert to CSV or Parquet only after validation.
- Keep the prompt stable across batches to reduce annotation drift.
- Prefer interpretable variables over free-form safety judgments.

## Minimal Next Build Step
1. Create a collector CLI that reads a segment table and writes an image manifest.
2. Create an annotator CLI that reads the manifest and writes QVI JSONL.
3. Add a schema validator for the LM output.
4. Add a small sample run on 5 to 10 segments.
