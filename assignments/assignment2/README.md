# RAV4 Video Detections

This dataset contains object detection results from a Toyota RAV4 exterior review video (YouTube ID: `YcvECxtXoxQ`), used for image-to-video semantic retrieval.

## Files

- **video_detections.parquet** — Frame-level detections from the input video
- **retrieval_results.parquet** — Query-to-video retrieval results

## Schema: video_detections.parquet

| Column | Type | Description |
|---|---|---|
| `video_id` | string | YouTube video ID |
| `frame_index` | int | Frame number |
| `timestamp_sec` | int | Timestamp in seconds (sampled at 1fps/5s) |
| `class_label` | string | Detected car part (e.g. hood, wheel, front_bumper) |
| `confidence_score` | float | Detection confidence (0–1) |
| `x_min` | float | Bounding box left |
| `y_min` | float | Bounding box top |
| `x_max` | float | Bounding box right |
| `y_max` | float | Bounding box bottom |

## Schema: retrieval_results.parquet

| Column | Type | Description |
|---|---|---|
| `query_timestamp` | string | Timestamp of query image in source video |
| `query_classes` | string | Car part classes detected in query image |
| `start_timestamp` | int | Start of matching video segment (seconds) |
| `end_timestamp` | int | End of matching video segment (seconds) |
| `number_of_supporting_detections` | int | Number of detections supporting the match |

## Detector

YOLOv8n-seg fine-tuned on the Ultralytics Car Parts Segmentation dataset (23 classes). Frames sampled at 1 frame every 5 seconds.
