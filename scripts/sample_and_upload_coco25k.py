#!/usr/bin/env python3
"""
Sample 25k images from COCO 2017 train using coco-minitrain and upload to HuggingFace.

Optimized workflow - downloads only what is needed:
  1. Clear existing aegean-ai/coco-25k HF dataset
  2. Download COCO 2017 train annotations (~240 MB) if not present
  3. Run sample_coco.py (annotations only - no images needed for sampling)
  4. Download only the 25k sampled images from COCO image servers
  5. Upload images + annotation JSON to aegean-ai/coco-25k

Run inside the pytorch docker container:
  python scripts/sample_and_upload_coco25k.py

  Or with an existing COCO annotations dir to skip download:
  python scripts/sample_and_upload_coco25k.py --coco_path /path/to/coco

Requirements:
  pip install huggingface_hub pycocotools scikit-image tqdm requests

Environment:
  HF_TOKEN must be set (read from .env or environment)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from tqdm import tqdm


def hf_with_retry(fn, *args, max_retries=8, base_wait=30, **kwargs):
    """Call an HF API function with exponential backoff on 429 rate limit errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except HfHubHTTPError as e:
            if "429" not in str(e) or attempt == max_retries - 1:
                raise
            wait = base_wait * (2 ** attempt)
            print(f"  Rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait}s ...")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")


REPO_ID = "aegean-ai/coco-25k"
REPO_TYPE = "dataset"

# Path to coco-minitrain src within the container
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspaces/eng-ai-agents"))
MINITRAIN_SRC = WORKSPACE / "coco-minitrain" / "src"

COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_IMAGE_BASE_URL = "http://images.cocodataset.org/train2017"


# ---------------------------------------------------------------------------
# Step 1: Clear HF dataset
# ---------------------------------------------------------------------------

def clear_hf_dataset(api: HfApi) -> None:
    """Delete and recreate the HF dataset repo to clear all content."""
    print(f"\n[1/5] Clearing {REPO_ID} by deleting and recreating the repo ...")
    hf_with_retry(api.delete_repo, repo_id=REPO_ID, repo_type=REPO_TYPE, missing_ok=True)
    hf_with_retry(api.create_repo, repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)
    print("  Done.")


# ---------------------------------------------------------------------------
# Step 2: Ensure COCO annotations are present (no images needed for sampling)
# ---------------------------------------------------------------------------

def ensure_coco_annotations(coco_path: Path) -> None:
    """Download COCO 2017 train annotations if not present at coco_path/annotations/."""
    ann_file = coco_path / "annotations" / "instances_train2017.json"
    if ann_file.exists():
        print(f"\n[2/5] Annotations found at {ann_file}")
        return

    import zipfile

    print(f"\n[2/5] Downloading COCO 2017 annotations to {coco_path} ...")
    coco_path.mkdir(parents=True, exist_ok=True)
    zip_path = coco_path / "annotations_trainval2017.zip"

    print(f"  Fetching {COCO_ANNOTATIONS_URL} ...")
    with requests.get(COCO_ANNOTATIONS_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="  Downloading"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print("  Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(coco_path)
    zip_path.unlink()
    print(f"  Annotations ready at {coco_path / 'annotations'}")


# ---------------------------------------------------------------------------
# Step 3: Run sampling (only needs annotation JSON, not images)
# ---------------------------------------------------------------------------

def run_sampling(
    coco_path: Path,
    output_json: Path,
    sample_count: int,
    run_count: int,
    debug: bool,
) -> None:
    """Run sample_coco.py to generate the minicoco annotation JSON."""
    if output_json.exists():
        print(f"\n[3/5] Sampled JSON already exists at {output_json}, skipping.")
        return

    output_json.parent.mkdir(parents=True, exist_ok=True)
    # sample_coco.py appends .json automatically - pass stem only
    save_file = str(output_json.parent / output_json.stem)

    cmd = [
        sys.executable,
        str(MINITRAIN_SRC / "sample_coco.py"),
        "--coco_path", str(coco_path),
        "--save_file_name", save_file,
        "--save_format", "json",
        "--sample_image_count", str(sample_count),
        "--run_count", str(run_count),
    ]
    if debug:
        cmd.append("--debug")

    print(f"\n[3/5] Sampling {sample_count} images (run_count={run_count}) ...")
    env = {**os.environ, "PYTHONPATH": str(MINITRAIN_SRC)}
    subprocess.check_call(cmd, env=env)
    print(f"  Sampled JSON written to {output_json}")


# ---------------------------------------------------------------------------
# Step 4: Download only the sampled images from COCO servers
# ---------------------------------------------------------------------------

def _download_one(args):
    """Download a single COCO image. Returns (filename, dest_path, ok, error)."""
    filename, dest_path = args
    if dest_path.exists():
        return filename, dest_path, True, None
    url = f"{COCO_IMAGE_BASE_URL}/{filename}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest_path.write_bytes(r.content)
        return filename, dest_path, True, None
    except Exception as e:
        return filename, dest_path, False, str(e)


def download_sampled_images(
    annotation_json: Path,
    images_dir: Path,
    workers: int = 16,
) -> list:
    """Download the 25k sampled images from the COCO image server."""
    print(f"\n[4/5] Loading sampled image list from {annotation_json} ...")
    with open(annotation_json) as f:
        data = json.load(f)

    images_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (img["file_name"], images_dir / img["file_name"])
        for img in data["images"]
    ]

    already = sum(1 for _, p in tasks if p.exists())
    print(f"  {already}/{len(tasks)} already downloaded. Fetching the rest ...")

    to_fetch = [(fn, p) for fn, p in tasks if not p.exists()]
    failed = []

    if to_fetch:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_one, t): t for t in to_fetch}
            with tqdm(total=len(to_fetch), unit="img") as bar:
                for fut in as_completed(futures):
                    fn, path, ok, err = fut.result()
                    if not ok:
                        failed.append((fn, err))
                    bar.update(1)

    if failed:
        print(f"  WARNING: {len(failed)} images failed to download:")
        for fn, err in failed[:10]:
            print(f"    {fn}: {err}")

    downloaded = [p for _, p in tasks if p.exists()]
    print(f"  {len(downloaded)}/{len(tasks)} images ready.")
    return downloaded


# ---------------------------------------------------------------------------
# Step 5: Upload to HF
# ---------------------------------------------------------------------------

def upload_to_hf(
    api: HfApi,
    image_paths: list,
    annotation_json: Path,
    upload_dir: Path,
) -> None:
    """
    Stage images + annotation into a local folder tree, then use
    upload_large_folder() which handles chunking, retries, and resumability
    automatically — the correct approach for 25k files.
    """
    print(f"\n[5/5] Staging {len(image_paths)} images for upload ...")
    images_dest = upload_dir / "images"
    images_dest.mkdir(parents=True, exist_ok=True)
    ann_dest = upload_dir / "annotations"
    ann_dest.mkdir(parents=True, exist_ok=True)

    # Hard-link images into the staging dir (no copy cost; same filesystem)
    for p in tqdm(image_paths, unit="img", desc="  Staging"):
        dest = images_dest / p.name
        if not dest.exists():
            try:
                os.link(p, dest)
            except OSError:
                import shutil
                shutil.copy2(p, dest)

    # Copy annotation JSON
    import shutil
    shutil.copy2(annotation_json, ann_dest / annotation_json.name)

    print(f"  Uploading via upload_large_folder to {REPO_ID} ...")
    api.upload_large_folder(
        folder_path=str(upload_dir),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print(f"\nDone. Dataset: https://huggingface.co/datasets/{REPO_ID}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    for env_file in [WORKSPACE / ".env", Path(".env")]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample COCO 25k via coco-minitrain and upload to HuggingFace"
    )
    parser.add_argument(
        "--coco_path",
        default="/tmp/coco_data",
        help="Path to COCO root (annotations/ subdir will be created/downloaded here)",
    )
    parser.add_argument(
        "--output_dir",
        default="/tmp/coco_minitrain_output",
        help="Directory for sampled annotation JSON and downloaded images",
    )
    parser.add_argument("--sample_image_count", type=int, default=25000)
    parser.add_argument(
        "--run_count",
        type=int,
        default=200000,
        help="Sampling iterations (higher = better class distribution fit, slower)",
    )
    parser.add_argument("--download_workers", type=int, default=16,
                        help="Parallel workers for image download")
    parser.add_argument("--debug", action="store_true", help="Print sampling debug info")
    parser.add_argument("--skip_clear", action="store_true",
                        help="Skip clearing the HF dataset (for resuming)")
    parser.add_argument("--skip_sampling", action="store_true",
                        help="Skip sampling if annotation JSON already exists")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip image download (images already in output_dir/images/)")
    args = parser.parse_args()

    token = load_token()
    if not token:
        sys.exit("Error: HF_TOKEN not set. Add to .env or export HF_TOKEN=<token>")

    api = HfApi(token=token)
    coco_path = Path(args.coco_path)
    output_dir = Path(args.output_dir)
    annotation_json = output_dir / "instances_train2017_minicoco.json"
    images_dir = output_dir / "images"

    if not args.skip_clear:
        clear_hf_dataset(api)

    ensure_coco_annotations(coco_path)

    if not args.skip_sampling:
        run_sampling(coco_path, annotation_json, args.sample_image_count,
                     args.run_count, args.debug)

    if not args.skip_download:
        image_paths = download_sampled_images(annotation_json, images_dir,
                                              workers=args.download_workers)
    else:
        image_paths = sorted(images_dir.glob("*.jpg"))
        print(f"\n[4/5] Skipping download. Found {len(image_paths)} images in {images_dir}")

    if not image_paths:
        sys.exit("Error: no images available. Check network access and --coco_path.")

    upload_staging_dir = output_dir / "hf_upload"
    upload_to_hf(api, image_paths, annotation_json, upload_staging_dir)


if __name__ == "__main__":
    main()
