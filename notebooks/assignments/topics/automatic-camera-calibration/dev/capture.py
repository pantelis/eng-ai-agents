"""Capture calibration data from your own webcam.

Run it on your laptop, not in Colab: Colab cannot open a local camera without a
JavaScript workaround. Upload the resulting folder to Colab afterwards.

    python capture.py list                      # which cameras can be opened
    python capture.py board                     # write a checkerboard image to print or show
    python capture.py capture --device 0 --pattern 10 7 --square-mm 21.5   # live preview, save with keys

Keys in the preview window:

    c   save a checkerboard view (only when the board is detected)
    s   save an ordinary scene (no checkerboard)
    v   start or stop recording a video (for the person-walking part)
    q   quit

Every frame in one capture folder must come from the same camera at the same
resolution, because intrinsics are only valid for the resolution they were
estimated at. The script records the camera and resolution in session.json and
refuses to add frames from a different one.

Requirements: opencv-python and numpy.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")  # quiet the probing of missing devices

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_PATTERN = (9, 6)  # inner corners (columns, rows)


def camera_name(index: int) -> str:
    """Best-effort device name. Linux exposes it in sysfs; other systems do not."""
    sysfs = Path(f"/sys/class/video4linux/video{index}/name")
    if sysfs.exists():
        return sysfs.read_text().strip()
    return "unknown (name not available on this system)"


def backend() -> int:
    """Pick the capture backend so that camera index N means the same device everywhere.

    On Linux, V4L2 makes index N mean /dev/videoN; other backends number devices differently.
    """
    system = platform.system()
    if system == "Linux":
        return cv2.CAP_V4L2
    if system == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def candidate_indices(max_index: int) -> list[int]:
    if platform.system() == "Linux":
        return sorted(int(p.name[5:]) for p in Path("/dev").glob("video[0-9]*"))
    return list(range(max_index))


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, backend())
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. Run `python capture.py list` to see which ones open.")
    # MJPG lets most USB webcams deliver full resolution at a usable frame rate.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def read_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ok, frame = cap.read()
    if not ok:
        sys.exit("The camera opened but returned no frame. Is another application using it?")
    return frame


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of millimeters")
    return value


def cmd_list(args: argparse.Namespace) -> None:
    print("index  resolution   channels  name")
    for index in candidate_indices(args.max_index):
        cap = cv2.VideoCapture(index, backend())
        if not cap.isOpened():
            continue
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        h, w = frame.shape[:2]
        gray_like = frame.ndim == 2 or np.allclose(frame[..., 0], frame[..., 1], atol=2)
        kind = "gray (infrared?)" if gray_like else "color"
        print(f"{index:5d}  {w:4d}x{h:<5d}  {kind:16s}  {camera_name(index)}")
    print("\nUse the color camera you will calibrate. On a Mac, check it is not your iPhone (Continuity Camera).")


def cmd_board(args: argparse.Namespace) -> None:
    cols, rows = args.pattern
    square = args.square_px
    margin = square
    board = np.full(((rows + 1) * square + 2 * margin, (cols + 1) * square + 2 * margin), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y, x = margin + r * square, margin + c * square
                board[y : y + square, x : x + square] = 0
    out = Path(args.out)
    cv2.imwrite(str(out), board)
    print(f"Wrote {out}: {cols}x{rows} inner corners, {cols + 1}x{rows + 1} squares.")
    print("Print it flat (no scaling) or show it full screen on a phone or tablet.")
    print("Measure one square in millimeters and pass it to `capture` as --square-mm.")


def board_found(gray: np.ndarray, pattern: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    # The sector-based detector copes with small, dim and tilted boards far better than
    # the classic findChessboardCorners, which missed most webcam views in testing.
    return cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_EXHAUSTIVE)


def open_video_writer(stem: Path, fps: float, size: tuple[int, int]):
    """Open a video writer, trying mp4v (.mp4) and then MJPG (.avi).

    OpenCV returns an unopened writer instead of raising when a codec is missing, so check.
    """
    fps = fps if 1 <= fps <= 120 else 30.0
    for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        path = stem.with_suffix(ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
        path.unlink(missing_ok=True)
    return None, None


def next_video_stem(folder: Path, tag: str) -> Path:
    """Next unused walk_NNN stem, counting both .mp4 and .avi so a fallback recording never overwrites one."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"walk_{tag}_" if tag else "walk_"
    used = {p.stem for ext in (".mp4", ".avi") for p in folder.glob(f"{stem}*{ext}")}
    n = 0
    while f"{stem}{n:03d}" in used:
        n += 1
    return folder / f"{stem}{n:03d}"


def load_or_create_session(out: Path, info: dict) -> dict:
    path = out / "session.json"
    if path.exists():
        session = json.loads(path.read_text())
        for key in ("camera_index", "camera_name", "width", "height", "pattern_inner_corners", "square_mm"):
            if session[key] != info[key]:
                sys.exit(
                    f"{path} was captured with {key}={session[key]!r}, but this camera gives "
                    f"{info[key]!r}. Use a new --out folder for a different camera, resolution or board."
                )
        return session
    session = {**info, "created": datetime.now().isoformat(timespec="seconds"), "files": []}
    path.write_text(json.dumps(session, indent=1))
    return session


def save_session(out: Path, session: dict) -> None:
    (out / "session.json").write_text(json.dumps(session, indent=1))


def next_name(folder: Path, prefix: str, ext: str, tag: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{tag}_" if tag else f"{prefix}_"
    n = len(list(folder.glob(f"{stem}*{ext}")))
    return folder / f"{stem}{n:03d}{ext}"


def cmd_capture(args: argparse.Namespace) -> None:
    cap = open_camera(args.device, args.width, args.height)
    frame = read_frame(cap)
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    info = {
        "camera_index": args.device,
        "camera_name": camera_name(args.device),
        "width": w,
        "height": h,
        "fps": fps,
        "pattern_inner_corners": list(args.pattern),
        "square_mm": args.square_mm,
        "system": platform.platform(),
    }
    print(f"Opened camera {args.device}: {info['camera_name']}")
    print(f"Resolution {w}x{h} at {fps:.0f} fps (asked for {args.width}x{args.height})")
    if (w, h) != (args.width, args.height):
        print("The camera chose a different resolution. That is fine, but keep it the same for every capture.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = load_or_create_session(out, info)

    pattern = tuple(args.pattern)
    saved_outlines = []  # where earlier boards sat in the image, to encourage coverage
    writer = None
    video_path = None
    window = "capture (c: checkerboard, s: scene, v: video, q: quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_check = 0.0
    found, corners = False, None

    while True:
        frame = read_frame(cap)
        if writer is not None:
            writer.write(frame)

        # Detect on a half-size copy a few times per second, so the preview stays smooth.
        if time.time() - last_check > 0.3:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (w // 2, h // 2))
            found, corners = board_found(small, pattern)
            if found:
                corners = corners * 2
            last_check = time.time()

        view = frame.copy()
        for outline in saved_outlines:
            cv2.polylines(view, [outline], True, (255, 200, 0), 2)
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        n_board = sum(f["kind"] == "checkerboard" for f in session["files"])
        n_scene = sum(f["kind"] == "scene" for f in session["files"])
        status = f"board views {n_board}  scenes {n_scene}"
        if writer is not None:
            status += "  RECORDING"
        cv2.putText(view, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow(window, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            # Check the frame being saved, not the preview's last detection.
            found, corners = board_found(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pattern)
            if not found:
                print("No checkerboard detected in this frame; not saved. Hold it closer or steadier.")
                continue
            path = next_name(out / "checkerboard", "board", ".png", args.tag)
            cv2.imwrite(str(path), frame)  # the clean frame, without the overlay
            hull = cv2.convexHull(corners.astype(np.int32))
            saved_outlines.append(hull)
            session["files"].append({"kind": "checkerboard", "file": str(path.relative_to(out)), "tag": args.tag})
            save_session(out, session)
            print(f"saved {path}")
        if key == ord("s"):
            path = next_name(out / "scenes", "scene", ".png", args.tag)
            cv2.imwrite(str(path), frame)
            session["files"].append({"kind": "scene", "file": str(path.relative_to(out)), "tag": args.tag})
            save_session(out, session)
            print(f"saved {path}")
        if key == ord("v"):
            if writer is None:
                stem = next_video_stem(out / "video", args.tag)
                writer, video_path = open_video_writer(stem, fps, (w, h))
                if writer is None:
                    print("Could not open a video writer (no mp4v or MJPG codec in this OpenCV); not recording.")
                else:
                    print(f"recording {video_path}")
            else:
                writer.release()
                writer = None
                session["files"].append({"kind": "video", "file": str(video_path.relative_to(out)), "tag": args.tag})
                save_session(out, session)
                print(f"saved {video_path}")

    if writer is not None:
        writer.release()
        session["files"].append({"kind": "video", "file": str(video_path.relative_to(out)), "tag": args.tag})
        save_session(out, session)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {len(session['files'])} files listed in {out / 'session.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list the cameras that can be opened")
    p.add_argument("--max-index", type=int, default=10)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("board", help="write a checkerboard image")
    p.add_argument("--pattern", type=int, nargs=2, default=DEFAULT_PATTERN, metavar=("COLS", "ROWS"))
    p.add_argument("--square-px", type=int, default=120)
    p.add_argument("--out", default="checkerboard.png")
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("capture", help="live preview and capture")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--pattern", type=int, nargs=2, default=DEFAULT_PATTERN, metavar=("COLS", "ROWS"))
    p.add_argument("--square-mm", type=positive_float, required=True, help="measured side of one checkerboard square, in mm")
    p.add_argument("--out", default="captures")
    p.add_argument("--tag", default="", help="label added to file names, for example tilt20 or blankwall")
    p.set_defaults(func=cmd_capture)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
