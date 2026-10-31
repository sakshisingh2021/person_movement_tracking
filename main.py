"""
Person Movement Tracking & Trajectory Visualization
Assignment implementation:
- Detect people in a video using YOLO
- Track each person with a persistent ID
- Represent each person as a dot on a black canvas
- Draw their movement trajectory
- Keep the last known dot visible during short detection gaps
- No pose estimation or speed estimation
"""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Track people and visualize movement as dots.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="output_tracking.mp4", help="Path to output video")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--trail", type=int, default=80, help="Maximum trail points per tracked person")
    parser.add_argument("--max-missed", type=int, default=45,
                        help="Frames to retain last known dot when temporarily not detected")
    parser.add_argument("--canvas-width", type=int, default=900, help="Output canvas width")
    parser.add_argument("--canvas-height", type=int, default=600, help="Output canvas height")
    return parser.parse_args()


def fit_point(x, y, frame_w, frame_h, canvas_w, canvas_h):
    """Map frame coordinates onto the black visualization canvas."""
    px = int(x * canvas_w / max(frame_w, 1))
    py = int(y * canvas_h / max(frame_h, 1))
    return max(0, min(canvas_w - 1, px)), max(0, min(canvas_h - 1, py))


def color_for_id(track_id):
    """Generate a stable, visible color for each tracker ID."""
    palette = [
        (0, 255, 255), (255, 120, 0), (0, 220, 100), (255, 0, 200),
        (0, 140, 255), (220, 220, 0), (180, 80, 255), (255, 255, 255)
    ]
    return palette[int(track_id) % len(palette)]


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open input video. Check the file format/path.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (args.canvas_width, args.canvas_height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video. Try an .mp4 output path.")

    model = YOLO(args.model)
    trails = defaultdict(lambda: deque(maxlen=args.trail))
    last_seen = {}
    last_position = {}
    frame_index = 0

    print("Processing video... Press Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # COCO class 0 = person. Tracker IDs persist across frames.
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=args.conf,
                tracker="bytetrack.yaml",
                verbose=False
            )

            current_ids = set()
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    ids = boxes.id.cpu().numpy().astype(int)

                    for box, track_id in zip(xyxy, ids):
                        x1, y1, x2, y2 = box
                        # Use bottom-center to approximate the person's ground position.
                        center_x = (x1 + x2) / 2.0
                        bottom_y = y2
                        point = fit_point(
                            center_x, bottom_y, frame_w, frame_h,
                            args.canvas_width, args.canvas_height
                        )
                        trails[track_id].append(point)
                        last_seen[track_id] = frame_index
                        last_position[track_id] = point
                        current_ids.add(track_id)

            # Black canvas: only dots and trails are shown, not the original video.
            canvas = np.zeros(
                (args.canvas_height, args.canvas_width, 3), dtype=np.uint8
            )

            for track_id, trail in list(trails.items()):
                missed = frame_index - last_seen.get(track_id, frame_index)
                if missed > args.max_missed:
                    continue

                color = color_for_id(track_id)
                points = list(trail)

                # Draw trajectory segments.
                for i in range(1, len(points)):
                    # Older trail segments are dimmer.
                    intensity = int(70 + 185 * i / max(len(points) - 1, 1))
                    segment_color = tuple(int(c * intensity / 255) for c in color)
                    cv2.line(canvas, points[i - 1], points[i], segment_color, 2)

                # Keep the last known dot visible during short detection gaps.
                if track_id in last_position:
                    dot = last_position[track_id]
                    radius = 7 if missed == 0 else 5
                    cv2.circle(canvas, dot, radius, color, -1)
                    cv2.putText(
                        canvas, f"ID {track_id}",
                        (dot[0] + 10, max(18, dot[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                    )

            cv2.putText(
                canvas, f"Frame: {frame_index}",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (200, 200, 200), 1, cv2.LINE_AA
            )
            writer.write(canvas)
            frame_index += 1

            if frame_index % 100 == 0:
                print(f"Processed {frame_index} frames...")

    finally:
        cap.release()
        writer.release()

    print(f"Done. Output saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
