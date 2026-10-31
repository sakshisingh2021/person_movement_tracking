"""
Pose estimation -> white COCO keypoint dots on a black background.

Usage:
    pip install ultralytics opencv-python
    python pose_dots.py --input dance.mp4 --output pose_output.mp4
    python pose_dots.py --input dance.mp4 --output pose_output.mp4 --skeleton
"""
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

# COCO 17-keypoint skeleton (pairs of keypoint indices)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms + shoulders
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to input video")
    parser.add_argument("--output", default="pose_output.mp4")
    parser.add_argument("--skeleton", action="store_true",
                        help="also draw lines between keypoints")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="min keypoint confidence")
    args = parser.parse_args()

    model = YOLO("yolov8n-pose.pt")  # downloads automatically on first run

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, verbose=False)[0]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)  # black background

        if result.keypoints is not None:
            all_xy = result.keypoints.xy.cpu().numpy()      # (people, 17, 2)
            all_conf = result.keypoints.conf
            all_conf = (all_conf.cpu().numpy() if all_conf is not None
                        else np.ones(all_xy.shape[:2]))

            for xy, conf in zip(all_xy, all_conf):
                if args.skeleton:
                    for a, b in SKELETON:
                        if conf[a] > args.conf and conf[b] > args.conf:
                            cv2.line(canvas,
                                     tuple(int(v) for v in xy[a]),
                                     tuple(int(v) for v in xy[b]),
                                     (255, 255, 255), 2)
                for (x, y), c in zip(xy, conf):
                    if c > args.conf:
                        cv2.circle(canvas, (int(x), int(y)), 6,
                                   (255, 255, 255), -1)  # white dot

        out.write(canvas)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done. Processed {frame_idx} frames -> {args.output}")


if __name__ == "__main__":
    main()
