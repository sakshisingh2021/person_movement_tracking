# Pose Estimation: Keypoint Dots on Black Background

Assignment 2: run pose estimation on a video with movement and display only the pose (COCO keypoints as white dots on a black background), not the person.

## How it works
- Model: Ultralytics YOLOv8 pose (`yolov8n-pose.pt`), which predicts the 17 COCO keypoints.
- For each frame, a black canvas is created and a white dot is drawn at every detected keypoint.
- Frames are written to an output video with OpenCV.

## Run
```
pip install -r requirements.txt
python pose_dots.py --input dance.mp4 --output pose_output.mp4
```
Optional: add `--skeleton` to draw lines between keypoints.

## Files
- `pose_dots.py`: code
- `dance.mp4`: input video
- `pose_output.mp4`: output (dots on black)
