# YOLOv8 Real-Time Object Detection

Real-time object detection using **YOLOv8** and **OpenCV** with special handling for `person` class.

## 🚀 Features

- Real-time detection via webcam
- Custom bounding box adjustment for persons
- Console logs with class & confidence

## 🛠 Requirements

```bash
pip install opencv-python ultralytics
```

## ▶️ Usage

```bash
python object_detection.py
```

Press `q` to quit the live feed.

## 📦 Model

Uses `yolov8m.pt` (auto-downloaded by Ultralytics).

## 📌 Note

- Person box shrinks 20% in width for custom logic.
- All other classes display default YOLOv8 bounding boxes.
