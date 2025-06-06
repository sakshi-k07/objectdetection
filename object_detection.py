import cv2
from ultralytics import YOLO


def object_detection(frame, model):
    """
    Perform object detection on a video frame using YOLOv8.

    Args:
        frame: Input video frame (numpy array)


        model: Loaded YOLO model

    Returns:
        frame: Annotated frame with bounding boxes
        detections: List of detected objects with class IDs and confidence
    """
    # Perform detection
    results = model(frame, verbose=False)

    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            label = f"{class_name} {conf:.2f}"

            # Store detection info (original coordinates)
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })

            # SPECIAL HANDLING ONLY FOR PERSON DETECTIONS
            if class_name == 'person':
                # Reduce box size by 20% while keeping center position
                width_reduction = 0.20  # Reduce width by 20%
                height_reduction = 0.00  # Reduce height by 10%

                width = x2 - x1
                height = y2 - y1

                # Calculate new coordinates
                new_x1 = int(x1 + width * width_reduction / 2)
                new_x2 = int(x2 - width * width_reduction / 2)
                new_y1 = int(y1 + height * height_reduction / 2)
                new_y2 = int(y2 - height * height_reduction / 2)

                # Ensure coordinates stay within frame bounds
                new_x1 = max(0, new_x1)
                new_y1 = max(0, new_y1)
                new_x2 = min(frame.shape[1], new_x2)
                new_y2 = min(frame.shape[0], new_y2)

                # Draw adjusted bounding box for person
                cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (new_x1, new_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Original bounding box for all other classes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detections


# Example usage (unchanged)
if __name__ == "__main__":
    # Load pretrained YOLOv8 model (medium size)
    model = YOLO('yolov8m.pt')  # Automatically downloads if not present

    # Initialize video capture (0 for webcam, or file path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        processed_frame, detections = object_detection(frame, model)

        # Display results
        cv2.imshow('Object Detection', processed_frame)

        # Print detections to console (optional)
        for obj in detections:
            print(f"Detected {obj['class']} with confidence {obj['confidence']:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()