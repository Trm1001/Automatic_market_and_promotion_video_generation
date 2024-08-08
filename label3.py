import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from urllib.error import HTTPError

def save_frame_as_image(frame, frame_id, path_format="mask_{}.png"):
    """Save the given frame as an image file."""
    cv2.imwrite(path_format.format(frame_id), frame)

def process_frame(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detections = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detections.append({
                "label": label,
                "confidence": confidences[i],
                "box": [x, y, w, h]
            })
            color = (0, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 1, color, 2)

    return detections, frame

# Load YOLO model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open video
cap = cv2.VideoCapture("path-to-webvid10M/005301_005350/87640.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
middle_frame_index = total_frames // 2  # Calculate the middle frame

# Set video to middle frame and process it
cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
ret, frame = cap.read()
if ret:
    detections, processed_frame = process_frame(frame)
    save_frame_as_image(processed_frame, middle_frame_index)  # Save the processed middle frame
    # Display the processed frame
    plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    # Append result
    results = [{"frame_id": middle_frame_index, "detections": detections}]
    # Save detections to JSON
    with open("mask.json", "w") as f:
        json.dump(results, f, indent=4)
else:
    print("Failed to capture the middle frame.")

cap.release()
plt.close()
