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

labelsPath = "coco.names"
# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# Get class labels
with open(labelsPath, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Read video
cap = cv2.VideoCapture("path-to-webvid10M/014701_014750/25274.mp4")

# Display image using matplotlib
def show_frame_with_matplotlib(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

results = []
frame_id = 0
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_3_sec = int(frame_rate * 3)  # Frame at the 3rd second
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Indices of frames to process (first frame, last frame, three middle frames)
frame_indices = [
    frame_3_sec,  # First frame
    total_frames - 1  # Last frame
]
# Three middle frames
interval = (total_frames - 2) // 4
frame_indices.extend([interval, interval * 2, interval * 3])

# Process frames
for frame_id in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        break

    try:
        detections, processed_frame = process_frame(frame)
    except HTTPError as e:
        if e.code == 429:
            print("HTTP 429 Error: Too Many Requests. Retrying after delay...")
            time.sleep(5)  # Wait for 5 seconds before retrying
            continue
        elif e.code == 500:
            print("HTTP 500 Error: Internal Server Error. Saving results and stopping...")
            # Save current results to JSON file
            with open("mask.json", "w") as f:
                json.dump(results, f, indent=4)
            break
        else:
            raise
    except Exception as e:
        if '429' in str(e):
            print("HTTP 429 Error: Too Many Requests. Retrying after delay...")
            time.sleep(5)  # Wait for 5 seconds before retrying
            continue
        elif '500' in str(e):
            print("HTTP 500 Error: Internal Server Error. Saving results and stopping...")
            # Save current results to JSON file
            with open("mask.json", "w") as f:
                json.dump(results, f, indent=4)
            break
        else:
            raise

    results.append({
        "frame_id": frame_id,
        "detections": detections
    })

    plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.1)

cap.release()
plt.close()

# Save results to JSON file
with open("mask.json", "w") as f:
    json.dump(results, f, indent=4)


# Setup for capturing and saving frames
cap = cv2.VideoCapture("path-to-webvid10M/014701_014750/25274.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate indices for first, last, and three middle frames
frame_indices = [0, total_frames - 1]  # first and last frames
middle_frame_step = total_frames // 4
middle_indices = [middle_frame_step, 2 * middle_frame_step, 3 * middle_frame_step]
frame_indices.extend(middle_indices)

# Process and save specified frames
for i, frame_index in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret:
        save_frame_as_image(frame, i + 1)  # Save frame as mask_{i+1}.png
    else:
        print(f"Failed to retrieve frame at index {frame_index}")

cap.release()