import os
HOME = os.getcwd()
print(HOME)

import yolox
print("yolox.__version__:", yolox.__version__)

#from IPython import display
#display.clear_output()

import ultralytics
ultralytics.checks()

import sys
#sys.path.append("C:\Users\MSII\Desktop\AI Detecting Counting Image\backend\ByteTrack") 

#from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


#from IPython import display
#display.clear_output()


import supervision
print("supervision.__version__:", supervision.__version__)

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from typing import List

import numpy as np


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids



# settings
MODEL = "yolov8x.pt"

from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names

print(CLASS_NAMES_DICT)
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7, 14, 15, 16, 39]


SOURCE_IMAGE_PATH = "2.jpg"
import cv2
from IPython.display import FileLink

# create frame generator
generator = get_video_frames_generator(SOURCE_IMAGE_PATH)
# create instance of BoxAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
# acquire first video frame
iterator = iter(generator)
frame = next(iterator)
# model prediction on single frame and conversion to supervision Detections
results = model(frame)
detections = Detections(
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
)
# format custom labels
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections
]

import matplotlib.pyplot as plt

# Create a larger figure size to make the image larger
plt.figure(figsize=(6, 6))

# annotate and display frame
frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

#creating an empty list to store the predicted stuffs
predicted_list = []
for i, (x1, y1, x2, y2, confidence, class_id) in enumerate(zip(
    detections.xyxy[:, 0],
    detections.xyxy[:, 1],
    detections.xyxy[:, 2],
    detections.xyxy[:, 3],
    detections.confidence,
    detections.class_id
)):
    print(f"Object {i+1}: Class ID {CLASS_NAMES_DICT[class_id]}, Confidence {confidence:.2f}")
    predicted_class = CLASS_NAMES_DICT[class_id]
    predicted_list.append(predicted_class)

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw a smaller bounding box
    #cv2.putText(frame, f"Object {i+1}: {CLASS_NAMES_DICT[class_id]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

output_image_filename = "otha dei.jpg"
print(f"Osaka Osaka {predicted_list}")
print(f"Number of objects: {len(predicted_list)}")

# Save the annotated image to a local directory
cv2.imwrite(output_image_filename, frame)


#%matplotlib inline
show_frame_in_notebook(frame, (8, 8))
plt.imshow(frame)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()