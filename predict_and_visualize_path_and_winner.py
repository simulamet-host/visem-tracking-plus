# OBB
import pandas as pd  # Import pandas for easy data handling if you choose to use DataFrame
from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/Users/palbentsen/Desktop/master/obb_Inference/bestYolov8S-obb.mlpackage")
# model = YOLO('/Users/palbentsen/Desktop/master/obb_Inference/bestYolov8S-obb.pt')

# Open the video file
video_path = "/Users/palbentsen/Desktop/master/Train/82/82.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history

track_history = defaultdict(list)
cache = {"frame_count": 0, "cache_period": 5, "last_results": None}

# Before the loop, initialize a dictionary to store lengths
closing_line_lengths = defaultdict(float)

# Before the loop, initialize variables for the overall longest line
overall_longest_length = 0
overall_longest_track_id = None

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        current_frame_longest_length = 0
        current_frame_longest_track_id = None

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, show_conf=False, show_labels=False)

        # Get the boxes and track IDs
        boxes = results[0].obb.xywhr.cpu()
        track_ids = results[0].obb.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot(labels=False)

        # Ensure this part correctly adds points to the track history
        for box, track_id in zip(boxes, track_ids):
            # Example point addition (ensure this matches your actual logic)
            x, y, w, h, r = box
            # Assuming x, y represent the center of the box
            track_history[track_id].append(
                (x, y)
            )  # Update this logic as per your actual data structure

        # Check each track's closing line length
        # we only want to do this every 10 frames
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 10 == 0:
            for track_id in track_ids:
                if track_history[track_id]:  # Check if the list is not empty
                    points = (
                        np.hstack(track_history[track_id])
                        .astype(np.int32)
                        .reshape((-1, 1, 2))
                    )
                    if len(points) > 1:
                        first_point = points[0][0]
                        last_point = points[-1][0]
                        length = np.linalg.norm(
                            np.array(last_point) - np.array(first_point)
                        )
                        closing_line_lengths[track_id] = length

                        # Update the longest line in the current frame
                        if length > current_frame_longest_length:
                            current_frame_longest_length = length
                            current_frame_longest_track_id = track_id

                        # Update the overall longest line seen so far
                        if length > overall_longest_length:
                            overall_longest_length = length
                            overall_longest_track_id = track_id

        # Before displaying the annotated frame
        for track_id in track_ids:
            # Check if the list is not empty and contains items that can be stacked
            if track_history[track_id] and all(
                isinstance(point, (np.ndarray, list, tuple))
                for point in track_history[track_id]
            ):
                points = (
                    np.hstack(track_history[track_id])
                    .astype(np.int32)
                    .reshape((-1, 1, 2))
                )
                if len(points) > 1:
                    color = (0, 255, 255)  # Default color for non-longest tracks
                    thickness = 1  # Default thickness for non-longest tracks
                    if track_id == overall_longest_track_id:
                        # Use a distinct color and thickness for the longest track
                        color = (0, 255, 255)  # Highlight color for the longest track
                        thickness = 1  # Increased thickness for the longest track

                        # Draw the polyline for the track
                        cv2.polylines(
                            annotated_frame,
                            [points],
                            isClosed=False,
                            color=color,
                            thickness=thickness,
                        )

                        # Add text annotation
                        text_position = points[-1][
                            0
                        ]  # Position the text at the end of the track
                        cv2.putText(
                            annotated_frame,
                            "Best swimmer",
                            (text_position[0] + 10, text_position[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (120, 120, 255),
                            2,
                        )
                    else:
                        # Draw the polyline for non-winning tracks
                        cv2.polylines(
                            annotated_frame,
                            [points],
                            isClosed=False,
                            color=color,
                            thickness=thickness,
                        )

                    first_point, last_point = points[0][0], points[-1][0]
                    if track_id == overall_longest_track_id:
                        # Highlight the closing line of the longest track differently
                        cv2.line(
                            annotated_frame,
                            tuple(first_point),
                            tuple(last_point),
                            color=(255, 0, 255),
                            thickness=2,
                        )
            else:
                # Skip this track_id if the history is empty or not properly formatted
                continue

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking with highlighted winner: Video 82", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Convert the lengths to a DataFrame for easy handling
lengths_df = pd.DataFrame(
    list(closing_line_lengths.items()), columns=["Track_ID", "Closing_Line_Length"]
)
print(lengths_df)

# Or simply print the dictionary if you prefer
print(closing_line_lengths)

# Processing after loop completion
cap.release()
cv2.destroyAllWindows()
