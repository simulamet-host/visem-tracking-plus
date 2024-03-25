import cv2
import numpy as np

def draw_obbs_from_yolo_format(image_path, label_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Read the label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(' ')
        class_id = int(parts[0])  # Assuming class_id is the first value
        # Extract and denormalize the corner points
        points = np.array([float(x) for x in parts[1:]]).reshape((4, 2))
        points[:, 0] *= width   # Scale x coordinates back to image width
        points[:, 1] *= height  # Scale y coordinates back to image height
        points = points.astype(np.int32)

        # Draw the polygon on the image
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Return the image with drawn OBBs
    return image

# Example usage:
image_path = "//home/pal.bentsen/D1/datasets2024/test-for-seg-to-aabb-label-conversion/obb-jpg-and-txt/imabes/val/52_frame_5.jpg"
obb_path = "/home/pal.bentsen/D1/datasets2024/test-for-seg-to-aabb-label-conversion/obb-jpg-and-txt/labels/val/52_frame_5.txt"
output_image_path = "/home/pal.bentsen/D1/datasets2024/test-for-seg-to-aabb-label-conversion/output_image5temp.jpg"

# Visualize and save the image with oriented bounding boxes
cv2.imwrite(output_image_path, draw_obbs_from_yolo_format(image_path, obb_path))
print(f"Image with oriented bounding boxes saved at {output_image_path}")
