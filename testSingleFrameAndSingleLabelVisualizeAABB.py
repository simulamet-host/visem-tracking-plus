import cv2
import numpy as np

def draw_bboxes_from_yolo_format(image_path, bbox_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image at path {image_path} does not exist.")
    
    height, width, _ = image.shape

    # Read the bounding boxes from the txt file
    with open(bbox_path, 'r') as file:
        bboxes = file.readlines()

    # Draw each bounding box on the image
    for bbox in bboxes:
        parts = bbox.strip().split()
        class_id, center_x, center_y, bbox_width, bbox_height = int(parts[0]), *map(float, parts[1:])
        
        # Convert from YOLO format to (x1, y1, x2, y2)
        x1 = int((center_x - bbox_width / 2) * width)
        y1 = int((center_y - bbox_height / 2) * height)
        x2 = int((center_x + bbox_width / 2) * width)
        y2 = int((center_y + bbox_height / 2) * height)

        # Draw rectangle on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put the class label if needed
        # cv2.putText(image, str(class_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image with bounding boxes
    return image

# Example usage:
# Replace 'image_path.jpg' and 'bbox_path.txt' with your file paths
image_path = "/home/pal.bentsen/D1/datasets2024/VISEM-Tracking-SILVER-AABB-AFTER-SAM-SEG/val/52_frame_0.jpg"
bbox_path = "/home/pal.bentsen/D1/datasets2024/VISEM-Tracking-SILVER-AABB-AFTER-SAM-SEG/val/52_frame_0.txt"

# Save the image with bounding boxes
output_image_path = "/home/pal.bentsen/D1/datasets2024/test-for-seg-to-aabb-label-conversion/output_image3frame0.jpg"
cv2.imwrite(output_image_path, draw_bboxes_from_yolo_format(image_path, bbox_path))
print(f"Image with bounding boxes saved at {output_image_path}")
