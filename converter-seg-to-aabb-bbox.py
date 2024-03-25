import cv2
import numpy as np
import os
import time

def segmentation_to_bboxes(seg_path, image_shape):
    with open(seg_path, 'r') as file:
        lines = file.readlines()

    bboxes = []
    for line in lines:
        parts = line.strip().split(' ')
        points = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        abs_points = np.array([(x * image_shape[1], y * image_shape[0]) for x, y in points], dtype=np.int32)
        rect = cv2.boundingRect(abs_points.reshape(1, -1, 2))
        bboxes.append(rect)

    return bboxes

def convert_to_yolo_format(bbox, image_width, image_height):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x / image_width, center_y / image_height, w / image_width, h / image_height

def process_directories(image_dir, seg_dir, output_dir):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.txt')]
    total_files = len(seg_files)
    failed_count = 0

    print(f"{total_files} segmentation label files are going to be converted.")

    for i, seg_filename in enumerate(seg_files, start=1):
        image_path = os.path.join(image_dir, seg_filename.replace('.txt', '.jpg'))
        seg_path = os.path.join(seg_dir, seg_filename)
        
        if not os.path.exists(image_path):
            print(f"Image {seg_filename.replace('.txt', '.jpg')} not found, skipping...")
            failed_count += 1
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}, skipping...")
            failed_count += 1
            continue
        
        image_shape = image.shape
        bounding_boxes = segmentation_to_bboxes(seg_path, image_shape)
        
        yolo_label_path = os.path.join(output_dir, seg_filename)
        with open(yolo_label_path, 'w') as yolo_file:
            for bbox in bounding_boxes:
                yolo_bbox = convert_to_yolo_format(bbox, image_shape[1], image_shape[0])
                yolo_file.write(f'0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n')
        
        if i % (total_files // 20) == 0:  # Update progress every 5%
            print(f"Progress: {i/total_files*100:.1f}%")

    end_time = time.time()
    print(f"Conversion completed. {total_files - failed_count} of {total_files} files converted successfully in {end_time - start_time:.2f} seconds. {failed_count} files failed.")

# Specify the directories directly here
image_dir = "/home/pal.bentsen/D1/datasets2024/VISEM-Tracking-SILVER-GT-SEG-Splitted/images/val"
seg_dir = "/home/pal.bentsen/D1/datasets2024/VISEM-Tracking-SILVER-GT-SEG-Splitted/labels/val"
output_dir = "/home/pal.bentsen/D1/datasets2024/VISEM-Tracking-SILVER-AABB-AFTER-SAM-SEG/val"

# Call the function with your directories
process_directories(image_dir, seg_dir, output_dir)
print("Conversion completed.")