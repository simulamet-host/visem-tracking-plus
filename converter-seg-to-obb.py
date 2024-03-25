import cv2
import numpy as np
import os
import time

def segmentation_to_obb_corners(seg_path, image_shape):
    with open(seg_path, 'r') as file:
        lines = file.readlines()

    obb_corners = []
    for line in lines:
        parts = line.strip().split(' ')
        points = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        abs_points = np.array([(x * image_shape[1], y * image_shape[0]) for x, y in points], dtype=np.int32)
        rect = cv2.minAreaRect(abs_points)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        obb_corners.append(box)

    return obb_corners

def process_directories(image_dir, seg_dir, output_dir):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.txt')]
    total_files = len(seg_files)
    failed_count = 0

    print(f"{total_files} segmentation label files are scheduled for conversion.")

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
        obb_corners = segmentation_to_obb_corners(seg_path, image_shape)
        
        yolo_label_path = os.path.join(output_dir, seg_filename)
        with open(yolo_label_path, 'w') as yolo_file:
            for corners in obb_corners:
                # Convert each corner's coordinates back to relative values
                corners_rel = [(x / image_shape[1], y / image_shape[0]) for x, y in corners]
                corners_flat = [coord for pair in corners_rel for coord in pair]
                # Assuming the class_id is 0 for all, as in the original script
                yolo_file.write(f'0 {" ".join(map(str, corners_flat))}\n')
        
        if i % (total_files // 20) == 0 or i == total_files:  # Update progress every 5% or at the last file
            print(f"Progress: {i/total_files*100:.1f}%")

    end_time = time.time()
    print(f"Conversion completed. {total_files - failed_count} files converted successfully in {end_time - start_time:.2f} seconds. {failed_count} files failed out of {total_files} files.")

# Specify the directories directly here
image_dir = "/home/pal.bentsen/D1/datasets2024/VISEM-Tracking-SILVER-GT-SEG-Splitted/images/train"
seg_dir = "/home/pal.bentsen/D1/datasets2024/VISEM-Tracking-SILVER-GT-SEG-Splitted/labels/train"
output_dir = "/home/pal.bentsen/D1/datasets2024/VISEM-Tracking-Silver-OBB-AFTER-SAM-SEG-manual-conversion12feb/labels/train"

process_directories(image_dir, seg_dir, output_dir)