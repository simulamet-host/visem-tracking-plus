from ultralytics import YOLO
model = YOLO('/home/pal.bentsen/D1/runs/obb/train16/weights/best.pt') #yolov8S-obb

# Define path to directory containing images for inference
#source = '/home/pal.bentsen/D1/frames_without_val_participants'  # Replace with your actual path
source = '/home/pal.bentsen/D1/datasets2024-OBB-february/OBB-frames-without-val/images/train'

# Run inference with specific arguments for full precision and saving pseudo-labels
results = model.predict(source,
                        save_txt=True,      # Save detection to txt for training
                        conf=0.30,          # Object confidence threshold
                        iou=0.6,           # IoU threshold for NMS
                        imgsz=640,          # Image size for inference
                        #device=[7],      # Specify 'cuda' or 'cpu'
                        agnostic_nms=False, # Class-agnostic NMS
                        max_det=1000,
                        stream=True)

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
