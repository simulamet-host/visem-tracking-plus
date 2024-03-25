#from ultralytics import YOLO
from ultralytics import RTDETR

# Load a pretrained YOLOv8n model
#model = YOLO('/home/pal.bentsen/archive/VISEM_Tracking_Train_v4/runs/detect/train14/weights/best.pt') # yolov8xcustomLR
#model = RTDETR('/home/pal.bentsen/archive/VISEM_Tracking_Train_v4/runs/detect/train23/weights/best.pt') # RTDETR X customLR
#model = RTDETR('/home/pal.bentsen/archive/VISEM_Tracking_Train_v4/runs/detect/train27/weights/best.pt') #RT-DETR after 3 rounds(gold-pseudo-gold)
#model = RTDETR('/home/pal.bentsen/D1/runs/detect/train16/weights/best.pt') #RTDETR after 4 four roounds (gold-pseudo-gold-pseudo)
#model = RTDETR('/home/pal.bentsen/archive/VISEM_Tracking_Train_v4/runs/detect/train56/weights/best.pt') #completely fresh RTDETRX on main class only
#model = RTDETR('/home/pal.bentsen/archive/VISEM_Tracking_Train_v4/runs/detect/train58/weights/best.pt') #after GPG mainclass (this should have mAP50=0.647 which is all time high)
#model = YOLO('/home/pal.bentsen/D1/runs/obb/train13/weights/best.pt') #yolov8-obb LARGE (mAP=0.857)
model = RTDETR('/home/pal.bentsen/D1/runs/detect/train38/weights/best.pt') #best silver AABB model (RTDETRX with 0.736)

# Define path to directory containing images for inference
source = '/home/pal.bentsen/D1/datasets2024-OBB-february/OBB-frames-without-val/images/train'  # Replace with your actual path

# Run inference with specific arguments for full precision and saving pseudo-labels
results = model.predict(source,
                        save_txt=True,      # Save detection to txt for training
                        conf=0.35,          # Object confidence threshold
                        iou=0.6,           # IoU threshold for NMS
                        imgsz=640,          # Image size for inference
                        #device=[7],      # Specify 'cuda' or 'cpu'
                        agnostic_nms=False, # Class-agnostic NMS
                        max_det=1000,
                        stream=True)

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
