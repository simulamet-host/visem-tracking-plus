from ultralytics import YOLO
from ultralytics import RTDETR

# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
#model = YOLO('/home/pal.bentsen/D1/runs/obb/train13/weights/best.pt')  # load a custom trained model
#model = RTDETR('/home/pal.bentsen/archive/VISEM_Tracking_Train_v4/runs/detect/train58/weights/best.pt')
model = RTDETR('/home/pal.bentsen/archive/VISEM_Tracking_Train_v4/runs/detect/train58/weights/best.onnx')
#model = RTDETR('rtdetr-x.pt')

# Export the model
model.export(format='engine', imgsz=640)
#model.export(format='coreml', imgsz=640, nms=True)