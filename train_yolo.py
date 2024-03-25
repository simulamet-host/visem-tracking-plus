from ultralytics import RTDETR
from ultralytics import YOLO

def main():
    # Load a model (you can load a pre-trained one for transfer learning)
    #model = RTDETR('rtdetr-x.pt')
    #model = RTDETR('/home/pal.bentsen/archive/VISEM_Tracking_Train_v4/runs/detect/train58/weights/best.pt') #GPG model which got map50=0.647
    #model = RTDETR('rtdetr-l.pt')
    #model = RTDETR('/home/pal.bentsen/archive/VISEM_Tracking_Train_v4/runs/detect/train58/weights/best.pt')
    #model = YOLO('yolov9e.pt')
    model = YOLO('yolov8n.pt')
    #model = RTDETR('/home/pal.bentsen/archive/VISEM_Tracking_Train_v4/runs/detect/train58/weights/best.onnx')
    #model = RTDETR('/home/pal.bentsen/D1/runs/detect/train38/weights/best.pt') #Silver G model which got mAP=0.736
    #display model information (optional)
    #model = RTDETR('/home/pal.bentsen/D1/runs/detect/train38/weights/best.pt')
    model.info()
    #results = model.train(data='yolov8_dataset.yaml', patience=7, epochs=100, imgsz=640)
    # Train the model on your dataset
    #model.train(data="yolov8_dataset.yaml", epochs=300, patience=50, imgsz=640)
    model.train(data="yolov8_dataset_legacy.yaml", epochs=300, patience=50, imgsz=640)
    #model.val(data="yolov8_dataset_legacy.yaml", imgsz=640)

    # Optionally, evaluate the model
    metrics = model.val()

if __name__ == '__main__':
    main()