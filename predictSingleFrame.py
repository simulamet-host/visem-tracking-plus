from ultralytics import YOLO
#from ultralytics.utils.benchmarks import benchmark
from ultralytics import RTDETR

#model = YOLO('/Users/palbentsen/Desktop/obb_Inference/bestYolov8S-obb.pt')
#model = YOLO('/Users/palbentsen/Desktop/bestYolov8S-obb.pt') OBB
#model = RTDETR('/Users/palbentsen/Desktop/rtdetrx-Legacy-best-nonSSL.pt') #legacy RTDETR non SSL
model = RTDETR('/Users/palbentsen/Desktop/rtdetrx-cleaned-best-aabb.pt') #cleaned RTDETR 

image = "/Users/palbentsen/Desktop/master/Train/54/images/54_frame_500.jpg"
#video = "/Users/palbentsen/Desktop/archive 3/VISEM_Tracking_Train_v4/Train/60/60.mp4"
#video = "/Users/palbentsen/Desktop/Train/60/60.mp4"
#video = "/Users/palbentsen/Desktop/visem-qc/videos/923.mkv"

results = model.predict(source=image, save=True, save_frames=True, save_txt=True, device='mps', show_labels=False, show_conf=False)