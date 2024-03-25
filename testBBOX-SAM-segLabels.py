# we are going to use ultralytics/data/converter.py
# to use this function yolo_bbox2obb
#from ultralytics.data.converter import yolo_bbox2obb
from ultralytics.data.converter import yolo_bbox2segment
#from ultralytics import YOLO
#from ultralytics import SAM
import numpy

dataPath = "/home/pal.bentsen/D1/datasetsPotentialTestForSAM/input"
#obb_savePath = "/home/pal.bentsen/D1/datasets2024/26jan-NewGT-seg
#seg_savePath = "/home/pal.bentsen/D1/datasets2024/BBOX-2-OBB-v1-stockModels/segOutput"
#det_modelPath = "/home/pal.bentsen/D1/runs/detect/train11/weights/best.pt" #best yolov8x model?
save_dir = "/home/pal.bentsen/D1/datasetsPotentialTestForSAM/output"
""" 

yolo_bbox2obb(data=dataPath, obb_save=obb_savePath, seg_save=seg_savePath, segment_data=False, det_model=det_modelPath, sam_model="sam_l.pt") 

"""
#Previously used a self-written function
#Now we can just use ultralytics bbox -> SAM -> Seg mask
yolo_bbox2segment(im_dir=dataPath, save_dir=save_dir, sam_model="sam_h.pt")
#yolo_bbox2segment(im_dir=dataPath, save_dir=save_dir, sam_model="mobile_sam.pt")
#def yolo_bbox2segment(im_dir, save_dir=None, sam_model="sam_b.pt"):