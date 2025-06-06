import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt


cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/MaskRCNN_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

img_path = "parrots.jpg"

img = cv2.imread(img_path)

print(img.shape)

outputs = predictor(img)

# plt.figure(figsize=[15,7,5])
# plt.imshow(img[:,:,::-1])

v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite("output.jpg", out.get_image()[:, :, ::-1])
print("Output saved to output.jpg")
print(outputs)