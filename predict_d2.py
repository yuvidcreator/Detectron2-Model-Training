import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2
import numpy as np

# Load model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) # Replace with your config file
cfg.MODEL.WEIGHTS = "output_models/2025-06-04_16:32:44/model_final.pth" # Replace with your model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for predictions
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

# Load image
image = cv2.imread("input_images/image_2.png") # Replace with your image path

# Get predictions
outputs = predictor(image)

# Get class names from metadata
metadata = predictor.metadata
class_names = metadata.get("thing_classes", None)

print(class_names)

# Get predicted bounding boxes, classes and scores
pred_boxes = outputs["instances"].pred_boxes
pred_classes = outputs["instances"].pred_classes
scores = outputs["instances"].scores

# Filter for a specific class (e.g., class index 1)
target_class_index = 1
for i, class_index in enumerate(pred_classes):
    if class_index == target_class_index:
        bbox = pred_boxes[i].cpu().numpy()
        score = scores[i].cpu().numpy()
        class_name = class_names[class_index]
        print(f"Class: {class_name}, BBox: {bbox}, Score: {score}")

# Visualize bounding boxes (optional)
v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()