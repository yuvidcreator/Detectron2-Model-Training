from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2
import json
import os

# Load a pre-trained model
cfg = get_cfg()

# cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "d2_pre_trained_model/model_final_f10217.pkl"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

# Load images and generate annotations
dataset_dir = "/path/to/images"
output_annotations = []

for image_name in os.listdir(dataset_dir):
    image_path = os.path.join(dataset_dir, image_name)
    img = cv2.imread(image_path)

    # Run inference
    outputs = predictor(img)

    # Process outputs
    instances = outputs["instances"]
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    masks = instances.pred_masks if instances.has("pred_masks") else None
    classes = instances.pred_classes

    # Create annotation entries
    for i in range(len(classes)):
        annotation = {
            "file_name": image_name,
            "bbox": boxes[i].tensor.cpu().numpy().tolist(),
            "category_id": int(classes[i].cpu().numpy()),
            "segmentation": masks[i].cpu().numpy().tolist() if masks is not None else None,
        }
        output_annotations.append(annotation)

# Save annotations to COCO format
with open("auto_annotations.json", "w") as f:
    json.dump(output_annotations, f)
