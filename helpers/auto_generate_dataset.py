import os
import cv2
import json
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from pycocotools import mask as coco_mask
from shutil import copyfile

# Paths for the new datasets
auto_train_img_dir = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/output_auto_den_dset/20250603/train/images"
auto_train_annotation_file = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/output_auto_den_dset/20250603/train/annotations.json"
auto_val_img_dir = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/output_auto_den_dset/20250603/val/images"
auto_val_annotation_file = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/output_auto_den_dset/20250603/val/annotations.json"

# Original dataset paths
train_dataset_img_path = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/train"
val_dataset_img_path = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/val"

# List of Classes
classes = [
    "triangle",
    "P237C_35_CIR",
    "P237C_55_CIR",
    "P237C_70_CIR",
    "P237C_85_CIR",
    "P237C_100_CIR",
    "P218C_70_CIR",
    "P218C_80_CIR",
    "P218C_90_CIR",
    "P218C_100_CIR",
    "P310C_35_CIR",
    "P310C_50_CIR",
    "P310C_65_CIR",
    "P298C_65_CIR",
    "P298C_85_CIR",
    "P298C_100_CIR",
    "P301C_65_CIR",
    "P301C_80_CIR",
    "P301C_100_CIR",
    "P237C_35_SQR",
    "P237C_55_SQR",
    "P237C_70_SQR",
    "P237C_85_SQR",
    "P237C_100_SQR",
    "P218C_70_SQR",
    "P218C_80_SQR",
    "P218C_90_SQR",
    "P218C_100_SQR",
    "P310C_35_SQR",
    "P310C_50_SQR",
    "P310C_65_SQR",
    "P298C_65_SQR",
    "P298C_85_SQR",
    "P298C_100_SQR",
    "P301C_65_SQR",
    "P301C_80_SQR",
    "P301C_100_SQR",
]

# Pre-trained Model Configuration
cfg = get_cfg()

# cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.MODEL.WEIGHTS = "d2_pre_trained_model/model_final_f10217.pkl"

cfg.MODEL.DEVICE = "cpu"  # IMPORTANT: Force CPU mode

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold

predictor = DefaultPredictor(cfg)

# Ensure directories for new datasets exist
os.makedirs(auto_train_img_dir, exist_ok=True)
os.makedirs(auto_val_img_dir, exist_ok=True)

# Function to generate and save annotations
def generate_and_copy_annotations(dataset_path, output_img_dir, output_annotation_file):
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for i, name in enumerate(classes):
        coco_annotations["categories"].append({
            "id": i + 1,
            "name": name,
            "supercategory": "none"
        })

    annotation_id = 1
    for image_id, image_name in enumerate(os.listdir(dataset_path)):
        img_path = os.path.join(dataset_path, image_name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # Copy image to the new directory
        new_image_path = os.path.join(output_img_dir, image_name)
        copyfile(img_path, new_image_path)

        # Add image information
        coco_annotations["images"].append({
            "id": image_id,
            "file_name": image_name,
            "height": height,
            "width": width,
        })

        # Run inference
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes.numpy()
        pred_boxes = instances.pred_boxes.tensor.numpy()
        pred_masks = instances.pred_masks.numpy()

        # Generate annotations
        for i in range(len(pred_classes)):
            category_id = int(pred_classes[i]) + 1
            bbox = pred_boxes[i].tolist()
            mask = pred_masks[i]
            rle = coco_mask.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            area = int(coco_mask.area(rle))

            # Add annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # Convert to COCO format
                "segmentation": rle,
                "area": area,
                "iscrowd": 0
            }
            coco_annotations["annotations"].append(annotation)
            annotation_id += 1

    # Save annotations to a JSON file
    with open(output_annotation_file, "w") as f:
        json.dump(coco_annotations, f)

# Generate annotations for train and val datasets
generate_and_copy_annotations(train_dataset_img_path, auto_train_img_dir, auto_train_annotation_file)
generate_and_copy_annotations(val_dataset_img_path, auto_val_img_dir, auto_val_annotation_file)

print("Automatic annotation generation complete.")
