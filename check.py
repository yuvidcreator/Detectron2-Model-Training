import copy
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
import detectron2.data.detection_utils as utils
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode



cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 37   # define our custom annotated total number of classes

cfg.MODEL.WEIGHTS = "output_models/2025-06-05_18:15:21/model_final.pth"  # path to the model we just trained
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "output_models/model_final.pth")  # path to the model we just trained

cfg.MODEL.DEVICE = "cpu"

dataset_name = "welleys_d2_dataset"

cir_class = [
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
]
sqr_classes = [
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

image_path = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/input_images/image_2.png"

# MetadataCatalog.get(dataset_name).set(thing_classes = classes, thing_colors = [(112, 36, 246), (81,155,81), (147, 69, 147)])
# metadata_obj = MetadataCatalog.get(dataset_name).set(thing_classes = classes)
metadata_obj = MetadataCatalog.get(dataset_name)


predictor = DefaultPredictor(cfg)

img_cv2_obj = cv2.imread(image_path)
# plt.figure(figsize=[15,7,5])
# plt.imshow(img_cv2_obj[:,:,::-1])

pred_output_obj = predictor(img_cv2_obj)

# Get class names from metadata
# metadata = metadata_obj.metadata
# class_names = metadata.get("thing_classes", None)
# print(class_names)

print(pred_output_obj["instances"].pred_classes)

vslr_obj = Visualizer(img_cv2_obj[:,:,::-1], metadata=metadata_obj, scale=1, instance_mode=ColorMode.SEGMENTATION)
img_output_pred = vslr_obj.draw_instance_predictions(pred_output_obj["instances"].to("cpu"))
plt.figure(figsize=[20,10])
plt.imshow(img_output_pred.get_image()[:, :, ::-1])
plt.show()
# cv2.imshow("Output", img_output_pred.get_image()[:, :, ::-1])
# cv2.waitKey(0)
plt.waitforbuttonpress()
plt.close()


def filter_dataset(dataset_dicts, class_ids):
    filtered_dicts = []
    for data in dataset_dicts:
        new_data = copy.deepcopy(data)
        new_data['annotations'] = [ann for ann in data['annotations'] if ann['category_id'] in class_ids]
        if len(new_data['annotations']) > 0:
            filtered_dicts.append(new_data)
    return filtered_dicts


def visualize_filtered_dataset(filtered_dicts, metadata):
    for data in filtered_dicts:
        img = utils.read_image(data["file_name"], format="BGR")
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)

        vis = visualizer.draw_dataset_dict(data)
        plt.figure(figsize=(10, 8))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()
        plt.waitforbuttonpress()
        plt.close()














# # Define the IDs of the two classes you want to visualize
# class_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,28]

# # Filter the dataset
# filtered_dicts = filter_dataset(pred_output_obj, class_ids)

# # Visualize the filtered dataset
# visualize_filtered_dataset(filtered_dicts, metadata_obj)

# print(filtered_dicts)

# utils.annotations_to_instances()

# # https://detectron2.readthedocs.io/en/v0.5/modules/data_transforms.html
# from detectron2.data.transforms import CropTransform
# from detectron2.structures import BitMasks

# BitMasks.crop_and_resize()