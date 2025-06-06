# import torch, detectron2
# # !nvcc --version
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)


import cv2
import copy
import detectron2.data.detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
import random
import matplotlib.pyplot as plt


def my_dataset_function():
    ...
    # return list[dict] in the following format

# DatasetCatalog.register("roboflow_dataset_d2", my_dataset_function)


"""
Registering a dataset
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
"""

# From Roboflow dataset
# register_coco_instances(
#     "roboflow_dataset_d2", 
#     {}, 
#     "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/dataset/from_roboflow/coco.json", 
#     "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/dataset/from_roboflow/train"
# )

train_dataset_d2_name = "train_dataset_d2"
val_dataset_d2_name = "val_dataset_d2"

# Train dataset
register_coco_instances(
    train_dataset_d2_name, 
    {}, 
    "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/2025-06-04_15:46:04/train.json", 
    "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/2025-06-04_15:46:04/train"
)

# Train dataset
register_coco_instances(
    val_dataset_d2_name, 
    {}, 
    "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/2025-06-04_15:46:04/val.json", 
    "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/2025-06-04_15:46:04/val"
)

# later, to access the data:
d2_dataset_dict = DatasetCatalog.get(train_dataset_d2_name)
d2_metadata_obj = MetadataCatalog.get(train_dataset_d2_name)


print(d2_metadata_obj)
# print(d2_dataset_dict)

# class_names = d2_metadata_obj.thing_classes

# classes_to_show = [
#         'P237C_35', 'P237C_55', 'P237C_70', 'P237C_85', 'P237C_100', 
#         'P218C_70', 'P218C_80', 'P218C_90', 'P218C_100', 'P310C_35', 
#         'P310C_50', 'P310C_65', 'P298C_65', 'P298C_85', 'P301C_65', 
#         'P301C_80', 'P298C_100', 'P298C_100'
# ]

# class_ids_to_show = [class_names.index(cls) for cls in classes_to_show]



def filter_dataset(dataset_dicts, class_ids):
    filtered_dicts = []
    for data in dataset_dicts:
        new_data = copy.deepcopy(data)
        new_data['annotations'] = [ann for ann in data['annotations'] if ann['category_id'] in class_ids]
        if len(new_data['annotations']) > 0:
            filtered_dicts.append(new_data)
    return filtered_dicts


# Assuming you have loaded your dataset into 'dataset_dicts'
# And class_ids are the ids of the classes you want to visualize
filtered_dicts = filter_dataset(d2_dataset_dict, class_ids=[0, 1])


def visualize_filtered_dataset(filtered_dicts, metadata):
    for data in filtered_dicts:
        img = utils.read_image(data["file_name"], format="BGR")
        visualizer = Visualizer(img, metadata=metadata)

        vis = visualizer.draw_dataset_dict(data)
        plt.figure(figsize=(10, 8))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()


def visualise_data():
    for d in random.sample(d2_dataset_dict, 3):
        print(d["file_name"])
        img = cv2.imread(d["file_name"])
        visualiser = Visualizer(img, metadata=d2_metadata_obj, scale=0.5)
        vslr_obj = visualiser.draw_dataset_dict(d)
        plt.imshow(vslr_obj.get_image()[:, :, ::-1])
        plt.show()


visualise_data()