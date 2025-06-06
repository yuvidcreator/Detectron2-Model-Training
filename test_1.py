import copy
from detectron2.data.datasets import register_coco_instances
import detectron2.data.detection_utils as utils
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog

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


def main():
    dataset_name = "train_sreeni_dataset_d2"
    val_dataset_name = "val_sreeni_dataset_d2"
    
    # Training dataset
    register_coco_instances(
        dataset_name, 
        {}, 
        "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/2025-06-04_15:46:04/train.json", 
        "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/2025-06-04_15:46:04/train"
    )
    
    # Validation dataset
    register_coco_instances(
        val_dataset_name, 
        {}, 
        "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/val.json", 
        "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/val"
    )

    # Assuming you have registered your dataset with DatasetCatalog
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    # d2_dataset_dict = DatasetCatalog.get("sreeni_dataset_d2")
    # d2_metadata_obj = MetadataCatalog.get("sreeni_dataset_d2")

    # Define the IDs of the two classes you want to visualize
    class_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

    # Filter the dataset
    filtered_dicts = filter_dataset(dataset_dicts, class_ids)

    # Get the metadata
    metadata = MetadataCatalog.get(dataset_name)
    
    print(metadata)

    # Visualize the filtered dataset
    visualize_filtered_dataset(filtered_dicts, metadata)

    print(metadata)


if __name__ == "__main__":
    main()