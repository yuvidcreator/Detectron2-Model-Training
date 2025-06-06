from datetime import datetime
import json
import os
import random
import shutil


now = datetime.now()
# Format it as a string: YYYY-MM-DD HH:MM:SS
timestamp_str = now.strftime("%Y-%m-%d_%H:%M:%S")


output_dir = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset"


os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/{timestamp_str}", exist_ok=True)
splited_data_dir = f"{output_dir}/{timestamp_str}"


# Paths
DATASET_DIR = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/dataset/from_sreeni/20250606/images"  # Directory containing your images
ANNOTATION_FILE = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/dataset/from_sreeni/20250606/welleys_d2dataset_v4.json"  # Original COCO-style JSON file


# Parameters
SPLIT_RATIO = 0.9  # 90% training, 10% validation
random.seed(42)  # For reproducibility


# Load the COCO JSON annotations
with open(ANNOTATION_FILE, 'r') as f:
    coco_data = json.load(f)


# Get all image IDs
image_ids = [img['id'] for img in coco_data['images']]
random.shuffle(image_ids)


# Split into train and val
split_index = int(len(image_ids) * SPLIT_RATIO)
train_ids = set(image_ids[:split_index])
val_ids = set(image_ids[split_index:])


# Create new JSON files for train and val
train_data = {k: [] for k in coco_data.keys()}
val_data = {k: [] for k in coco_data.keys()}


# Split images and annotations
for img in coco_data['images']:
    if img['id'] in train_ids:
        train_data['images'].append(img)
    else:
        val_data['images'].append(img)


for ann in coco_data['annotations']:
    if ann['image_id'] in train_ids:
        train_data['annotations'].append(ann)
    else:
        val_data['annotations'].append(ann)


train_data['categories'] = coco_data['categories']
val_data['categories'] = coco_data['categories']


# Save the new JSON files
os.makedirs(splited_data_dir, exist_ok=True)
with open(os.path.join(splited_data_dir, 'train.json'), 'w') as f:
    json.dump(train_data, f)
with open(os.path.join(splited_data_dir, 'val.json'), 'w') as f:
    json.dump(val_data, f)


# (Optional) Copy images into separate directories
for subset, ids in [('train', train_ids), ('val', val_ids)]:
    subset_dir = os.path.join(splited_data_dir, subset)
    os.makedirs(subset_dir, exist_ok=True)
    for img in coco_data['images']:
        if img['id'] in ids:
            src_path = os.path.join(DATASET_DIR, img['file_name'])
            dst_path = os.path.join(subset_dir, img['file_name'])
            shutil.copy(src_path, dst_path)


