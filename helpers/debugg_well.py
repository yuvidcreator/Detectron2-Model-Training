import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
from detectron2 import model_zoo
# Load image
image = cv2.imread("your_image.jpg")

# Configure Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
# cfg.merge_from_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml") # replace with path to your config
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "path/to/your/model.pth" # replace with path to your model
predictor = DefaultPredictor(cfg)
outputs = predictor(image)

# Get metadata
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
class_names = metadata.thing_classes

# Filter instances
filtered_instances = []
target_classes = ["class_name_1", "class_name_2"] # replace with your classes
for i, instance in enumerate(outputs["instances"].pred_classes):
    if class_names[instance] in target_classes:
        filtered_instances.append(outputs["instances"][i])

# Visualize
v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
out = v.draw_instance_predictions(filtered_instances)
plt.imshow(out.get_image()[:, :, ::-1])
plt.show()