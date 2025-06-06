import os
import gc
from datetime import datetime
import multiprocessing as mp
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import DatasetEvaluator




class TrainerWithEvaluator(DefaultTrainer):
    @staticmethod
    def build_evaluator(cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("./output_eval", exist_ok=True)
            output_folder = "./output_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)


def train_my_model():
    now = datetime.now()
    # Format it as a string: YYYY-MM-DD HH:MM:SS
    timestamp_str = now.strftime("%Y-%m-%d_%H:%M:%S")

    output_model_dir_path = "output_models"
    output_eval_dir_path = "output_eval"

    os.makedirs(output_model_dir_path, exist_ok=True)
    os.makedirs(f"{output_model_dir_path}/{timestamp_str}", exist_ok=True)
    
    os.makedirs(output_eval_dir_path, exist_ok=True)
    os.makedirs(f"{output_eval_dir_path}/{timestamp_str}", exist_ok=True)
    
    final_model_dir = f"{output_model_dir_path}/{timestamp_str}"
    final_model_eval_dir = f"{output_eval_dir_path}/{timestamp_str}"

    train_dataset_name = "welleys_d2_train_dataset"
    val_dataset_name = "welleys_d2_val_dataset"

    annotation_file_for_training = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/2025-06-06_13:53:06/train.json"
    img_dataset_path_for_training = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/2025-06-06_13:53:06/train"

    annotation_file_for_eval = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/2025-06-06_13:53:06/val.json"
    img_dataset_file_for_eval = "/Users/MGBiMACmini_0002/Desktop/Yuvraaj/POCs/ML_Projects/D2_model_training/splited_dataset/2025-06-06_13:53:06/val"

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

    register_coco_instances(
        train_dataset_name, 
        {}, 
        annotation_file_for_training, 
        img_dataset_path_for_training
    )

    register_coco_instances(
        val_dataset_name, 
        {}, 
        annotation_file_for_eval, 
        img_dataset_file_for_eval
    )

    # MetadataCatalog.get(train_dataset_name).set(thing_classes = classes, thing_colors = [(112, 36, 246), (81,155,81), (147, 69, 147)])
    MetadataCatalog.get(train_dataset_name).set(thing_classes = classes)

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))   # Evaluate on `custom_val` every 100 iterations

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)

    cfg.TEST.EVAL_PERIOD = 100

    """
    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    """
    cfg.SOLVER.MAX_ITER = 5000

    """
    Use a learning rate warm-up to avoid abrupt initial updates:
    """
    cfg.SOLVER.WARMUP_ITERS = 500
    
    # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_FACTOR = 0.001

    cfg.SOLVER.WEIGHT_DECAY = 0.0001     # To prevent overfitting due to the small dataset size:

    cfg.SOLVER.BASE_LR = 0.0001 # pick a good LR

    cfg.SOLVER.WEIGHT_DECAY = 0.0001

    cfg.SOLVER.STEPS = []        # do not decay learning rate

    # cfg.SOLVER.WARMUP_METHOD = "linear"

    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people

    """
    # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    """
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    
    """
    # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    """
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 37 

    cfg.DATALOADER.AUGMENTATIONS = [
        T.RandomFlip(prob=0.5),
        T.RandomRotation([0, 90, 180, 270]),
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2)
    ]

    """
    If certain classes are underrepresented, consider adjusting the sampling strategy or loss weighting:
    Enable focal loss for better handling of class imbalance:
    """
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Increase for better sampling
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5     # Confidence threshold

    # cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.NUM_WORKERS = mp.cpu_count()
    # cfg.SOLVER.REFERENCE_WORLD_SIZE = 0

    cfg.OUTPUT_DIR = final_model_dir
    cfg.SEED = -1
    cfg.MODEL.DEVICE = "cpu"

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # trainer = DefaultTrainer(cfg) 
    trainer = TrainerWithEvaluator(cfg) 

    # trainer.resume_or_load(resume=False)
    trainer.resume_or_load(True)
    trainer.train()

    # Evaluation after training
    evaluator = COCOEvaluator(val_dataset_name, cfg, False, output_dir=final_model_eval_dir)
    val_loader = build_detection_test_loader(cfg, val_dataset_name)
    inference_on_dataset(trainer.model, val_loader, evaluator)
    
    print("\nEvaluation Completed Successfully.!!\n")
    
    gc.collect()



if __name__ == "__main__":
    train_my_model()