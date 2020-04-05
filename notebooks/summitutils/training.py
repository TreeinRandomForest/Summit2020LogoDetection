import numpy as np
import matplotlib.pylab as plt
import detectron2
import cv2
import os
import pathlib
import json
import random
from PIL import Image, ImageDraw, ImageDraw2
import pandas as pd
import torchvision
from torchvision import transforms
import torch
import shutil
import glob

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader

def get_dicts(IMG_DIR):
    '''Returns a list of dicts - one for each image
    Each dict contains labels and bounding boxes
    
    Each folder (train, val, test) contains a data.json file
    '''

    path = os.path.join(IMG_DIR, 'data.json')
    dataset_dict = json.load(open(path))
    
    #this is hacky but replace boxmode (add enum encoder/decoder)
    for item in dataset_dict:
        for ann in item['annotations']:
            ann['bbox_mode'] = BoxMode.XYXY_ABS    

    return dataset_dict

def unregister():
    DatasetCatalog.clear()
    MetadataCatalog._NAME_TO_META = {}

def register(IMG_DIR, class_names, subfolders=['train', 'test']):
    '''Register datasets for detectron2
    '''

    for d in subfolders:
        DatasetCatalog.register(f"{IMG_DIR}_{d}", lambda d=d: get_dicts(f'{IMG_DIR}/{d}'))
        MetadataCatalog.get(f"{IMG_DIR}_{d}").set(thing_classes=class_names)

def get_metadata(dataset_name):
    metadata = MetadataCatalog.get(dataset_name)

    return metadata   

def sample_plot(dataset_dict, metadata, LOC):
    #LOC = 'logos3/train'

    d = random.sample(dataset_dict, 1)[0]
    print(d)
    img = cv2.imread(os.path.join(LOC, d["file_name"]))
    print(img)
    visualizer = Visualizer(img, metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.imshow(vis.get_image()[:, :, ::-1])
    
def prepare_for_training(N_iter,
                         output_dir,
                         train_dataset_name,
                         N_classes,
                         start_training=False,
                         gpu_avail=True,
                         model_type="COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_type))
    cfg.OUTPUT_DIR = output_dir
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_type)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.GAMMA = 0.99 #lr decay
    cfg.SOLVER.STEPS = list(range(1000, N_iter, 1000)) #(decay steps,)
    cfg.SOLVER.WARMUP_ITERS = 500 #warmup steps
    cfg.SOLVER.MAX_ITER = N_iter    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = N_classes  # 4 classes

    if not gpu_avail:
        cfg.MODEL.DEVICE = 'cpu'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    if start_training:
        trainer.train()

    return trainer, cfg    

def prepare_for_inference(cfg, test_dataset_name, threshold=0.70):
    print(f"Reading weights from output dir: {cfg.OUTPUT_DIR}")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set the testing threshold for this model
    cfg.DATASETS.TEST = (test_dataset_name, )
    predictor = DefaultPredictor(cfg)    

    return predictor

def infer_img(predictor, img_filename, metadata, tfm=None):
    #img = cv2.imread(img_filename)
    img = Image.open(img_filename)
    if tfm is not None:
        img = tfm(img)
    
    outputs = predictor(img)

    v = Visualizer(img[:,:,::-1], metadata=metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image())
    
    return outputs

def infer_img_v2(predictor, img_filename, metadata, tfm=None, reverse=False):
    img = Image.open(img_filename).convert('RGB')
    if tfm is not None:
        img = tfm(img)
    img = np.array(img)
    
    if reverse:
        outputs = predictor(img[:,:,::-1])
    else:
        outputs = predictor(img)
        
    #v = Visualizer(img[:,:,::-1], metadata=metadata, scale=0.8)
    v = Visualizer(img, metadata=metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image())
    
    return outputs
