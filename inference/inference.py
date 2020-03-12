import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
import os
import pathlib
import json
import random
from PIL import Image, ImageDraw2
import torchvision
from torchvision import transforms
import torch
import shutil
import glob

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader

plt.ion()
setup_logger()

#should move into a config file
MODEL_CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml" 
class_names = ['sas', 'rh', 'anaconda', 'cloudera']
N_classes = len(class_names)

#Suspicion:
#predictor.cfg.MODEL.DEVICE - for input tensors
#predictor.model - send to device

def load_model(model_location, threshold, N_classes):
	'''
	Read persisted weights and construct model
	'''

	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG_FILE))
	
	cfg.OUTPUT_DIR = model_location
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = N_classes
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

	predictor = DefaultPredictor(cfg)

	if torch.cuda.is_available():
		predictor.model = predictor.model.to('cuda')
		predictor.cfg.MODEL.DEVICE = 'cuda'

	return predictor

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

def infer_img(predictor, img_filename, class_names):
    '''
    Use the predictor returned by load_model to make
    predictions on img_filename
    '''

    img = cv2.imread(img_filename)
    outputs = predictor(img)

    metadata = MetadataCatalog.get('inference').set(thing_classes=class_names)

    v = Visualizer(img[:,:,::-1], metadata=metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image())
    
    return outputs

def register(IMG_DIR, class_names, subfolders=['train', 'test']):
    '''
    Register train, val, test sets for computing metrics (mAP)
    '''

    for d in subfolders:
        DatasetCatalog.register(f"{IMG_DIR}_{d}", lambda d=d: get_dicts(f'{IMG_DIR}/{d}'))
        MetadataCatalog.get(f"{IMG_DIR}_{d}").set(thing_classes=class_names)

def unregister():
	DatasetCatalog.clear()
	MetadataCatalog._NAME_TO_META = {}

def compute_metrics(dataset_name, output_loc, predictor):
	'''Needs gpu for reasonable performance
	'''
	cfg = predictor.cfg

	evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=output_loc)
	loader = build_detection_test_loader(cfg, dataset_name)
	metrics = inference_on_dataset(predictor.model, loader, evaluator)

	return metrics

def inference_example(img_filename, threshold=0.70):
	N_classes = len(class_names)

	predictor = load_model('logo_detector_output/', threshold, N_classes)

	outputs = infer_img(predictor, img_filename, class_names)

	return outputs

def compute_metrics_example():
	'''Still working on this
	'''
	threshold = 0.70
	N_classes = len(class_names)

	predictor = load_model('logo_detector_output/', threshold, N_classes)	

	#register train and test sets
	unregister()
	register('logo_detector/combined', class_names, subfolders=['train', 'test'])

	train_metrics = compute_metrics('logo_detector/combined_train',
									'test_output',
									predictor)
		
	return train_metrics