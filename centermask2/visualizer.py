import logging
import os
from collections import OrderedDict
import torch
import random
from detectron2.data.datasets import register_coco_instances
register_coco_instances("MCF7", {}, "/lunit/home/stevekang/decentAI/cellDetection/MCF7/LIVECell_dataset_2021/annotations/LIVECell_single_cells/mcf7/train.json","/lunit/home/stevekang/decentAI/cellDetection/MCF7/LIVECell_dataset_2021/images/livecell_train_val_images" )
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog
import detectron2.utils.comm as comm
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    # CityscapesInstanceEvaluator,
    # CityscapesSemSegEvaluator,
    # COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from centermask.evaluation import (
    COCOEvaluator,
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from centermask.config import get_cfg



cfg=get_cfg()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set the testing threshold for this model
cfg.DATASETS.TEST = ("MCF7", )
predictor = DefaultPredictor(cfg)


from detectron2.utils.visualizer import ColorMode
dataset_dicts = DatasetCatalog.get("MCF7")

for d in random.sample(dataset_dicts, 7):  
    print(d["file_name"])  
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1] 
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
              

    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("/lunit/home/stevekang/decentAI/cellDetection/centermask2/visualize/"+d["file_name"].split("/")[-1],out.get_image()[:, :, ::-1])

