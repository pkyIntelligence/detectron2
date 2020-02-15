import torch
import os
import cv2

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger


setup_logger(output='logs')

WINDOW_NAME = "Visual Genome detections"

from demo.predictor import VisualizationDemo

with open('data/objects_vocab.txt', 'r') as  f:
    obj_vocab = f.read().splitlines()

# Registering metadata for Visual Genome Object Classes
# MetadataCatalog.get("VisualGenomeObjects").thing_classes = obj_vocab


cfg = get_cfg()
cfg.merge_from_file('configs/COCO-Detection/faster_rcnn_X_101_64x4d_FPN_2x_vlp.yaml')
cfg.freeze()

predictor = DefaultPredictor(cfg)
demo = VisualizationDemo(cfg)

#model = build_model(cfg)
#model.eval()

#checkpointer = DetectionCheckpointer(model)
#checkpointer.load(os.path.join('model_weights', 'e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl'))


img1 = read_image(os.path.join('test_images', '12283150_12d37e6389_z.jpg'), format="BGR")
img2 = read_image(os.path.join('test_images', '25691390_f9944f61b5_z.jpg'), format="BGR")
img3 = read_image(os.path.join('test_images', '9247489789_132c0d534a_z.jpg'), format="BGR")

preds, vis_out = demo.run_on_image(img3, obj_vocab)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.imshow(WINDOW_NAME, vis_out.get_image()[:, :, ::-1])

#preds['instances'].get_fields()['box_features'].shape
#preds['instances'].get_fields()['probs'].shape
