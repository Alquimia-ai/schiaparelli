from preprocessing.detectron2.config import get_cfg
from preprocessing.detectron2.engine import DefaultPredictor
from .vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer as Visualizer,
)
from .vis.extractor import DensePoseResultExtractor
from .config import add_densepose_config
import numpy as np
import torch
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"


class DensePosePredictor(object):
    def __init__(
        self,
        config_path="../checkpoints/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
        weights_path="../checkpoints/densepose/model_final_162be9.pkl",
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)

        cfg.merge_from_file(
            config_path
        )  # Use the path to the config file from densepose

        cfg.MODEL.WEIGHTS = (
            weights_path  # Use the path to the pre-trained model weights
        )
        cfg.MODEL.DEVICE = device
        print(f"[DENSEPOSE/DEVICE] {cfg.MODEL.DEVICE}")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust as needed
        self.predictor = DefaultPredictor(cfg)
        self.extractor = DensePoseResultExtractor()
        self.visualizer = Visualizer()

    def predict(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        with torch.no_grad():
            outputs = self.predictor(image)["instances"]
        outputs = self.extractor(outputs)
        return outputs

    def predict_iuv(self, image):
        outputs = self.predict(image)

        img_i = outputs[0][0].labels[None, ...]  # type: ignore
        img_uv = outputs[0][0].uv  # type: ignore
        img_uv = (img_uv - img_uv.min()) / (img_uv.max() - img_uv.min())
        img_uv *= 255
        img_iuv = torch.cat([img_i, img_uv], dim=0)
        img_iuv = img_iuv.permute(1, 2, 0)
        img_iuv = img_iuv.cpu().numpy()

        position = [int(x) for x in outputs[1][0].cpu().numpy().tolist()]  # type: ignore
        x1, y1, w, h = position
        x2 = x1 + w
        y2 = y1 + h
        image_iuv = np.zeros(image.shape, dtype=image.dtype)
        image_iuv[y1:y2, x1:x2, :] = img_iuv
        image_iuv = image_iuv[:, :, [0, 2, 1]]

        return image_iuv

    def predict_seg(self, image):
        outputs = self.predict(image)

        image_seg = np.zeros(image.shape, dtype=image.dtype)
        self.visualizer.visualize(image_seg, outputs)  # type: ignore

        return image_seg
