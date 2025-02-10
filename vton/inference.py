import torch
import numpy as np
import logging
from .executionPipeline import Pipeline
from typing import Dict, Any


def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


class Inference(object):
    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.model.eval()
        self.pipe = Pipeline(model=self.model)

    def to_gpu(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        return data

    def __call__(
        self,
        data: Dict[str, Any],
        ref_acceleration: bool = False,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: int = 42,
        repaint: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        data = self.to_gpu(data)

        generator = torch.Generator(self.pipe.device).manual_seed(seed)
        logging.debug(f"[INFENRECE/ARGS] reference acceleration: {ref_acceleration}")
        logging.debug(f"[INFENRECE/ARGS] inference steps: {num_inference_steps}")
        logging.debug(f"[INFENRECE/ARGS] guidance_scale: {guidance_scale}")
        logging.debug(f"[INFENRECE/ARGS] seed: {seed}")
        logging.debug(f"[INFENRECE/ARGS] repaint: {repaint}")
        images = self.pipe(
            src_image=data["src_image"],
            ref_image=data["ref_image"],
            mask=data["mask"],
            densepose=data["densepose"],
            ref_acceleration=ref_acceleration,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            repaint=repaint,
        )[0]

        outputs = {}
        outputs["src_image"] = (data["src_image"] + 1.0) / 2.0
        outputs["ref_image"] = (data["ref_image"] + 1.0) / 2.0
        outputs["generated_image"] = images
        return outputs
