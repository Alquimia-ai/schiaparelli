# Prediction interface for Cog ⚙️
# https://cog.run/python
from cog import BasePredictor, Input, Path, BaseModel
import json
from preprocessing.densepose.predictor import DensePosePredictor
from preprocessing.humanparsing.run_parsing import HumanParsing
from preprocessing.openpose.run_openpose import OpenPose
from PIL import Image
import tempfile
from typing import Iterator, Optional
from helpers import resize_and_center, agnostic_mask
import numpy as np


class Schiaparelli(BaseModel):
    human_parsing: Optional[Path] = None
    open_pose: Optional[Path] = None
    dense_pose: Optional[Path] = None
    mask: Optional[Path] = None


class Predictor(BasePredictor):
    def setup(self) -> None:  # type: ignore
        """Load the model into memory to make running multiple predictions efficient"""
        with open("config.json", "r") as file:
            self.config = json.load(file)
            self.parsing_res = tuple(self.config["parsing_res"])
            self.resize_res = tuple(self.config["resize_res"])

        ## Preprocessing Part (HumanParsing, OpenPose & DensePose)
        self.human_parsing = HumanParsing(
            atr_path=self.config["human_parsing"]["atr_path"],
            lip_path=self.config["human_parsing"]["lip_path"],
        )
        self.open_pose = OpenPose(
            body_model_path=self.config["openpose"]["body_model_path"]
        )
        self.dense_pose = DensePosePredictor(
            config_path=self.config["densepose"]["config_path"],
            weights_path=self.config["densepose"]["weights_path"],
        )

    def predict(  # type: ignore
        self,
        person: Path = Input(
            description="The person garment that is going to be replaced"
        ),
        garment: Path = Input(
            description="The garment that is going to be placed on the person"
        ),
        repaint: bool = Input(
            description="With this option you are telling the model to make a smooth transition between the generated garment and the person body",
            default=False,
        ),
        seed: int = Input(
            description="Seed for the diffusion model. Useful to reproduce the same results",
            default=42,
        ),
        guidance_scale: float = Input(
            description="Controls how strongly the model sticks to the reference garment.",
            default=2.5,
        ),
        denoising_steps: int = Input(
            description="Number of denoising steps to apply to the final image",
            default=30,
        ),
        reference_unet_acceleration: bool = Input(
            description="If its false then the reference unet is going to be run for each denoising step, otherwise it will be run only once",
            default=False,
        ),
        stroke_width_mask: int = Input(
            description="Stroke width of the mask that is going to be applied to the final image",
            default=30,
        ),
        neck_offset_removal: int = Input(
            description="Amount in px of how much is the neck from the elbows",
            default=20,
        ),
        dilatation_kernel: list[int] = Input(
            description="Define an nxn matrix to apply dilatation to the mask",
            default=[10, 10],
        ),
        dilatation_iterations: int = Input(
            description="How many times is the dilatation kernel is going to be expanded",
            default=5,
        ),
        garment_type: str = Input(
            description="The type of garment that is going to be placed on the person",
            choices=["upper_body", "dresses", "lower_body"],
            default="upper_body",
        ),
    ) -> Iterator[Schiaparelli]:
        """Run a single prediction on the model"""
        ## Preprocessing
        output = Schiaparelli()
        src_image = Image.open(person)
        rf_image = Image.open(garment)
        src_image = resize_and_center(src_image, *self.resize_res)
        rf_image = resize_and_center(rf_image, *self.resize_res)
        model_parse, _ = self.human_parsing(src_image.resize(self.parsing_res))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            model_parse.save(tmp.name)
            human_parsing_path = Path(tmp.name)
            output.human_parsing = human_parsing_path
            yield output
        keypoints = self.open_pose(src_image.resize(self.parsing_res))
        mask, kp = agnostic_mask(
            model_parse,
            keypoints,
            garment_type,
            neck_offset_removal,
            tuple(dilatation_kernel),
            dilatation_iterations,
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            mask.save(tmp.name)
            mask_path = Path(tmp.name)
            output.mask = mask_path
            yield output

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            kp.save(tmp.name)
            kp_path = Path(tmp.name)
            output.open_pose = kp_path
            yield output

        ## Maps each detected human pixel to a 3D human body model (UV coordinates).
        # •IUV Map → Each pixel gets:
        # •I (Instance ID): Which body part the pixel belongs to (e.g., face, torso, left leg, etc.).
        # •U, V (Texture Coordinates): Where that pixel is located in 3D space.
        src_image_iuv_array = self.dense_pose.predict_iuv(np.array(src_image))
        src_image_seg_array = src_image_iuv_array[:, :, 0:1]
        src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
        src_image_seg = Image.fromarray(src_image_seg_array)
        densepose = src_image_seg

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            densepose.save(tmp.name)
            densepose_path = Path(tmp.name)
            output.dense_pose = densepose_path
            yield output
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
