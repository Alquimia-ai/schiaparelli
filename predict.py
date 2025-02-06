# Prediction interface for Cog ⚙️
# https://cog.run/python
from cog import BasePredictor, Input, Path
import json
from preprocessing.humanparsing.run_parsing import HumanParsing
from PIL import Image
import tempfile


class Predictor(BasePredictor):
    def setup(self) -> None:  # type: ignore
        """Load the model into memory to make running multiple predictions efficient"""
        with open("config.json", "r") as file:
            self.config = json.load(file)
            self.parsing_res = tuple(self.config["parsing_res"])

        ## Preprocessing Part (HumanParsing, OpenPose & DensePose)
        self.human_parsing = HumanParsing(
            atr_path=self.config["human_parsing"]["atr_path"],
            lip_path=self.config["human_parsing"]["lip_path"],
        )

    def predict(  # type: ignore
        self,
        person: Path = Input(
            description="The person garment that is going to be replaced"
        ),
        garment: Path = Input(
            description="The garment that is going to be placed on the person"
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        src_image = Image.open(person)
        model_parse, _ = self.human_parsing(src_image.resize(self.parsing_res))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            model_parse.save(tmp.name)
            output_path = Path(tmp.name)

        return output_path
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
