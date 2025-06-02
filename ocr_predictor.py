import os
import sys

PADDLE_OCR_PATH = "../PaddleOCR"
if PADDLE_OCR_PATH not in sys.path:
    sys.path.insert(0, PADDLE_OCR_PATH)

import yaml
import numpy as np
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model


class PaddleOCRRecognizer:
    def __init__(self, config_path, model_path):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Set inference mode and model path
        self.config['Global']['pretrained_model'] = model_path
        self.config['Global']['infer_mode'] = True

        # Build post process class
        self.post_process = build_post_process(self.config["PostProcess"], self.config["Global"])

        # Set correct number of output classes
        if hasattr(self.post_process, "character"):
            char_num = len(getattr(self.post_process, "character"))
            self.config["Architecture"]["Head"]["out_channels"] = char_num

        # Build model and load weights
        self.model = build_model(self.config["Architecture"])
        load_model(self.config, self.model)
        self.model.eval()

        # Create image preprocessing pipeline (operators)
        self.ops = create_operators(
            self._build_transforms(self.config["Eval"]["dataset"]["transforms"]),
            self.config["Global"]
        )

    def _build_transforms(self, transforms_config):
        transforms = []
        for op in transforms_config:
            op_name = list(op.keys())[0]
            if "Label" in op_name:
                continue  # skip label-related transforms
            if op_name == "RecResizeImg":
                op[op_name]["infer_mode"] = True
            if op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image"]
            transforms.append(op)
        return transforms

    def predict(self, image_path):
        # Load image bytes
        with open(image_path, "rb") as f:
            img = f.read()
        data = {"image": img}

        # Apply preprocessing
        batch = transform(data, self.ops)
        image = paddle.to_tensor(np.expand_dims(batch[0], axis=0))

        # Model inference
        preds = self.model(image)
        result = self.post_process(preds)

        # Format result
        if isinstance(result, list) and len(result[0]) >= 2:
            return {"text": result[0][0], "score": float(result[0][1])}
        return {"text": "", "score": 0.0}


# Example usage
if __name__ == "__main__":
    config_path = "./config.yaml"
    model_path = "./model/model/best_accuracy"
    image_path = "img_00265.png"

    recognizer = PaddleOCRRecognizer(config_path, model_path)
    result = recognizer.predict(image_path)
    print("Recognized:", result)
