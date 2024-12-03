import torch
import cv2
import requests
import numpy as np
from typing import Union, List, Dict
from pathlib import Path
from PIL import Image
from io import BytesIO

from scene_graph_benchmark.AttrRCNN import AttrRCNN
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.miscellaneous import set_seed

from .base import BaseFeatureExtractor
from scene_graph_benchmark.wrappers.utils import cv2Img_to_Image, encode_spatial_features


class VinVLFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor using the VinVL model.
    Supports single or batched inputs (paths, URLs, or image arrays/tensors).
    """

    def __init__(self, config_file: str, weight_file: str, device: str = "cuda"):
        """
        Initialize the VinVL feature extractor.

        Args:
            config_file (str): Path to the VinVL configuration file.
            weight_file (str): Path to the pretrained weights file.
            device (str): Device to run the model (e.g., "cuda" or "cpu").
        """
        set_seed(1000, torch.cuda.device_count())
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load configuration and model
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHT = weight_file
        cfg.MODEL.DEVICE = str(self.device)
        cfg.TEST.OUTPUT_FEATURE = True
        cfg.freeze()

        if cfg.MODEL.META_ARCHITECTURE != "AttrRCNN":
            raise ValueError(f"Invalid META_ARCHITECTURE: {cfg.MODEL.META_ARCHITECTURE}. Expected 'AttrRCNN'.")

        self.model = AttrRCNN(cfg)
        self.model.to(self.device).eval()
        self.transforms = build_transforms(cfg, is_train=False)

    def __call__(self, inputs: Union[str, List[Union[str, np.ndarray, torch.Tensor]]]) -> List[Dict]:
        """
        Make the extractor callable to handle various input formats.

        Args:
            inputs (Union[str, List[str], List[np.ndarray], List[torch.Tensor]]): 
                Path(s), URL(s), image array(s), or tensor(s) representing images.

        Returns:
            List[Dict]: Extracted features for each input.
        """
        return self.extract_features(inputs)

    def _load_image(self, image: Union[str, np.ndarray, torch.Tensor]) -> Image.Image:
        """
        Load an image from a path, URL, array, or tensor.

        Args:
            image (Union[str, np.ndarray, torch.Tensor]): Image path, URL, array, or tensor.

        Returns:
            Image.Image: Loaded image as a PIL Image.
        """
        if isinstance(image, str):  # Path or URL
            if image.startswith("http"):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):  # NumPy array
            image = Image.fromarray(image[..., ::-1])  # Convert BGR to RGB
        elif isinstance(image, torch.Tensor):  # PyTorch tensor
            image = Image.fromarray(image.cpu().numpy().transpose(1, 2, 0))  # CHW -> HWC
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")
        return image

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """
        Process a single image into the model's input format.

        Args:
            image (Image.Image): PIL image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_input = cv2Img_to_Image(cv2_img)
        img_input, _ = self.transforms(img_input, target=None)
        return img_input.to(self.device)

    def _extract_single(self, image: Union[str, np.ndarray, torch.Tensor]) -> Dict:
        """
        Extract features for a single image.

        Args:
            image (Union[str, np.ndarray, torch.Tensor]): Image path, URL, or array/tensor.

        Returns:
            Dict: Extracted features including boxes, classes, scores, etc.
        """
        pil_image = self._load_image(image)
        img_tensor = self._process_image(pil_image)

        with torch.no_grad():
            predictions = self.model([img_tensor])[0].to("cpu")

        # Extract details
        img_width, img_height = pil_image.size
        predictions = predictions.resize((img_width, img_height))
        boxes = predictions.bbox.tolist()
        classes = predictions.get_field("labels").tolist()
        scores = predictions.get_field("scores").tolist()
        features = predictions.get_field("box_features").cpu().numpy()
        spatial_features = encode_spatial_features(features, (img_width, img_height), mode="xyxy")

        return {
            "boxes": boxes,
            "classes": classes,
            "scores": scores,
            "features": features,
            "spatial_features": spatial_features,
        }

    def extract_features(self, inputs: Union[str, List[Union[str, np.ndarray, torch.Tensor]]]) -> List[Dict]:
        """
        Extract features from single or multiple inputs.

        Args:
            inputs (Union[str, List[Union[str, np.ndarray, torch.Tensor]]]): Image paths, URLs, arrays, or tensors.

        Returns:
            List[Dict]: Extracted features for each input.
        """
        if isinstance(inputs, (str, np.ndarray, torch.Tensor)):
            inputs = [inputs]

        return [self._extract_single(image) for image in inputs]
