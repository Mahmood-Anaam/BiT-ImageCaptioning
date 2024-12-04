import torch
import requests
import numpy as np
from typing import Union, List, Dict
from PIL import Image
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
import json


from  .scene_graph_benchmark.scene_graph_benchmark.AttrRCNN import AttrRCNN
from  .scene_graph_benchmark.scene_graph_benchmark.config import sg_cfg
from  .scene_graph_benchmark.maskrcnn_benchmark.config import cfg
from  .scene_graph_benchmark.maskrcnn_benchmark.data.transforms import build_transforms
from  .scene_graph_benchmark.maskrcnn_benchmark.utils.miscellaneous import set_seed

from .base import BaseFeatureExtractor
from .scene_graph_benchmark.maskrcnn_benchmark.config import cfg
from .scene_graph_benchmark.scene_graph_benchmark.wrappers.utils import cv2Img_to_Image, encode_spatial_features






class VinVLFeatureExtractor:
    """
    VinVL Feature Extractor for extracting visual features from images.
    Supports various input types (file path, URL, PIL.Image, numpy array, and torch.Tensor).
    """

    BASE_PATH = Path(__file__).parent.parent.parent
    CONFIG_FILE = Path(BASE_PATH, 'sgg_configs/vgattr/vinvl_x152c4.yaml')
    MODEL_DIR = Path(BASE_PATH, "models/vinvl_vg_x152c4")
    MODEL_URL = "https://huggingface.co/michelecafagna26/vinvl_vg_x152c4/resolve/main/vinvl_vg_x152c4.pth"
    LABEL_URL = "https://huggingface.co/michelecafagna26/vinvl_vg_x152c4/resolve/main/VG-SGG-dicts-vgoi6-clipped.json"

    def __init__(self, config_file=None, opts=None, device="cuda"):
        """
        Initializes the VinVL Feature Extractor.

        Args:
            config_file (str, optional): Path to the configuration file.
            opts (dict, optional): Additional configuration options.
            device (str, optional): Device to run the model on ("cuda" or "cpu").
        """
        # Set random seed and device
        num_of_gpus = torch.cuda.device_count()
        set_seed(1000, num_of_gpus)
        self.device = device

        # Load default or custom options
        self.opts = {
            "MODEL.WEIGHT": str(self.MODEL_DIR / "vinvl_vg_x152c4.pth"),
            "MODEL.ROI_HEADS.NMS_FILTER": 1,
            "MODEL.ROI_HEADS.SCORE_THRESH": 0.2,
            "TEST.IGNORE_BOX_REGRESSION": False,
            "DATASETS.LABELMAP_FILE": str(self.MODEL_DIR / "VG-SGG-dicts-vgoi6-clipped.json"),
            "TEST.OUTPUT_FEATURE": True
        }
        if opts:
            self.opts.update(opts)

        # Load the configuration file
        self.config_file = config_file or self.CONFIG_FILE
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.merge_from_file(self.config_file)
        cfg.update(self.opts)
        cfg.set_new_allowed(False)
        cfg.freeze()

        # Initialize the model
        if cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
            self.model = AttrRCNN(cfg)
        else:
            raise ValueError("MODEL.META_ARCHITECTURE must be 'AttrRCNN'.")

        self.model.eval()
        self.model.to(self.device)

        # Download model and label files if not present
        self._download_model_and_labels()

        # Load the model weights
        self.checkpointer = DetectronCheckpointer(cfg, self.model, save_dir="")
        self.checkpointer.load(str(self.MODEL_DIR / "vinvl_vg_x152c4.pth"))

        # Load label maps
        with open(self.MODEL_DIR / "VG-SGG-dicts-vgoi6-clipped.json", "rb") as fp:
            label_dict = json.load(fp)
        self.idx2label = {int(k): v for k, v in label_dict["idx_to_label"].items()}
        self.label2idx = {k: int(v) for k, v in label_dict["label_to_idx"].items()}

        # Initialize image transforms
        self.transforms = build_transforms(cfg, is_train=False)

    def _download_model_and_labels(self):
        """Downloads the model weights and label files if they do not exist."""
        if not (self.MODEL_DIR / "vinvl_vg_x152c4.pth").is_file():
            self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

            # Download model weights
            self._download_file(self.MODEL_URL, self.MODEL_DIR / "vinvl_vg_x152c4.pth")

            # Download label map
            self._download_file(self.LABEL_URL, self.MODEL_DIR / "VG-SGG-dicts-vgoi6-clipped.json")

    @staticmethod
    def _download_file(url, destination):
        """Helper function to download a file."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(destination, "wb") as f, tqdm(
            desc=f"Downloading {destination.name}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))

    def _prepare_image(self, img):
        """
        Prepares an image for feature extraction.

        Args:
            img: Input image (file path, URL, PIL.Image, numpy array, or tensor).

        Returns:
            PIL.Image: The prepared PIL.Image in RGB format.
        """
        if isinstance(img, str):
            # File path or URL
            if img.startswith("http://") or img.startswith("https://"):
                response = requests.get(img)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            # PIL.Image
            img = img.convert("RGB")
        elif isinstance(img, np.ndarray):
            # NumPy array (assume RGB)
            img = Image.fromarray(img)
        elif torch.is_tensor(img):
            # Tensor (assume CHW format)
            img = img.permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC
            img = Image.fromarray(np.uint8(img))
        else:
            raise ValueError("Unsupported image input type.")
        return img

    def __call__(self, imgs):
        """
        Extract features from images.

        Args:
            imgs: Single image or a batch of images (list or single instance).

        Returns:
            List[dict]: List of extracted features for each image.
        """
        if not isinstance(imgs, list):
            imgs = [imgs]  # Convert to batch format

        results = []
        for img in imgs:
            # Prepare the image
            img = self._prepare_image(img)

            # Apply transforms
            img_tensor, _ = self.transforms(img, target=None)
            img_tensor = img_tensor.to(self.device)

            # Perform inference
            with torch.no_grad():
                prediction = self.model(img_tensor)
                prediction = prediction[0].to("cpu")

            # Scale predictions to original image size
            img_width, img_height = img.size
            prediction = prediction.resize((img_width, img_height))

            # Extract features
            boxes = prediction.bbox.tolist()
            classes = [self.idx2label[c] for c in prediction.get_field("labels").tolist()]
            scores = prediction.get_field("scores").tolist()
            features = prediction.get_field("box_features").cpu().numpy()
            spatial_features = encode_spatial_features(features, (img_width, img_height), mode="xyxy")

            results.append({
                "boxes": boxes,
                "classes": classes,
                "scores": scores,
                "features": features,
                "spatial_features": spatial_features,
            })

        return results

