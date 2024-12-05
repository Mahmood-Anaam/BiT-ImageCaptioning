import torch
import numpy as np
from torch.nn import Module
from pytorch_transformers import BertTokenizer, BertConfig
from bit_image_captioning.modeling.modeling_bert import BertForImageCaptioning
from bit_image_captioning.tokenizers.bert_tokenizer import CaptionTensorizer
from bit_image_captioning.feature_extractors.vinvl import VinVLFeatureExtractor
from tqdm import tqdm

class BiTImageCaptioningPipeline(Module):
    """
    Pipeline for image captioning using a BERT-based model.
    This class integrates image feature extraction, tokenization, and model inference.
    """

    def __init__(self, cfg):
        """
        Initializes the pipeline with the specified configuration.

        Args:
            cfg (BiTConfig): Configuration object containing all settings for the pipeline.
        """
        super().__init__()

        # Load configuration, tokenizer, and model
        self.config = BertConfig.from_pretrained(cfg.checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained(cfg.checkpoint)
        self.caption_tensorizer = CaptionTensorizer(
            tokenizer=self.tokenizer,
            is_train=cfg.is_train
        )
        self.feature_extractor = VinVLFeatureExtractor()
        self.model = BertForImageCaptioning.from_pretrained(cfg.checkpoint, config=self.config)

        # Move model to the configured device
        self.model.to(cfg.device)
        self.model.eval()

        self.cfg = cfg

        # Automatically set token IDs from the tokenizer
        cfg.set_token_ids(self.tokenizer)

    def get_image_features(self, images):
        """
        Extract image features and object detection labels for a batch of images.

        Args:
            images (list of PIL.Image, np.ndarray, or str): List of images to process.

        Returns:
            list of dict: Object detection details for each image.
            torch.Tensor: Extracted image features as a tensor (batch size, num_features, feature_dim).
            list of str: Concatenated object detection labels for each image (if applicable).
        """
        try:
            # Use the feature extractor to process the images
            object_detections = self.feature_extractor(images)

            # Combine features for all images
            image_features = []
            od_labels = []
            for detection in object_detections:
                # Combine visual features and spatial features
                v_feats = np.concatenate(
                    (detection['features'], detection['spatial_features']),
                    axis=1
                )
                image_features.append(torch.Tensor(v_feats))

                # Generate object detection labels if enabled
                if self.cfg.add_od_labels:
                    od_labels.append(" ".join(detection['classes']))
                else:
                    od_labels.append(None)

            # Stack features into a batch tensor
            image_features = torch.stack(image_features)

            return object_detections, image_features, od_labels

        except Exception as e:
            raise RuntimeError(f"Failed to extract features for the images: {e}")

    def prepare_inputs(self, images):
        """
        Prepare the inputs for the model using the CaptionTensorizer.

        Args:
            images (list of PIL.Image, np.ndarray, or str): List of images to process.

        Returns:
            dict: Prepared inputs for the model.
        """
        try:
            # Extract image features and object detection labels
            object_detections, image_features, od_labels = self.get_image_features(images)

            # Use the CaptionTensorizer to create tokenized inputs for the batch
            input_ids, attention_masks, token_type_ids, img_feats, masked_pos = [], [], [], [], []
            for i, image_feat in enumerate(image_features):
                # Tensorize each example
                ids, attn, token_type, img_feat, masked = self.caption_tensorizer.tensorize_example(
                    text_a=None,
                    img_feat=image_feat,
                    text_b=od_labels[i]
                )
                input_ids.append(ids)
                attention_masks.append(attn)
                token_type_ids.append(token_type)
                img_feats.append(img_feat)
                masked_pos.append(masked)

            # Stack all inputs into batch tensors
            inputs = {
                "input_ids": torch.stack(input_ids).to(self.cfg.device),
                "attention_mask": torch.stack(attention_masks).to(self.cfg.device),
                "token_type_ids": torch.stack(token_type_ids).to(self.cfg.device),
                "img_feats": torch.stack(img_feats).to(self.cfg.device),
                "masked_pos": torch.stack(masked_pos).to(self.cfg.device)
            }

            return inputs

        except Exception as e:
            raise RuntimeError(f"Failed to prepare inputs for the images: {e}")

    def generate_captions(self, images):
        """
        Generate captions for a batch of images.

        Args:
            images (list of PIL.Image, np.ndarray, or str): List of images to generate captions for.

        Returns:
            list of list of dict: List of generated captions and confidence scores for each image.
        """
        try:
            # Prepare inputs for the model
            inputs = self.prepare_inputs(images)

            # Add generation parameters
            inputs.update({
                "is_decode": True,
                "do_sample": False,
                "bos_token_id": self.cfg.cls_token_id,
                "pad_token_id": self.cfg.pad_token_id,
                "eos_token_ids": [self.cfg.sep_token_id],
                "mask_token_id": self.cfg.mask_token_id,
                "add_od_labels": self.cfg.add_od_labels,
                "od_labels_start_posid": self.cfg.max_seq_a_length,
                "max_length": self.cfg.max_gen_length,
                "num_beams": self.cfg.num_beams,
                "temperature": self.cfg.temperature,
                "top_k": self.cfg.top_k,
                "top_p": self.cfg.top_p,
                "repetition_penalty": self.cfg.repetition_penalty,
                "length_penalty": self.cfg.length_penalty,
                "num_return_sequences": self.cfg.num_return_sequences,
                "num_keep_best": self.cfg.num_keep_best
            })

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract captions and confidence scores
            all_captions = outputs[0]  # batch_size * num_keep_best * max_len
            all_confidences = torch.exp(outputs[1])  # Confidence scores

            # Decode the generated captions for each image
            results = []
            for caps, confs in zip(all_captions, all_confidences):
                image_results = []
                for cap, conf in zip(caps, confs):
                    caption = self.tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                    image_results.append({"caption": caption, "confidence": conf.item()})
                results.append(image_results)

            return results

        except Exception as e:
            raise RuntimeError(f"Failed to generate captions for the images: {e}")
