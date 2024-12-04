from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torch
from ..tokenizers.bert_tokenizer import CaptionTensorizer

class OKVQADataset(Dataset):
    """
    Dataset class for handling the OK-VQA dataset with image features and multilingual questions/answers.
    """

    def __init__(self, 
                 path = "MahmoodAnaam/ok-vqa-ar-en-2",
                 feature_extractor=None, 
                 tokenizer=None,
                 language="ar", 
                 split="validation", 
                 add_od_labels=True,
                 max_img_seq_length=50,
                 max_seq_length=70,
                 max_seq_a_length=40, 
                 is_train=False, 
                 mask_prob=0.15, 
                 max_masked_tokens=3, 
                 **kwargs):
        """
        Constructor for initializing the dataset.

        Args:
            path: datset path.
            feature_extractor: Feature extractor object for processing image features.
            tokenizer: Tokenizer for processing text (questions, answers).
            language (str): Desired language for questions/answers ("ar" for Arabic or "en" for English).
            split (str): Dataset split to load ("train", "validation", or "test").
            add_od_labels (bool): Whether to add object detection (OD) labels to text.
            max_img_seq_length (int): Maximum length of image sequence (number of features per image).
            max_seq_length (int): Maximum length of combined text sequence.
            max_seq_a_length (int): Maximum length of caption sequence.
            is_train (bool): Whether the dataset is used for training.
            mask_prob (float): Probability of masking tokens for training.
            max_masked_tokens (int): Maximum number of tokens to mask in a sentence.
            **kwargs: Additional arguments.
        """
        # Validate the language input (must be either Arabic or English)
        if language not in ["ar", "en"]:
            raise ValueError("Language must be 'ar' (Arabic) or 'en' (English).")
        
        # Validate the dataset split (must be "train", "validation", or "test")
        if split not in ["train", "validation", "test"]:
            raise ValueError("Split must be 'train', 'validation', or 'test'.")
        
        # Ensure a feature extractor is provided
        if feature_extractor is None:
            raise ValueError("Feature extractor must be provided.")
        
        # Ensure a tokenizer is provided
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")

        # Initialize class properties
        self.path = path
        self.language = language
        self.split = split
        self.add_od_labels = add_od_labels
        self.is_train = is_train
        self.kwargs = kwargs

        # Load the dataset split using the Hugging Face `datasets` library
        self.dataset = load_dataset("MahmoodAnaam/ok-vqa-ar-en-2", split=split,cache_dir='./OKVQA')

        # Set the feature extractor and tokenizer
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        # Initialize a CaptionTensorizer for preparing text and image inputs for the model
        self.caption_tokenizer = CaptionTensorizer(
            self.tokenizer,
            max_img_seq_length=max_img_seq_length,
            max_seq_length=max_seq_length,
            max_seq_a_length=max_seq_a_length,
            mask_prob=mask_prob,
            max_masked_tokens=max_masked_tokens,
            is_train=is_train
        )

    def get_image_features(self, idx):
        """
        Extract image features and object detection labels (if applicable) for a given index.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            object_detections (dict): Details from the feature extractor, including bounding boxes, classes, scores, features, and spatial features.
            image_features (torch.Tensor): Extracted image features as a tensor.
            od_labels (str): Concatenated object detection labels (if applicable).
        """
        # Load the image from the dataset
        image = self.dataset[idx]["image"]
        
        # Use the feature extractor to get object detection details
        object_detections = self.feature_extractor([image])[0]
        
        # Combine visual features and spatial features into a single array
        v_feats = np.concatenate((object_detections['features'], object_detections['spatial_features']), axis=1)
        image_features = torch.Tensor(np.array(v_feats))  # Convert features to PyTorch tensor

        # If OD labels are enabled, combine all detected classes into a single string
        if self.add_od_labels:
            od_labels = " ".join(object_detections['classes'])
        else:
            od_labels = None

        return object_detections, image_features, od_labels

    def __getitem__(self, idx):
        """
        Retrieve an example from the dataset at a given index.

        Args:
            idx (int): Index of the example.

        Returns:
            idx (int): The index of the example.
            example (dict): A dictionary containing all relevant data for the model.
        """
        # Retrieve the sample from the dataset
        sample = self.dataset[idx]

        # Extract image features and object detection labels
        object_detections, image_features, od_labels = self.get_image_features(idx)

        # Use the CaptionTensorizer to prepare the inputs for the model
        inputs = self.caption_tokenizer.tensorize_example(
            text_a=None,
            img_feat=image_features,
            text_b=od_labels
        )

        # Construct the example dictionary containing all the processed data
        example = {
            "metadata": sample["metadata"],  # Metadata (e.g., image_id, question_id, etc.)
            "image": sample["image"],  # Original PIL image
            "question": sample["question"].get(self.language),  # Question in the specified language
            "answers": sample["answers"].get(self.language),  # Answers in the specified language
            "raw_answers": sample["answers"].get(f"raw_{self.language}"),  # Raw answers
            "confidence_answers": sample["answers"].get("confidence"),  # Confidence levels for each answer
            "object_detections": object_detections,  # Full details from the feature extractor
            "inputs": inputs  # Prepared inputs for the model
        }

        return idx, example

    def __len__(self):
        """
        Return the total number of examples in the dataset.

        Returns:
            int: Total number of examples in the dataset.
        """
        return len(self.dataset)
    
