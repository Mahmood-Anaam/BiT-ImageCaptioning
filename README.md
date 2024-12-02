# BiT-ImageCaptioning

**BiT-ImageCaptioning** is a Python package for generating **Arabic image captions** using **Bidirectional Transformers (BiT)**. This library is designed to provide high-quality and accurate captions for Arabic datasets by leveraging pre-trained deep learning models.


## Installation

Clone the repository and install the package locally:

```bash
git clone https://github.com/Mahmood-Anaam/BiT-ImageCaptioning.git
cd BiT-ImageCaptioning
pip install -e .
```

If you're working in a Jupyter Notebook, restart the environment after installation:

```python
import os
os.kill(os.getpid(), 9)
```



## Quick Start

```python
from BiTImageCaptioning.generation import generate_caption
from BiTImageCaptioning.feature_extraction import ImageFeatureExtractor

# Extract image features
extractor = ImageFeatureExtractor()
image_features = extractor.extract_features("path_to_image.jpg")

# Generate a caption
caption = generate_caption(model="path_to_model", image_features=image_features)
print("Generated Caption:", caption)
```



## Package Architecture

The package is modularly designed to make it easy to understand, extend, and use. Below is the file structure of the package:

```plaintext
BiT-ImageCaptioning/
├── src/
│   ├── BiTImageCaptioning/
│   │   ├── __init__.py               # Initialization file for the package
│   │   ├── configuration.py          # Handles model configurations
│   │   ├── modeling.py               # Contains the Transformer-based model
│   │   ├── processing.py             # Handles tokenization and text processing
│   │   ├── feature_extraction.py     # Handles image feature extraction
│   │   ├── dataset.py                # Prepares datasets for training and evaluation
│   │   ├── generation.py             # Contains caption generation logic
│   │   └── utils.py                  # Helper functions and utilities
├── tests/                            # Unit tests for the package
│   ├── test_modeling.py              # Tests for the model logic
│   ├── test_feature_extraction.py    # Tests for feature extraction
│   ├── test_processing.py            # Tests for text processing
│   └── test_generation.py            # Tests for caption generation
├── examples/                         # Usage examples for the package
│   ├── example_generate_caption.py   # Example: Generate captions for an image
│   ├── example_finetune.py           # Example: Fine-tuning the model
│   └── example_evaluate.py           # Example: Evaluate model performance
├── README.md                         # Documentation for the package
├── LICENSE                           # License file
├── setup.py                          # Setup script for installation
└── requirements.txt                  # List of dependencies
```



## Core Components

### 1. Feature Extraction
- File: `feature_extraction.py`
- Extracts features from images using pre-trained models.

```python
from BiTImageCaptioning.feature_extraction import ImageFeatureExtractor

extractor = ImageFeatureExtractor()
image_features = extractor.extract_features("path_to_image.jpg")
```

### 2. Caption Generation
- File: `generation.py`
- Generates captions based on extracted image features.

```python
from BiTImageCaptioning.generation import generate_caption

caption = generate_caption(model="path_to_model", image_features=image_features)
print("Generated Caption:", caption)
```

### 3. Configuration and Utilities
- File: `configuration.py`
- Manages model configurations and utility functions.






