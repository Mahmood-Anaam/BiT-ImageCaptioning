# BiT-ImageCaptioning

**BiT-ImageCaptioning** is a Python package for **image captioning in Arabic** using **Bidirectional Transformers (BiT)**. It provides tools for generating high-quality captions for Arabic datasets, leveraging cutting-edge pre-trained models.



## Installation

```bash
!git clone https://github.com/Mahmood-Anaam/BiT-ImageCaptioning.git
%cd BiT-ImageCaptioning

!pip install -e .
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

