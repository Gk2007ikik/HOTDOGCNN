# Hot Dog CNN 🌭

A convolutional neural network that classifies images as **hot dog** or **not hot dog**, trained on the [Food101](https://www.tensorflow.org/datasets/catalog/food101) dataset via `tensorflow_datasets`.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gk2007ikik/git_test/blob/main/HOTDOGCNN.ipynb)

## Overview

Food101 contains 101 food categories, one of which is `hot_dog`. This project reframes the dataset as a binary classification problem — hot dog vs. everything else — and trains a small CNN from scratch to solve it.

Pipeline:
1. Load Food101 with `tensorflow_datasets`, resize images to 128×128.
2. Relabel every image as `1` (hot dog) or `0` (not hot dog).
3. Balance classes by oversampling the hot dog class and sampling 50/50 from each class per batch.
4. Apply data augmentation (random horizontal flip, random rotation).
5. Train a `Conv2D` + `MaxPooling2D` + `Dropout` stack with a sigmoid/binary-crossentropy output.
6. Visualize predictions on validation samples.

## Repository structure

```
.
├── HOTDOGCNN.ipynb   # Main notebook: data loading, model, training, evaluation
├── requirements.txt  # Python dependencies for running locally
├── LICENSE
└── README.md
```

## Getting started

### Option 1: Google Colab (recommended)
Click the "Open in Colab" badge above. The first cell upgrades `protobuf`; restart the runtime once, then run the rest of the notebook top to bottom. A GPU runtime (`Runtime > Change runtime type > GPU`) is recommended for training speed.

### Option 2: Run locally
```bash
git clone https://github.com/Gk2007ikik/git_test.git
cd git_test
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook HOTDOGCNN.ipynb
```

> Note: Food101 is a large dataset (~5 GB) and will be downloaded automatically by `tensorflow_datasets` on first run.

## Model

A sequential CNN:
- Rescaling (1/255) + data augmentation (flip, rotation)
- `Conv2D(128) → MaxPool → Dropout`
- `Conv2D(64, L2-regularized) → ...`
- Dense output with binary crossentropy loss, Adam optimizer (`lr=1e-4`)
- Trained for 50 epochs

## Results

After training, the notebook plots a batch of validation images alongside their true labels so you can visually sanity-check predictions. (Add your accuracy/loss curves and final metrics here once you have a training run you're happy with.)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
