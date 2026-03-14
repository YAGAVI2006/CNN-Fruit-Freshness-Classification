# CNN Fruit Freshness Classification

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify fruits and vegetables as either Fresh or Rotten.

## Project Overview

The model uses deep learning to analyze images of fruits and vegetables and determine their freshness state. It includes data augmentation, attention mechanisms, and comprehensive evaluation metrics.

## Features

- **CNN Architecture**: Custom CNN with convolutional layers, max pooling, dropout, and dense layers
- **Attention Mechanism**: Channel attention blocks to improve feature emphasis
- **Data Augmentation**: Rotation, horizontal flip, zoom, and brightness adjustments
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix
- **Visualization**: Training accuracy and loss graphs
- **Prediction Script**: Command-line tool for single image classification

## Dataset Structure

The project expects the dataset to be organized as follows:

```
dataset/
├── train/
│   ├── fresh/
│   └── rotten/
├── validation/
│   ├── fresh/
│   └── rotten/
└── test/
    ├── fresh/
    └── rotten/
```

## Requirements

- Python 3.8+
- TensorFlow 2.13.0
- Pillow 10.0.0
- Matplotlib 3.7.1
- Scikit-learn 1.3.0
- NumPy 1.24.3

## Installation

1. Clone or download the project files.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Ensure your dataset is placed in the `dataset/` directory with the correct structure.
2. Run the training script:

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train the CNN model
- Save the trained model as `model.h5`
- Generate training history plots in the `results/` folder
- Evaluate the model on test data

### Making Predictions

To classify a single image:

```bash
python predict.py path/to/your/image.jpg
```

Replace `path/to/your/image.jpg` with the actual path to your image file.

## Model Architecture

The CNN consists of:
- 3 convolutional blocks with increasing filters (32, 64, 128)
- Channel attention mechanisms after the first two blocks
- Max pooling and dropout for regularization
- Dense layers with dropout
- Softmax output for binary classification

## Results

After training, the model achieves:
- Training and validation accuracy/loss plots saved as `results/training_history.png`
- Test accuracy and detailed classification metrics printed to console
- Confusion matrix for detailed analysis

## Notes

- The model is trained for 20 epochs by default. Adjust `EPOCHS` in `train_model.py` as needed.
- Images are resized to 224x224 pixels and normalized to [0,1].
- The attention mechanism helps focus on important features for better classification.
- Ensure your dataset has sufficient images for training (recommended: at least 1000 images per class).

## License

This project is for educational purposes. Feel free to modify and use as needed.