# Deep-Learning-image-classification-project
## Overview

This repository contains a Python script for training and evaluating neural network models on image classification tasks using TensorFlow and Keras. The script includes implementations for a Deep Neural Network (DNN) for binary classification, a Convolutional Neural Network (CNN) for binary classification, and a more complex CNN for multiclass classification.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- scipy

## Usage

1. **Clone the Repository:**

git clone https://github.com/AzzedineNed/Deep-Learning-image-classification-project.git

cd Deep-Learning-image-classification-project

2. **Run the Script:**

Execute the script by running:

python main.py


3. **Follow the Console Prompts:**

The script will guide you through loading the datasets, preprocessing images, training models, and evaluating performance.

## Dataset

The dataset consists of images from two categories for binary classification tasks and four categories for multiclass classification. The script provides functions to load and preprocess these datasets, including applying Fast Fourier Transform (FFT) and data augmentation.

The dataset is organized as follows:

dataset/
├── ernest_celestine/
├── toy_story_1/
├── toy_story_2/
├── toy_story_3/
Each subdirectory contains images for the respective categories. The images are of size 920x540.

## Training and Evaluation

The script includes training and evaluation procedures for each implemented model. You can adjust hyperparameters, model architectures, and training configurations in the script as needed.

## Results

### DNN Binary Classification

- Training Loss: 0.0000
- Training Accuracy: 1.0000
- Validation Loss: 0.0093
- Validation Accuracy: 1.0000
- Test Loss: 0.3026
- Test Accuracy: 0.9800

![DNN Accuracy Plot](dnn_accuracy_plot.png) <!-- Placeholder for the accuracy plot image -->

![DNN Loss Plot](dnn_loss_plot.png) <!-- Placeholder for the loss plot image -->

### CNN Binary Classification

- Training Loss: 0.1017
- Training Accuracy: 0.9625
- Validation Loss: 0.0846
- Validation Accuracy: 0.9800
- Test Loss: 0.0650
- Test Accuracy: 0.9900

![CNN Binary Accuracy Plot](cnn_binary_accuracy_plot.png) <!-- Placeholder for the accuracy plot image -->

![CNN Binary Loss Plot](cnn_binary_loss_plot.png) <!-- Placeholder for the loss plot image -->

<!-- Add descriptions or captions as needed -->

### CNN 4 Classes Classification

- Training Loss: 0.0559
- Training Accuracy: 0.9819
- Validation Loss: 0.1396
- Validation Accuracy: 0.9550
- Test Loss: 0.1819
- Test Accuracy: 0.9450

![CNN 4 Classes Accuracy Plot](cnn_4_classes_accuracy_plot.png) <!-- Placeholder for the accuracy plot image -->

![CNN 4 Classes Loss Plot](cnn_4_classes_loss_plot.png) <!-- Placeholder for the loss plot image -->

<!-- Add descriptions or captions as needed -->


## Contributing

Contributions are welcome! If you find issues or have suggestions for improvements, feel free to open an issue or submit a pull request. 

## License

This project is licensed under the [MIT License](LICENSE).


