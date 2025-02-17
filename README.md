# Digit Recognition with CNN on MNIST Dataset
# Handwritten Digit Recognition using CNN

This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to recognize handwritten digits from the MNIST dataset.

## Project Overview

- **Data Preparation**: Load, reshape, and normalize the MNIST dataset.
- **Model Architecture**: Create a CNN model with convolutional, pooling, and fully connected dense layers.
- **Training**: Train the model on the training data and record performance metrics.
- **Evaluation & Visualization**: Predict digits from test images and plot accuracy/loss curves.
- **Model Saving & Loading**: Save the trained model as an `.h5` file and reload it for predictions.
- **Predictions**: Make predictions for individual and multiple test images.

## Code Details

### Libraries

- **tensorflow** and **keras**: For building and training the CNN model.
- **matplotlib** and **numpy**: For data manipulation and visualization.

### Model Preparation

#### Load Dataset

Load the MNIST dataset, consisting of 28x28 grayscale images of handwritten digits (0-9).

#### Reshape and Normalize

Reshape data to a 4D tensor format and normalize pixel values.

### Model Architecture

A sequential CNN model with the following layers:

- **Convolutional layers**: With ReLU activation.
- **MaxPooling layers**: To reduce spatial dimensions.
- **Flatten layer**: To convert 2D matrices to a vector.
- **Dense layers**: With ReLU activation for fully connected layers.
- **Output layer**: With sigmoid activation to predict class probabilities.

### Model Compilation

The model uses:

- **Optimizer**: Adam.
- **Loss**: Sparse Categorical Crossentropy.
- **Metrics**: Accuracy.

### Training

Train the model for 10 epochs using the training images and labels.

### Visualization of Training Results

Plot accuracy and loss for each epoch using Matplotlib to visualize model performance.

### Saving and Loading Model

Save the model to `Digit_CNN.h5` after training and reload it for making predictions.

### Making Predictions

#### Single Image Prediction

Predict the digit for a single test image.

#### Multiple Images Prediction

Predict digits for a batch of test images.

## Instructions

### Requirements

Install the following Python libraries before running the code:
- h5py
- matplotlib
- numpy
- tensorflow

### Running the Project

1. **Run the Script**: Execute the script to load the dataset, train the model, and save the trained model as `Digit_CNN.h5`.
2. **View Results**: After training, the model accuracy and loss graphs will display.
3. **Load Model for Prediction**: Reload the model and make predictions for single and multiple test images, displaying the predicted digit above each image.

### Code Structure

1. **Data Loading & Preprocessing**: Load MNIST data, reshape, and normalize it.
2. **Model Creation**: Define the CNN model architecture.
3. **Training the Model**: Compile and train the model for 10 epochs.
4. **Model Evaluation & Visualization**: Visualize accuracy and loss, then save the model.
5. **Prediction**: Make predictions for individual and multiple test images, showing the model’s output alongside the test images.

### Example Output

After running the script, you should see:

- Training accuracy and loss plots.
- Predictions on test images, with each displayed alongside its predicted digit label.
