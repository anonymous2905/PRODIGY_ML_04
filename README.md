# PRODIGY_ML_04

# Hand Gesture Recognition using CNN

This repository contains code for a hand gesture recognition system using Convolutional Neural Networks (CNNs). The dataset used is the Leap GestRecog dataset, which consists of images of different hand gestures.

## Dataset

The dataset can be downloaded from Kaggle: [LeapGestRecog](https://www.kaggle.com/gti-upm/leapgestrecog)

The dataset contains 10 classes of hand gestures:

1. 01_palm
2. 02_l
3. 03_fist
4. 04_fist_moved
5. 05_thumb
6. 06_index
7. 07_ok
8. 08_palm_moved
9. 09_c
10. 10_down

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Kaggle API

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/hand-gesture-recognition.git
    cd hand-gesture-recognition
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset:**

    Ensure you have your Kaggle API key (`kaggle.json`) set up. If not, follow the instructions [here](https://www.kaggle.com/docs/api).

    ```python
    !mkdir ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

    !kaggle datasets download -d gti-upm/leapgestrecog
    !unzip leapgestrecog.zip
    ```

## Training the Model

The Jupyter Notebook `hand_gesture_recognition.ipynb` contains the complete code for loading the data, preprocessing, building the CNN model, and training it.

1. **Open the Jupyter Notebook:**

    ```bash
    jupyter notebook hand_gesture_recognition.ipynb
    ```

2. **Run the cells to execute the code step-by-step:**

    - Load and preprocess the data
    - Build the CNN model
    - Train the model
    - Evaluate the model
    - Plot the training history

## Model Architecture

The model consists of:

- Two convolutional layers with 16 and 32 filters respectively, each followed by Batch Normalization and ReLU activation.
- A max-pooling layer with a pool size of (2, 2) and a dropout layer with a rate of 0.25.
- A flatten layer to convert the 2D matrix to a vector.
- A fully connected (Dense) layer with 128 units, followed by Batch Normalization and ReLU activation.
- An output layer with 10 units and softmax activation for classification.

## Results

After training for 7 epochs, the model achieved a high accuracy on the validation set. The training and validation accuracy can be visualized using the plot generated at the end of the notebook.


## Acknowledgments

- The dataset used in this project is provided by the [LeapGestRecog dataset](https://www.kaggle.com/gti-upm/leapgestrecog) on Kaggle.
- This project is inspired by the need for efficient hand gesture recognition systems in various applications.

