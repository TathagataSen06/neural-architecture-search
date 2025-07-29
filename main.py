# main.py
# This script demonstrates how to use the AutoKeras library to perform
# Neural Architecture Search (NAS) for an image classification task.
# We will use the popular MNIST dataset of handwritten digits.

# First, you need to install AutoKeras and TensorFlow.
# You can do this by running the following command in your terminal:
# pip install autokeras tensorflow

import autokeras as ak
import tensorflow as tf
import numpy as np

def run_nas_example():
    """
    A complete example of using AutoKeras for Neural Architecture Search.
    """
    print("--- Loading MNIST Dataset ---")
    # Load the MNIST dataset from Keras datasets
    # x_train: training images
    # y_train: training labels
    # x_test: testing images
    # y_test: testing labels
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print("-----------------------------\n")


    print("--- Initializing AutoKeras Image Classifier ---")
    # Initialize the ImageClassifier.
    # AutoKeras will automatically search for the best neural network architecture.
    # - max_trials: The maximum number of different Keras models to try.
    #   The search is finished when this number is reached. A higher number
    #   will likely find a better model, but will take longer.
    # - project_name: The name of the project. AutoKeras will save the search
    #   progress to a directory with this name.
    # - overwrite: If True, it will overwrite the existing project.
    clf = ak.ImageClassifier(
        max_trials=3,  # For a real-world problem, you'd use a higher number (e.g., 50 or 100)
        project_name='autokeras_mnist_example',
        overwrite=True
    )
    print("Classifier initialized. Starting the search process...")
    print("This may take a while depending on your hardware...")
    print("--------------------------------------------------\n")


    print("--- Starting Neural Architecture Search ---")
    # Start the search for the best model architecture.
    # The `fit` method will train and evaluate multiple architectures.
    # - x_train, y_train: The training data.
    # - validation_split: The portion of training data to use for validation.
    # - epochs: The number of epochs to train each model.
    clf.fit(
        x_train,
        y_train,
        epochs=2 # Using a small number of epochs for this demo
    )
    print("-------------------------------------------\n")


    print("--- Evaluating the Best Model Found ---")
    # Evaluate the best model found by AutoKeras on the test data.
    loss, accuracy = clf.evaluate(x_test, y_test)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")
    print(f"Loss on test data: {loss:.4f}")
    print("---------------------------------------\n")


    print("--- Exporting the Best Keras Model ---")
    # Get the best model found by the search.
    best_model = clf.export_model()

    # You can now use the best_model as a standard Keras model.
    # Print a summary of the discovered architecture.
    best_model.summary()
    print("----------------------------------------\n")


    print("--- Making Predictions with the Best Model ---")
    # Use the best model to make a prediction on a sample image.
    sample_image = x_test[0]
    # The model expects a batch of images, so we add a dimension.
    sample_image_batch = np.expand_dims(sample_image, axis=0)
    
    prediction = best_model.predict(sample_image_batch)
    predicted_class = np.argmax(prediction[0])
    actual_class = y_test[0]

    print(f"Predicted class for the first test image: {predicted_class}")
    print(f"Actual class for the first test image:   {actual_class}")
    print("---------------------------------------------\n")

if __name__ == "__main__":
    run_nas_example()
