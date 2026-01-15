from typing import Optional
import numpy as np

import Network
import Layers
import Layers as l
from utils import train_val_split

import keras.layers as k
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn
import matplotlib.pyplot as plt
import pandas as pd

# Creates graphs to compare this nn to tensorflow models
def main():
    # Data preparation
    X_train, y_train, X_test, y_test, input_size = get_data()

    # get validation set from training data
    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_split=0.1) 

    # Neural Network setup
    # k.Dense would correspond to keras Dense
    # l.dense would correspond to the Dense in layers.py 
    tensor_model = Sequential(layers= (k.Input(shape= (input_size,)), 
                                    k.Dense(64, activation= 'relu'),
                                    k.Dense(10, activation= 'softmax')))
    
    model = Network.Network(layers= (l.Input(input_size),
                                  l.Dense(64, activation= 'relu'),
                                  l.Dense(10, activation= 'softmax')))
    
    # Compile both Models
    tensor_model.compile(optimizer= 'adam', loss= 'categorical_crossentropy')
    model.compile(optimizer= 'adam', loss_func= 'categorical_crossentropy')

    # Train Networks using Training set and Validation set
    tensor_loss = tensor_model.fit(X_train, y_train, batch_size= 32, epochs= 8, validation_data = (X_val, y_val), shuffle= False)
    model_loss = model.fit(X_train, y_train, batch_size= 32, epochs= 8, validation_set= (X_val, y_val))

    # Get the predicted outputs for both models on the test data
    tensor_y_pred = tensor_model.predict(X_test)
    model_y_pred = model.predict(X_test)

    # Get Accuracy and save into accuracy.txt

    tensor_accuracy = accuracy_score(y_test.argmax(1), tensor_y_pred.argmax(1))
    model_accuracy = accuracy_score(y_test.argmax(1), model_y_pred.argmax(1))

    filepath = "figures\\accuracy.txt"

    try:
        with open(filepath, 'w') as file:
            file.write(f"Tensorflow Accuracy: {tensor_accuracy * 100}%\nNeural Network from Scratch Accuracy: {model_accuracy * 100}%")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


    print(f"Tensorflow model Accuracy: {tensor_accuracy}")
    print(f"NN from Scratch Accuracy: {model_accuracy}")

    # Generate Figures

    # Confusion Matrix
    tensor_cfm = confusion_matrix(y_test.argmax(1), tensor_y_pred.argmax(1))
    model_cfm = confusion_matrix(y_test.argmax(1), model_y_pred.argmax(1))

    generate_cfm_graph(tensor_cfm, title= "Tensor Flow Model Confusion Matrix", x_axis_text= "Predicted", y_axis_text= "Actual", filepath= "figures\\Tensor_Confusion_Matrix.png")
    generate_cfm_graph(model_cfm, title= "Neural Network from Scratch Confusion Matrix", x_axis_text= "Predicted", y_axis_text= "Actual", filepath= "figures\\Model_Confusion_Matrix.png")

    # Loss vs Epochs Graphs
    tensor_loss = tensor_loss.history

    # Tensor loss and Tensor val loss
    generate_loss_plot({"Training Loss": model_loss["loss"], "Validation Loss" : model_loss["val_loss"]}, title = "Neural Network from Scratch Loss vs Epochs", x_axis_text= "Epochs", y_axis_text= "Loss", filepath= "figures\\Model_Loss_Graph.png")
    generate_loss_plot({"Training Loss": tensor_loss["loss"], "Validation Loss" : tensor_loss["val_loss"]}, title = "Tensor Flow Model Loss vs Epochs", x_axis_text= "Epochs", y_axis_text= "Loss", filepath= "figures\\Tensor_Loss_Graph.png")
    
    # Comparing both training loss to each other and validation loss

    # Training Loss vs Training Loss
    training_loss_dict = {"Tensor Flow Model Training Loss": tensor_loss["loss"], "Neural Network From Scratch Training Loss": model_loss["loss"]}
    generate_loss_plot(training_loss_dict, title = "Training Loss vs Epochs Graph", x_axis_text= "Epochs", y_axis_text= "Loss", filepath= "figures\\Training_Loss_Graph.png")

    # Validiation loss vs Validation Loss
    validation_loss_dict = {"Tensor Flow Model Validation Loss" : tensor_loss["val_loss"], "Neural Network From Scratch Validation Loss": model_loss["val_loss"]}
    generate_loss_plot(validation_loss_dict, title = "Training Loss vs Epochs Graph", x_axis_text= "Epochs", y_axis_text= "Loss", filepath= "figures\\Validation_Loss_Graph.png")


# Function to get and preprocess data
def get_data():
    # Getting dataset
    (train_img, train_label), (test_img, test_label) = mnist.load_data()

    # convert pixel values from 0-255 to 0-1 to normalize data
    train_img = train_img / 255
    test_img = test_img / 255

    # convert label to a vector where the 1 is at the index of the label
    y_train = to_categorical(train_label, 10)
    y_test = to_categorical(test_label, 10)

    # convert the pixel 2d grid into a 1d vector for training
    train_num, img_rows, img_cols = train_img.shape
    test_num, _, _ = test_img.shape

    X_train = train_img.reshape(train_num, img_rows * img_cols)
    X_test = test_img.reshape(test_num, img_rows * img_cols)

    return (X_train, y_train, X_test, y_test, img_rows * img_cols)

def generate_cfm_graph(cfm, title: Optional[str] = None, x_axis_text: Optional[str] = None, y_axis_text = None, filepath = None):
    plt.clf()
    map = seaborn.heatmap(cfm, annot = True, fmt= "d")

    if title is not None:
        map.set_title(title)
    if x_axis_text is not None:
        map.set_xlabel(x_axis_text)
    if y_axis_text is not None:
        map.set_ylabel(y_axis_text)

    if filepath is not None:
        plt.savefig(filepath)

def generate_loss_plot(history: dict, title: Optional[str] = None, x_axis_text: Optional[str] = None, y_axis_text = None, filepath = None):
    plt.clf()

    map = seaborn.lineplot(data= history)

    if title is not None:
        map.set_title(title)
    if x_axis_text is not None:
        map.set_xlabel(x_axis_text)
    if y_axis_text is not None:
        map.set_ylabel(y_axis_text)

    if filepath is not None:
        plt.savefig(filepath)

if __name__ == "__main__":
    main()