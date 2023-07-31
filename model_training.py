import os
import uuid

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import layers
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from tensorflow import keras

import processing_data


def create_model(max_label):
    model = keras.Sequential(
        [
            layers.Dense(max_label, activation="relu"),
            layers.Dense(max_label, activation="relu"),
            layers.Dense(max_label, activation="relu"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()

    # make unique directory for model saving
    dir_path = "models/" + str(uuid.uuid4())
    os.mkdir(dir_path)

    # save model as a json file
    with open(dir_path + "/model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(dir_path + "/model.h5")
    print("Saved model to disk")


def load_model(path_to_model):
    # load json and create model
    json_file = open(path_to_model + "/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_to_model + "/model.h5")
    print("Loaded model from disk")
    return loaded_model


def train_model(
    label_fields="attack",
    file_path="data/KDDTrain.csv",
):
    data = processing_data.read_and_prepare_data("data/KDDTrain.csv", label_fields)
    features = data["training_data"]
    labels = data["labels"]
    max_label = max(labels) + 1

    model = create_model(max_label)
    model.fit(features, labels, epochs=5)

    save_model(model)


def train_model_v2(
    label_fields="attack",
    file_path="data/KDDTrain.csv",
):
    # Read the dataset from a CSV file
    data = processing_data.read_and_prepare_data(file_path, label_fields)
    X = np.array(data["training_data"])
    y = np.array(data["labels"])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test = (X_test - np.mean(X_train)) / np.std(X_train)

    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y))  # Number of unique classes in the labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Build the neural network model
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],))
    )
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Train the model
    batch_size = 64
    epochs = 5
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    # save model to json
    save_model(model)

    return {"loss": loss, "accuracy": accuracy}
