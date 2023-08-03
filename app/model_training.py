import datetime
import os

import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from tensorflow import keras

from app.processing_data import read_and_prepare_data


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


def save_model(model, uuid):
    # serialize model to JSON
    model_json = model.to_json()

    # make unique directory for model saving
    dir_path = (
        "models/"
        + datetime.date.today().strftime("%Y")
        + "/"
        + datetime.date.today().strftime("%m")
        + "/"
        + str(uuid)
    )

    os.makedirs(dir_path)

    model_json_path = dir_path + "/model.json"
    model_weights_path = dir_path + "/model.h5"

    # save model as a json file
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_weights_path)

    return {
        "model_json_path": model_json_path,
        "model_weights_path": model_weights_path,
    }


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
    data = read_and_prepare_data(file_path, label_fields)
    features = data["training_data"]
    labels = data["labels"]
    max_label = max(labels) + 1

    model = create_model(max_label)
    model.fit(features, labels, epochs=5)

    save_model(model)


def train_model_v2(
    config,
    label_fields="attack",
    file_path="data/KDDTrain.csv",
):
    # Read the dataset from a CSV file
    data = read_and_prepare_data(file_path, label_fields)
    X = np.array(data["training_data"])
    y = np.array(data["labels"])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"]
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
        tf.keras.layers.Dense(
            num_classes,
            activation=config["activation"],
            input_shape=(X_train.shape[1],),
        )
    )
    model.add(tf.keras.layers.Dense(num_classes, activation=config["activation"]))
    model.add(tf.keras.layers.Dense(num_classes, activation=config["activation"]))
    model.add(tf.keras.layers.Dense(num_classes, activation=config["activation"]))
    model.add(tf.keras.layers.Dense(num_classes, activation=config["activation"]))
    model.add(
        tf.keras.layers.Dense(num_classes, activation=config["additional_activation"])
    )

    # Compile the model
    model.compile(
        loss=config["loss"], optimizer=config["optimizer"], metrics=["accuracy"]
    )

    # Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=(X_test, y_test),
    )

    # save model to json
    saved_model = save_model(model, config["uuid"])
    saved_model.update({"uuid": str(config["uuid"])})

    print(saved_model)
