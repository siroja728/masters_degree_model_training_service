import datetime
import os

import numpy as np
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from app.processing_data import read_and_prepare_data


def create_model(max_label):
    model = Sequential(
        [
            Dense(max_label, activation="relu"),
            Dense(max_label, activation="relu"),
            Dense(max_label, activation="relu"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()],
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


def load_model(path_to_model, path_to_weight):
    if path_to_model:
        # load json and create model
        json_file = open(path_to_model, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

    if path_to_weight:
        # load weights into new model
        loaded_model.load_weights(path_to_weight)

    if path_to_model and path_to_weight:
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
    try:
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
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        # Build the neural network model
        model = Sequential()
        model.add(
            Dense(
                num_classes,
                activation=config["activation"],
                input_shape=(X_train.shape[1],),
            )
        )
        model.add(Dense(num_classes, activation=config["activation"]))
        model.add(Dense(num_classes, activation=config["activation"]))
        model.add(Dense(num_classes, activation=config["activation"]))
        model.add(Dense(num_classes, activation=config["activation"]))
        model.add(Dense(num_classes, activation=config["additional_activation"]))

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

        print(
            "Successfully train model. Send data to main service via request or web sockets.",
            saved_model,
        )
    except:
        print(
            "Error train model. Send data to main service via request or web sockets.",
            {"uuid": str(config["uuid"])},
        )
