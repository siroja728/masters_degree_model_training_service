import os

import tensorflow as tf


def train_model(data, job_token):
    try:
        print(f"Incoming job token: {job_token}")

        # network parameters
        epochs = data["configuration"]["epochs"]
        optimizer = data["configuration"]["optimizer"]
        activation = data["configuration"]["activation"]
        loss = data["configuration"]["loss"]
        layers = data["configuration"]["layers"]
        batch_size = data["configuration"]["batch_size"]
        model_id = data["processStructureId"]

        # directories and model path
        models_dir = "models"
        model_path = f"{models_dir}/{model_id}.h5"

        # creating model directory if not exists
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir)

        # training data
        train_x = tf.convert_to_tensor(
            [input_data["data"] for input_data in data["training"]["inputs"]],
            dtype=tf.float32,
        )
        train_y_list = []
        for output in data["training"]["outputs"]:
            train_y_list.append(
                tf.convert_to_tensor(
                    output["data"],
                    dtype=tf.float32,
                )
            )

        # evaluating data
        test_x = tf.convert_to_tensor(
            [input_data["data"] for input_data in data["checking"]["inputs"]],
            dtype=tf.float32,
        )
        test_y_list = []
        for output in data["checking"]["outputs"]:
            test_y_list.append(
                tf.convert_to_tensor(
                    output["data"],
                    dtype=tf.float32,
                )
            )

        # init model
        model = tf.keras.models.Sequential()

        # add layers
        for _ in range(layers):
            model.add(
                tf.keras.layers.Dense(
                    4, input_shape=(train_x.shape[1],), activation=activation
                )
            )

        # for approximation purposes we always need a layer with linear activation function
        model.add(tf.keras.layers.Dense(4, activation="linear"))

        # model compilation
        model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])

        # model training
        model.fit(
            train_x, train_y_list, epochs=epochs, batch_size=batch_size, verbose=1
        )

        # evaluating model on test data
        loss, accuracy = model.evaluate(test_x, test_y_list)

        # saving model
        model.save(filepath=model_path)

        print(f"Loss on test data: {loss}")
        print(f"Accuracy on test data: {accuracy}")
        print(f"Add job to responses queue that indicate that job is finished.")
    except:
        print("An exception occurred")


def model_predict(model, prediction_data):
    return True
