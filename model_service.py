import os

import tensorflow as tf


def train_model(data, job_token):
    epochs = data["configuration"]["epochs"]
    optimizer = data["configuration"]["optimizer"]
    activation = data["configuration"]["activation"]
    loss = data["configuration"]["loss"]
    layers = data["configuration"]["layers"]
    batch_size = data["configuration"]["batch_size"]
    model_id = data["processStructureId"]

    models_dir = "models"
    model_path = f"{models_dir}/{model_id}.h5"

    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)

    train_x = tf.convert_to_tensor(
        [input_data["data"] for input_data in data["training"]["inputs"]],
        dtype=tf.float32,
    )

    train_y = tf.convert_to_tensor(
        data["training"]["outputs"][0]["data"], dtype=tf.float32
    )

    test_x = tf.convert_to_tensor(
        [input_data["data"] for input_data in data["checking"]["inputs"]],
        dtype=tf.float32,
    )

    test_y = tf.convert_to_tensor(
        data["checking"]["outputs"][0]["data"], dtype=tf.float32
    )

    model = tf.keras.models.Sequential()

    for _ in range(layers):
        model.add(
            tf.keras.layers.Dense(
                4, input_shape=(train_x.shape[1],), activation=activation
            )
        )
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    loss, accuracy = model.evaluate(test_x, test_y)

    model.save(filepath=model_path)

    print(f"Loss on test data: {loss}")
    print(f"Accuracy on test data: {accuracy}")


def model_predict(model, prediction_data):
    return True
