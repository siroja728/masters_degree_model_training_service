import tensorflow as tf


def train_model(data, job_token):
    epochs = data["configuration"]["epochs"]
    optimizer = data["configuration"]["optimizer"]
    activation = data["configuration"]["activation"]
    loss = data["configuration"]["loss"]
    layers = data["configuration"]["layers"]

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

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    model.fit(train_x, train_y, epochs=epochs, batch_size=1, verbose=1)

    loss, accuracy = model.evaluate(test_x, test_y)

    print(f"Loss on test data: {loss}")
    print(f"Accuracy on test data: {accuracy}")
