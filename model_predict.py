import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

import model_training
import processing_data


def predict():
    evaluate_data = processing_data.read_and_prepare_data("data/KDDTest.csv", "attack")

    test_data = evaluate_data["training_data"]

    model = model_training.load_model("models/8a5b4579-322a-4f9c-855e-6638b1be0b2b")
    predictions = model.predict(test_data)

    # Clip the prediction values to a minimum threshold
    min_threshold = 1e-10
    predictions_clipped = np.clip(predictions, min_threshold, 1.0)

    # Normalize the clipped predictions within the range of 0 to 1
    min_value = np.min(predictions_clipped)
    max_value = np.max(predictions_clipped)
    scaled_predictions = (predictions_clipped - min_value) / (max_value - min_value)

    # Round the scaled predictions to a desired number of decimal places
    decimal_places = 2
    rounded_predictions = np.round(scaled_predictions, decimal_places)

    # Save predictions to csv
    df = pd.DataFrame(rounded_predictions)
    df.to_csv("predictions/8a5b4579-322a-4f9c-855e-6638b1be0b2b.csv")


predict()
