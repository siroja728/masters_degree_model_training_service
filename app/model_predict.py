import datetime
import os

import numpy as np
import pandas as pd

from app.model_training import load_model
from app.processing_data import read_and_prepare_data


def predict(
    prediction_uuid, file_path="data/KDDTest.csv", model_path="", model_weights_path=""
):
    try:
        # label_fields is hardcoded because id don't know for now why we need it when predicting
        evaluate_data = read_and_prepare_data(file_path, label_fields="attack")

        test_data = evaluate_data["training_data"]

        model = load_model(model_path, model_weights_path)
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

        model_id = model_path.split("/")[3]
        dir_path = (
            "predictions/"
            + datetime.date.today().strftime("%Y")
            + "/"
            + datetime.date.today().strftime("%m")
            + "/"
            + model_id
            + "/"
        )

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        predictions_file_path = dir_path + str(prediction_uuid) + ".csv"

        df.to_csv(predictions_file_path)

        result = {
            "predictions_file_path": predictions_file_path,
            "prediction_uuid": prediction_uuid,
            "model_path": model_path,
            "model_weights_path": model_weights_path,
            "predicting": False,
        }

        print("Prediction: Ok. Predictions saved!", result)
    except:
        print("Prediction: Oops something went wrong!")
