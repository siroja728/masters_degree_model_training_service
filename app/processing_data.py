import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_and_prepare_data(file_path="", label_fields=""):
    result = dict()

    if file_path:
        # read data form csv file
        training_data = pd.read_csv(file_path)

        # deleting rows that has null values
        training_data = training_data.dropna()

        # detecting columns that contains only numbers
        num_columns = training_data.select_dtypes(include=np.number).columns

        # detecting columns with non number values
        cat_columns = []
        for col in training_data.columns:
            if col not in num_columns:
                cat_columns.append(col)

        # convert non number columns to numeric
        for cc in cat_columns:
            training_data[cc] = pd.Categorical(training_data[cc])
            training_data[cc] = training_data[cc].cat.codes

        result["training_data"] = training_data
        result["columns"] = training_data.columns
        result["labels"] = training_data.copy().pop(label_fields)

        return result
    else:
        print("Please specify filePath")
        return result
