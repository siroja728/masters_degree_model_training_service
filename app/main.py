import uuid
from typing import Annotated, List, Literal

from fastapi import BackgroundTasks, FastAPI, Form, UploadFile
from pydantic import BaseModel

from app.model_predict import predict
from app.model_training import train_model_v2

ACTIVATION_FUNCTIONS = Literal[
    "elu",
    "exponential",
    "gelu",
    "hard_sigmoid",
    "linear",
    "mish",
    "relu",
    "selu",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "swish",
    "tanh",
]

LOSS_FUNCTIONS = Literal[
    "binary_crossentropy",
    "binary_focal_crossentropy",
    "categorical_crossentropy",
    "categorical_focal_crossentropy",
    "categorical_hinge",
    "cosine_similarity",
    "hinge",
    "huber",
    "kl_divergence",
    "kld",
    "kullback_leibler_divergence",
    "mae",
    "mape",
    "mse",
    "msle",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "mean_squared_logarithmic_error",
    "poisson",
    "squared_hinge",
]

OPTIMIZERS = Literal[
    "adadelta",
    "adafactor",
    "adagrad",
    "adam",
    "adamw",
    "adamax",
    "ftrl",
    "lion",
    "nadam",
    "rmsprop",
]

app = FastAPI()


class AppModel(BaseModel):
    service: str
    author: str
    contributors: str


@app.get("/")
def read_root() -> AppModel:
    return AppModel(
        service="Welcome to model training service. This is my work for masters degree diploma =)",
        author="Serhii Yemelianov",
        contributors="Vadym Nemchencko, Vadym Horban, Serhii Levchenko",
    )


class DictionariesResponse(BaseModel):
    activation_functions: List
    loss_functions: List
    optimizers: List


@app.get("/dictionaries", response_model=DictionariesResponse)
def get_dictionaries():
    return {
        "activation_functions": ACTIVATION_FUNCTIONS.__args__,
        "loss_functions": LOSS_FUNCTIONS.__args__,
        "optimizers": OPTIMIZERS.__args__,
    }


# Train model API endpoint (BEGIN)
class TrainModelResponse(BaseModel):
    filename: str
    file_size: int
    labels_field: str
    uuid: str


@app.post("/train_model", response_model=TrainModelResponse)
def train_model(
    background_tasks: BackgroundTasks,
    labels_field: Annotated[str, Form()],
    file: Annotated[UploadFile, Form()],
    activation: Annotated[ACTIVATION_FUNCTIONS, Form()] = "relu",
    additional_activation: Annotated[ACTIVATION_FUNCTIONS, Form()] = "softmax",
    loss: Annotated[LOSS_FUNCTIONS, Form()] = "categorical_crossentropy",
    optimizer: Annotated[OPTIMIZERS, Form()] = "adam",
    batch_size: Annotated[int, Form()] = 64,
    epochs: Annotated[int, Form()] = 5,
    test_size: Annotated[float, Form()] = 0.2,
    random_state: Annotated[int, Form()] = 40,
):
    model_uuid = uuid.uuid4()
    background_tasks.add_task(
        train_model_v2,
        config={
            "uuid": model_uuid,
            "activation": activation,
            "additional_activation": additional_activation,
            "loss": loss,
            "optimizer": optimizer,
            "batch_size": batch_size,
            "epochs": epochs,
            "test_size": test_size,
            "random_state": random_state,
        },
        label_fields=labels_field,
        file_path=file.file,
    )

    return {
        "filename": file.filename,
        "file_size": file.size,
        "labels_field": labels_field,
        "uuid": str(model_uuid),
    }


# Train model API endpoint (END)

# Predict API endpoint (BEGIN)


@app.post("/predict_model")
def model_predict(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, Form()],
    path_to_model: Annotated[str, Form()],
    path_to_weights: Annotated[str, Form()],
):
    prediction_uuid = uuid.uuid4()
    background_tasks.add_task(
        predict,
        file_path=file.file,
        model_path=path_to_model,
        model_weights_path=path_to_weights,
        prediction_uuid=prediction_uuid,
    )

    return {
        "model_path": path_to_model,
        "model_weights_path": path_to_weights,
        "prediction_uuid": str(prediction_uuid),
        "predicting": True,
    }


# Predict API endpoint (END)

# To run the server without docker: uvicorn app.main:app --reload
