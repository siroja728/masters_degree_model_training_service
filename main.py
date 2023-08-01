import uuid
from typing import Annotated, List, Literal

from fastapi import BackgroundTasks, FastAPI, Form, UploadFile
from pydantic import BaseModel

import model_training

activation_functions = Literal[
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

loss_functions = Literal[
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

optimizers = Literal[
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
        "activation_functions": activation_functions.__args__,
        "loss_functions": loss_functions.__args__,
        "optimizers": optimizers.__args__,
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
    activation: Annotated[activation_functions, Form()] = "relu",
    additional_activation: Annotated[activation_functions, Form()] = "softmax",
    loss: Annotated[loss_functions, Form()] = "categorical_crossentropy",
    optimizer: Annotated[optimizers, Form()] = "adam",
    batch_size: Annotated[int, Form()] = 64,
    epochs: Annotated[int, Form()] = 5,
    test_size: Annotated[float, Form()] = 0.2,
    random_state: Annotated[int, Form()] = 40,
):
    model_uuid = uuid.uuid4()
    background_tasks.add_task(
        model_training.train_model_v2,
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

# To run the server: uvicorn main:app --reload
