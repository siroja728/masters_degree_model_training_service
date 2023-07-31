from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, Form, UploadFile
from pydantic import BaseModel

import model_training

app = FastAPI()


# to run a server go to project root and run: uvicorn main:app --reload


class AppModel(BaseModel):
    service: str
    author: str
    contributors: str


@app.get("/")
async def read_root() -> AppModel:
    return AppModel(
        service="Welcome to model training service. This is my work for masters degree diploma =)",
        author="Serhii Yemelianov",
        contributors="Vadym Nemchencko, Vadym Horban, Serhii Levchenko",
    )


class TrainModel(BaseModel):
    labels_field: str = Form(...)
    file: UploadFile = Form(...)

    @classmethod
    def as_form(cls, labels_field: str = Form(...), file: UploadFile = Form(...)):
        return cls(labels_field=labels_field, file=file)


class TrainModelResponse(BaseModel):
    filename: str
    file_size: int
    labels_field: str
    uuid: str
    is_model_training: bool


@app.post("/train_model")
async def train_model(
    background_tasks: BackgroundTasks,
    form_data: TrainModel = Depends(TrainModel.as_form),
) -> TrainModelResponse:
    background_tasks.add_task(
        model_training.train_model_v2,
        label_fields=form_data.labels_field,
        file_path=form_data.file.file,
    )

    return TrainModelResponse(
        filename=form_data.file.filename,
        file_size=form_data.file.size,
        labels_field=form_data.labels_field,
        # uuid="qwerty123test",
        # is_model_training=True,
    )
