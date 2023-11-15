import asyncio
import threading
from typing import List, Literal

import uvicorn
from bullmq import Queue, Worker
from fastapi import FastAPI
from pydantic import BaseModel

from model_service import train_model

LOSS_FUNCTIONS = Literal["mean_squared_error", "mean_absolute_error"]
ACTIVATION_FUNCTIONS = Literal["linear", "relu", "sigmoid"]
OPTIMIZERS = Literal["adam", "rmsprop", "sgd", "adagrad"]


class AppModel(BaseModel):
    service: str
    author: str
    contributors: str


class ParametersResponse(BaseModel):
    activation_functions: List
    loss_functions: List
    optimizers: List


async def process(job, job_token):
    return train_model(job.data, job_token)


app = FastAPI()
queue = Queue("jobs")
worker = Worker("jobs", process, {"connection": {"host": "localhost", "port": 6379}})


def run_worker():
    asyncio.run(worker.run())


def start_background_tasks():
    thread = threading.Thread(target=run_worker)
    thread.start()


@app.get("/")
def read_root() -> AppModel:
    return AppModel(
        service="Welcome to model training service. This is my work for masters degree diploma =)",
        author="Serhii Yemelianov",
        contributors="Vadym Nemchencko, Vadym Horban, Serhii Levchenko",
    )


@app.get("/parameters", response_model=ParametersResponse)
def get_parameters():
    return {
        "activation_functions": ACTIVATION_FUNCTIONS.__args__,
        "loss_functions": LOSS_FUNCTIONS.__args__,
        "optimizers": OPTIMIZERS.__args__,
    }


@app.post("/enqueue_job")
async def enqueue_job(data: dict):
    await queue.add("job", data)

    return {"message": "Job enqueued successfully."}


if __name__ == "__main__":
    # Start the worker in the background
    start_background_tasks()
