import asyncio
import threading

import uvicorn
from bullmq import Queue, Worker
from fastapi import FastAPI

from model_training import train_model

app = FastAPI()


async def process(job, job_token):
    return train_model(job.data, job_token)


queue = Queue("jobs")
worker = Worker("jobs", process, {"connection": {"host": "localhost", "port": 6379}})


def run_worker():
    asyncio.run(worker.run())


def start_background_tasks():
    thread = threading.Thread(target=run_worker)
    thread.start()


@app.post("/enqueue_job")
async def enqueue_job(data: dict):
    await queue.add("job", data)

    return {"message": "Job enqueued successfully."}


if __name__ == "__main__":
    # Start the worker in the background
    start_background_tasks()

    # Start the FastAPI server
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        log_level="info",
        reload=True,
        reload_dirs=["."],
        workers=1,
    )
