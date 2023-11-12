import asyncio

import tensorflow as tf
from bullmq import Worker

from model_training import train_model


async def process(job, job_token):
    # job.data will include the data added to the queue
    return train_model(job.data, job_token)


async def main():
    # Feel free to remove the connection parameter, if your redis runs on localhost
    worker = Worker(
        "jobs", process, {"connection": {"host": "localhost", "port": 6379}}
    )

    while True:  # Add some breaking conditions here
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
