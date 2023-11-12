import asyncio

from bullmq import Queue

queue = Queue("jobs")

# Possible values for approximation function
LOSS_FUNCTIONS = ["mean_squared_error", "mean_absolute_error"]
ACTIVATION_FUNCTIONS = ["linear", "relu", "tanh", "sigmoid"]
OPTIMIZERS = ["adam", "rmsprop", "sgd", "adagrad"]


async def add():
    await queue.add(
        "myJob",
        {
            "configuration": {
                "optimizer": "adagrad",
                "loss": "mean_absolute_error",
                "epochs": 50,
                "batch_size": 500,
                "activation": "tanh",
                "layers": 20,
            },
            "training": {
                "inputs": [
                    {"name": "name1", "data": [1, 2, 3, 4]},
                    {"name": "name2", "data": [1, 2, 3, 4]},
                    {"name": "name3", "data": [1, 2, 3, 4]},
                    {"name": "name4", "data": [1, 2, 3, 4]},
                ],
                "outputs": [
                    {"name": "name1", "data": [1, 2, 3, 4]},
                ],
            },
            "checking": {
                "inputs": [
                    {"name": "nameX1", "data": [1, 2, 3, 4]},
                    {"name": "nameX2", "data": [1, 2, 3, 4]},
                    {"name": "nameX3", "data": [1, 2, 3, 4]},
                    {"name": "nameX4", "data": [1, 2, 3, 4]},
                ],
                "outputs": [{"name": "nameY", "data": [1, 2, 3, 4]}],
            },
        },
    )
    await queue.close()


if __name__ == "__main__":
    asyncio.run(add())
