import logging
import unittest

from fastapi.testclient import TestClient

from main import app

logging.basicConfig(level=logging.INFO)


class AppUnitTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_read_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "service": "Welcome to model training service. This is my work for masters degree diploma =)",
                "author": "Serhii Yemelianov",
                "contributors": "Vadym Nemchencko, Vadym Horban, Serhii Levchenko",
            },
        )

    def test_get_parameters(self):
        response = self.client.get("/parameters")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "activation_functions": ["linear", "relu", "sigmoid"],
                "loss_functions": ["mean_squared_error", "mean_absolute_error"],
                "optimizers": ["adam", "rmsprop", "sgd", "adagrad"],
            },
        )

    def test_enqueue_job(self):
        data = {
            "processId": "training_test",
            "processStructureId": "training_test",
            "configuration": {
                "optimizer": "adam",
                "loss": "mean_squared_error",
                "epochs": 10,
                "batch_size": 10,
                "activation": "linear",
                "layers": 20,
            },
            "training": {
                "inputs": [
                    {"name": "name1", "data": [1, 2, 3, 4]},
                    {"name": "name2", "data": [1, 2, 3, 4]},
                    {"name": "name3", "data": [1, 2, 3, 4]},
                    {"name": "name4", "data": [1, 2, 3, 4]},
                ],
                "outputs": [{"name": "name1", "data": [1, 2, 3, 4]}],
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
        }
        response = self.client.post("/enqueue_job", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Job enqueued successfully."})


if __name__ == "__main__":
    unittest.main()
