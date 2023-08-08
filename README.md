# masters_degree_model_training_service

## Project description
This is a service that allow user to train model and make a predictions using tensorflow.

## Run project
- ### Docker way:
    - Install Docker
    - Clone the project
    - In project root from the console run the command: `docker-compose up -d`
    - That's everything and now you can make api requests to `http://localhost:8000/`

- ###  Without docker:
    - Install Python
    - In root folder run `pip install -r requirements.txt`
    - After successful install of all requirements run `uvicorn app.main:app --reload`

## Stop project
If you started Compose with `docker compose up -d`, stop your services once you’ve finished with them:
`docker compose stop`

You can bring everything `down`, removing the containers entirely, with the down command. Pass `--volumes` to also remove the data volume used by the Redis container:
`docker compose down --volumes`

## Docs with all endpoints
If you go to `http://localhost:8000/docs` you will see all endpoints that available to use.