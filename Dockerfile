FROM python:3.9

WORKDIR /model_service

COPY ./requirements.txt /model_service/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /model_service/requirements.txt

COPY ./app /model_service/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]