FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN curl -sSL https://get.docker.com/ | sh

COPY app.py app.py

EXPOSE 5000


CMD ["python", "app.py"]
