# syntax=docker/dockerfile:1

FROM python:3.11-slim

RUN mkdir -p /tweets_classification_project
WORKDIR /tweets_classification_project

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "tweet_app.py" ]
