# syntax=docker/dockerfile:1

# base image
FROM python:3.11-slim

# make and set working directory
RUN mkdir -p /tweet_app
WORKDIR /tweet_app

# copy and install requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# download NLTK files
RUN python -m nltk.downloader stopwords

# copy remaining files
COPY . .

# expose the application port
EXPOSE 5000

# start the application
CMD [ "python", "tweet_app.py" ]
