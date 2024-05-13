## Introduction

In this project we consider [the Kaggle nlp challenge](https://www.kaggle.com/competitions/nlp-getting-started) involving designing a disaster tweet predictor using Twitter (now X) data.
The project aims to create a program which builds and trains a DeBERTa model.
In addition to obtaining a model we aim to create an application using Flask which uses the model to make a prediction on some test provided by the user.

## Installation

In order to run this project please install the Python packages listed in requirements.txt.
In addition, please install the nltk stopwords by running the following command:

```bash
nltk.download("stopwords")
```

## Usage

In order to run the application with a trained model please include a folder called "model" inside the "tweet_app" directory which contains a state dictionary file called "model.pth".

In case you wish to train a model [using optimizer_model.py](https://github.com/acolgithub/tweets_classification_project/blob/main/create_model/optimize_model.py), please inlucde a folder called "data" inside the "create_trained_model" directory and add the training (train.csv) file from [the Kaggle nlp challenge page](https://www.kaggle.com/competitions/nlp-getting-started).
