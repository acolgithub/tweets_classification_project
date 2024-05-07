from create_model.model import Make_model
from create_model.parameters import Params
from create_model.preprocessing import preprocess_text

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple
from transformers import DebertaForSequenceClassification


from flask import Flask, render_template, request

app = Flask(__name__)
app.config['TESTING'] = True

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/", methods=["GET", "POST"])
def get_user_input():

    # if user enters data return prediction
    if request.method == "POST":

        # get user input
        user_input = request.form.get("input")

        # get parameters used for model
        params = Params()

        # process text
        preprocessed_user_input = preprocess_text([user_input], params)

        # get best saved model
        model = Make_model(params)
        model.load_state_dict(torch.load, PATH)
        model.eval()

        # create logits
        logits = model(
            input_ids = preprocessed_user_input[:, :params.max_len],
            attention_mask = preprocessed_user_input[:, params.max_len:])
        
        # create prediction
        prediction = torch.argmax(logits, axis=1).flatten()

        return render_template("home.html", user_text_input=user_input, response_output=prediction)
    
    # if retrieving data
    elif request.method == "GET":
        return render_template("home.html")
