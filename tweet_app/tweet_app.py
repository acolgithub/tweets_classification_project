from get_model.model import Make_model
from get_model.parameters import Params
from get_model.preprocessing import preprocess_text

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple
from transformers import DebertaForSequenceClassification


from flask import Flask, render_template, request, abort, Response

app = Flask(__name__)

# check if cuda is available else use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get parameters used for model
params = Params(device=device)

# get model to store saved model
model = Make_model(params)

@app.route("/")
def home():

    # try to read in model
    try:
        # get best saved model
        model.model.load_state_dict(torch.load("create_model/model/model.pth"), strict=False)
        model.model.eval()

    except FileNotFoundError:
        abort(500)

    return render_template("home.html")

@app.route("/", methods=["GET", "POST"])
def get_user_input():

    # if user enters data return prediction
    if request.method == "POST":

        # get user input
        user_input = request.form.get("input")

        # process text
        preprocessed_user_input = preprocess_text([user_input], params)

        # create logits
        logits = model(
            input_ids = preprocessed_user_input[:, :params.max_len],
            attention_masks = preprocessed_user_input[:, params.max_len:],
            target = None
        )
        
        # create prediction
        prediction = torch.argmax(logits, axis=1).flatten().item()

        # create response
        response = "Your text does " + ("not " * prediction) + "indicate a disaster."

        return render_template("home.html", user_text_input=user_input, response_output=response)
    
    # if retrieving data
    elif request.method == "GET":
        return render_template("home.html")
    
@app.errorhandler(500)
def read_error(error):
    error_message = """
                        500 Error:
                        Please create a folder called "model" inside "create_model" and
                        place a model state dictionary called "model.pth" inside.
                        """
    return error_message
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")
