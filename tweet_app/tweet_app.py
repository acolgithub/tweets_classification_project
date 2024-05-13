from model_class.model import Make_model
from model_class.parameters import Params
from model_class.preprocessing import preprocess_text

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple
from transformers import DebertaForSequenceClassification


from flask import Flask, render_template, request, abort

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/", methods=["GET", "POST"])
def get_user_input():

    # check if cuda is available else use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get parameters used for model
    params = Params(device=device)

    # get untrained model
    model = Make_model(params)
    model.model.to(device)

    # if user enters data return prediction
    if request.method == "POST":

        # get user input
        user_input = request.form.get("input")

        # process text
        preprocessed_user_input = preprocess_text([user_input], params)

        # if user wants trained prediction replace untrained model with trained one
        if request.form["button"] == "trained_prediction":
            
            # try to read in model
            try:
                # get best saved model
                model.load_state_dict(torch.load("model/model.pth"), strict=False)

            except FileNotFoundError:
               abort(500)

        # create response
        response = ""

        if len(user_input) > 0:

                response = create_response(model, params, preprocessed_user_input)

        return render_template("home.html", user_text_input=user_input, response_output=response)
    
    # if retrieving data
    elif request.method == "GET":
        return render_template("home.html")
    
@app.errorhandler(500)
def read_error(error):
    error_message = """
                        500 Error:
                        Please create a folder called "model" inside the current directory and
                        place a model state dictionary called "model.pth" inside.
                    """
    return error_message



# define function to obtain application reponse to user input
def create_response(
        model: Make_model,
        params: Params,
        preprocessed_user_input: str
) -> str:
    """
    Function to create a response to user input.

    Keyword arguments:
    model -- model to use for prediction
    params -- parameters used to create the model
    preprocessed_user_input -- preprocessed text from user input
    """

    # set evaluation mode
    model.model.eval()

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
        
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0")