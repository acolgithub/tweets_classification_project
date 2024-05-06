from flask import Flask, render_template, request
from .fns import hello

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

        # create response
        response = user_input.upper()

        return render_template("home.html", user_text_input=user_input, response_output=response)
    
    # if retrieving data
    elif request.method == "GET":
        return render_template("home.html")
