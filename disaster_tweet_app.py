from flask import Flask, render_template

app = Flask(__name__)
app.config['TESTING'] = True

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predictions')
def hello_world():
    return "Hi."
