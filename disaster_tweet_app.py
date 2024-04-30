from flask import Flask

app = Flask(__name__)
app.config['TESTING'] = True

@app.route("/")
def home():
    return 'This is the home page for the app. You can see the predictions from the model by appending /predictions'

@app.route('/predictions')
def hello_world():
    return "Testing Flask"
