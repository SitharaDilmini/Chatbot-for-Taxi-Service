# importing the necessary libraries and modules
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from chatbot import start_chat

# initializing the flask app
app = Flask(__name__, static_folder='./frontend/chatbot-frontend/src/app')
CORS(app)

# defining the routes
@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'app.component.html')

# defining the predict route
@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    check = start_chat(text)

    res = {"answer": check}
    res_msg = jsonify(res)
    return res_msg

# running the app
if __name__ == "__main__" :
    app.run(debug = True)  