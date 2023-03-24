import pickle
import numpy as np
from flask import Flask, request
from flask_cors import CORS


model = None
app = Flask(__name__)
CORS(app)


def load_model():
    global model
    with open('./models/heart_disease.pkl', 'rb') as f:
        model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():

    if request.method == 'POST':
        data = request.get_json()
        data = np.array(data)[np.newaxis, :]
        prediction = model.predict(data)
        return str(prediction[0])

    return 'Invalid request'


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=80)
