from flask import Flask, jsonify, request

from model import split_data_and_train_models
from contrller import resize

HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

app = Flask(__name__)
#split_data_and_train_models()

def flask_app():


    @app.route('/', methods=['GET'])
    def server_is_up():
        return 'server is up'

    @app.route('/resize', methods=['POST'])
    def start():
        body = request.json

        print(body)
        pred = resize(body)
        print(pred)
        return jsonify(str(pred))


flask_app()
app.run(debug=True, port=4444)