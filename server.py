import os
from flask import Flask, jsonify, request

import json
from model import resize


HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

def flask_app():
    app = Flask(__name__)


    @app.route('/', methods=['GET'])
    def server_is_up():
        return 'server is up'

    @app.route('/resize', methods=['POST'])
    def start():
        body = request.json

        print(body)
        print('body')
        pred = resize(body)
        print(pred)
        return jsonify(str(pred))
    return app

if __name__ == '__main__':
    app = flask_app()
    app.run(debug=True, port=4444)