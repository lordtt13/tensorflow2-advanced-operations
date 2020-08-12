import numpy as np

from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.config["DEBUG"] = True

transformer = SentenceTransformer('distilbert-base-nli-mean-tokens').to('cuda')

@app.route("/emotions/get_embeddings", methods = ["GET","POST"])
def embeddings():
    result = {}
    result['reqs'] = request.args['sentence']
    result['id'] = request.args['id']
    sentence = transformer.encode([request.args['sentence']])
    result['embeddings'] = sentence[0]
    return jsonify(result)


if __name__== "__main__":
	app.run(host = '0.0.0.0')