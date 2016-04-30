### App imports
from flask import Flask, request
from flask import render_template
import cPickle as pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Model imports

import io
import skimage.transform
import urllib
import lasagne
import pickle
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from appcode.model import DeployModel


app = Flask(__name__)


model = DeployModel()

# home page
@app.route('/')
def index():
   return render_template('index.html')

# My vintage classifier
@app.route('/predictor', methods=['POST'] )
def predictor():

    url = request.form['user_input']
    tag = model.predict(url)[0]
    sim = model.find_similar()
    similar = [sim[0], sim[1], sim[2]]    

    return render_template('prediction.html', url=url, tag=tag, similar=similar)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

