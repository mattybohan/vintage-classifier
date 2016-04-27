### App imports
from flask import Flask, request
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



app = Flask(__name__)


# home page
@app.route('/')
def index():
    return "This is a test!!!"

# Form page to submit text
@app.route('/submission_page')
def submission_page():
    return '''
        <form action="/predictor" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''

# My vintage classifier
@app.route('/predictor', methods=['POST'] )
def predictor():

    url = str(request.form['user_input'])

    nn_model = pickle.load(open('/home/ubuntu/Recipes/examples/vgg_cnn_s.pkl'))

    net2 = {}
    net2['input'] = InputLayer((None, 3, 224, 224))
    net2['conv1'] = ConvLayer(net2['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
    net2['norm1'] = NormLayer(net2['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net2['pool1'] = PoolLayer(net2['norm1'], pool_size=3, stride=3, ignore_border=False)
    net2['conv2'] = ConvLayer(net2['pool1'], num_filters=256, filter_size=5, flip_filters=False)
    net2['pool2'] = PoolLayer(net2['conv2'], pool_size=2, stride=2, ignore_border=False)
    net2['conv3'] = ConvLayer(net2['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net2['conv4'] = ConvLayer(net2['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net2['conv5'] = ConvLayer(net2['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net2['pool5'] = PoolLayer(net2['conv5'], pool_size=3, stride=3, ignore_border=False)
    net2['fc6'] = DenseLayer(net2['pool5'], num_units=4096)
    net2['drop6'] = DropoutLayer(net2['fc6'], p=0.5)
    net2['fc7'] = DenseLayer(net2['drop6'], num_units=4096)
    output_layer2 = net2['fc7']

    CLASSES = nn_model['synset words']
    MEAN_IMAGE = nn_model['mean image']
    lasagne.layers.set_all_param_values(output_layer2, nn_model['values'][:14])

    from prep_image import prep_image
    rawim, im = prep_image(url, MEAN_IMAGE, CLASSES)
    vector = np.array(lasagne.layers.get_output(output_layer2, im, deterministic=True).eval())[0]


    with open('/home/ubuntu/project/my_dumped_classifier.pkl') as f:
        model = pickle.load(f)
    result = str(model.predict(vector))
    proba = str(model.predict_proba(vector))
    vect = str(vector)
    final = result + proba + vect
    pickle.dump(vector, open("vector.pkl","wb"))
    return final



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

