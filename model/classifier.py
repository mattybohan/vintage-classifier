import numpy as np
import io
import sys
import skimage.transform
import matplotlib.pyplot as plt
import urllib
import pandas as pd
import lasagne
import pickle
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from os import listdir
from os.path import isfile, join

"""NOTE TO LEE: I am currently rewriting this. This will be my main classifier file. It will:
-Load Nolearn model from pickle
-Load SVM classifier model from pickle
-Vectorize images (single or array)
-Predict class of image using SVM classifier (call from web app)
-Compute image similarity (for recommendation on web app)
"""



def build_net():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    output_layer = net['fc7']
    return net, output_layer

def load_pretrained(output_layer):
    model = pickle.load(open('vgg_cnn_s.pkl'))
    CLASSES = model['synset words']
    MEAN_IMAGE = model['mean image']
    lasagne.layers.set_all_param_values(output_layer, model['values'][:14])
    return model, CLASSES, MEAN_IMAGE

def prep_image(url, MEAN_IMAGE):
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_IMAGE
    return rawim, floatX(im[np.newaxis])

def process_one(path, filename, MEAN_IMAGE, output_layer):
    url = path + filename
    rawim, im = prep_image(url, MEAN_IMAGE)
    prob = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())[0]

    if '90s' in filename:
        label = '90s'
    elif '80s' in filename:
        label = '80s'
    elif '70s' in filename:
        label = '70s'
    elif '60s' in filename:
        label = '60s'
    elif '50s' in filename:
        label = '50s'

    return [label, prob]

def build_dataframe(path, start, end, MEAN_IMAGE, output_layer):

    df = pd.DataFrame()
    # Make list of all
    filenames = [f for f in listdir(path) if isfile(join(path, f))]

    for filename in filenames[start:end]:

        label_prob = process_one(path, filename, MEAN_IMAGE, output_layer)
        item_id = filename[4:-4]
        df = df.append({'item_id': item_id, 'label': label_prob[0], 'feature': label_prob[1]}, ignore_index=True)

    return df

def pickle_dataframe(df, filename):
    path = '/home/ubuntu/project/scrape/' + str(filename)
    df.to_pickle(path)

def run(path, start, end, pickle_filename):
    net, output_layer = build_net()
    model, CLASSES, MEAN_IMAGE = load_pretrained(output_layer)
    df = build_dataframe(path, start, end, MEAN_IMAGE, output_layer)
    pickle_dataframe(df, pickle_filename)


if __name__ == "__main__":
    run('/home/ubuntu/project/scrape/images/', int(sys.argv[1])-1000, int(sys.argv[1]), str(sys.argv[2]))

