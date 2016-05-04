import sys
import numpy as np
import lasagne
import cPickle as pickle
from sklearn.externals import joblib
from lasagne.utils import floatX
from lasagne.updates import adam
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from nolearn.lasagne import NeuralNet


class LasagneToNolearn(object):

    """This class builds the VGG_CNN_S model from pickled weights and
    biases (vgg_cnn_s.pkl) in Lasagne and converts the model for use in Nolearn.
    Nolearn is a Lasagne wrapper that is used here to facilitate and increase speed
    of vectorizing images.

    VGG_CNN_S is a Convolutional Neural Network (CNN) trained by the Visual
    Geometry Group at Oxford Univeristy. More information on this CNN can be
    found elsewhere:

    The Devil is in the Details: An evaluation of recent feature encoding methods
    K. Chatfield, V. Lempitsky, A. Vedaldi and A. Zisserman, In Proc. BMVC, 2011.
    http://www.robots.ox.ac.uk/~vgg/research/deep_eval/

    vgg_cnn_s.pkl was obtained from the Lasagne Model Zoo:
    https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl
    """

    def __init__(self, path_to_pkl):
        '''
        INPUT: Local path to vgg_cnn_s.pkl
        OUTPUT:

        Points to path of the stored weights and biases.
        '''
        self.path_to_pkl = path_to_pkl

    def lasagne_layers_method(self):
        '''
        INPUT: None
        OUTPUT: Dict

        Creates dictionary of vgg_cnn_s model Lasagne layer objects. Here the
        original output layer (softmax, 1000 classes) has been removed and
        the output layer returns a vector of shape (1,4096).
        '''
        # Create dictionary of VGG_CNN_S model layers
        self.lasagne_layers = {}
        self.lasagne_layers['input'] = InputLayer((None, 3, 224, 224))
        self.lasagne_layers['conv1'] = ConvLayer(self.lasagne_layers['input'],
                num_filters=96, filter_size=7, stride=2, flip_filters=False)
        self.lasagne_layers['norm1'] = NormLayer(self.lasagne_layers['conv1'],
                alpha=0.0001)
        self.lasagne_layers['pool1'] = PoolLayer(self.lasagne_layers['norm1'],
                pool_size=3, stride=3, ignore_border=False)
        self.lasagne_layers['conv2'] = ConvLayer(self.lasagne_layers['pool1'],
                num_filters=256, filter_size=5, flip_filters=False)
        self.lasagne_layers['pool2'] = PoolLayer(self.lasagne_layers['conv2'],
                pool_size=2, stride=2, ignore_border=False)
        self.lasagne_layers['conv3'] = ConvLayer(self.lasagne_layers['pool2'],
                num_filters=512, filter_size=3, pad=1, flip_filters=False)
        self.lasagne_layers['conv4'] = ConvLayer(self.lasagne_layers['conv3'],
                num_filters=512, filter_size=3, pad=1, flip_filters=False)
        self.lasagne_layers['conv5'] = ConvLayer(self.lasagne_layers['conv4'],
                num_filters=512, filter_size=3, pad=1, flip_filters=False)
        self.lasagne_layers['pool5'] = PoolLayer(self.lasagne_layers['conv5'],
                pool_size=3, stride=3, ignore_border=False)
        self.lasagne_layers['fc6'] = DenseLayer(self.lasagne_layers['pool5'],
                num_units=4096)
        self.lasagne_layers['drop6'] = DropoutLayer(self.lasagne_layers['fc6'],
                p=0.5)
        self.lasagne_layers['fc7'] = DenseLayer(self.lasagne_layers['drop6'],
                num_units=4096)

    def build_lasagne(self):
        '''
        INPUT: None
        OUTPUT: None

        Builds the CNN model using Lasagne.
        '''
        model = pickle.load(open(self.path_to_pkl))
        output_layer = self.lasagne_layers['fc7']
        self.mean_image = model['mean image']
        lasagne.layers.set_all_param_values(output_layer, model['values'][:14])

    def extract_layers(self):
        '''
        INPUT: None
        OUTPUT: None

        Extracts relavent layers from Lasagne model for use with Nolearn model.
        '''
        self.extracted_layers = {}
        for layer in self.lasagne_layers:
            if layer[:4] != 'drop' and layer != 'input' and \
                layer[:4] != 'pool' and layer[:4] != 'norm':
                self.extracted_layers[layer] = [self.lasagne_layers[layer].W.get_value(),
                self.lasagne_layers[layer].b.get_value()]

    def nolearn_layers_method(self):
        '''
        INPUT: None
        OUTPUT: None

        Creates list of layers for Nolearn model.
        '''
        self.nolearn_layers = [(InputLayer, {'name': 'input',
                'shape': (None, 3, 224, 224)}),
        (ConvLayer, {'name': 'conv1', 'num_filters': 96, 'filter_size': (7,7),
                'stride': 2, 'flip_filters': False, 'W': self.extracted_layers['conv1'][0],
                'b': self.extracted_layers['conv1'][1]}),
        (NormLayer, {'name': 'norm11', 'alpha': .0001}),
        (PoolLayer, {'name': 'pool1', 'pool_size': (3,3), 'stride': 3,
                'ignore_border': False}),
        (ConvLayer, {'name': 'conv2', 'num_filters': 256, 'filter_size': (5,5),
                'flip_filters': False, 'W': self.extracted_layers['conv2'][0],
                'b': self.extracted_layers['conv2'][1]}),
        (PoolLayer, {'name': 'pool2', 'pool_size': (2,2), 'stride': 2,
                'ignore_border': False}),
        (ConvLayer, {'name': 'conv3', 'num_filters': 512, 'filter_size': (3,3),
                'flip_filters': False, 'pad': 1, 'W': self.extracted_layers['conv3'][0],
                'b': self.extracted_layers['conv3'][1]}),
        (ConvLayer, {'name': 'conv4', 'num_filters': 512, 'filter_size': (3,3),
                'flip_filters': False, 'pad': 1, 'W': self.extracted_layers['conv4'][0],
                'b': self.extracted_layers['conv4'][1]}),
        (ConvLayer, {'name': 'conv5', 'num_filters': 512, 'filter_size': (3,3),
                'flip_filters': False, 'pad': 1, 'W': self.extracted_layers['conv5'][0],
                'b': self.extracted_layers['conv5'][1]}),
        (PoolLayer, {'name': 'pool5', 'pool_size': (3,3), 'stride': 3,
                'ignore_border': False}),
        (DenseLayer, {'name': 'fc6', 'num_units': 4096,
                'W': self.extracted_layers['fc6'][0], 'b': self.extracted_layers['fc6'][1]}),
        (DropoutLayer, {'name': 'drop6', 'p': 0.5}),
        (DenseLayer, {'name': 'fc7', 'num_units': 4096, 'W': self.extracted_layers['fc7'][0],
                'b': self.extracted_layers['fc7'][1]})]

    def build_nolearn(self):
        '''
        INPUT: None
        OUTPUT: None

        Builds CNN model using Nolearn.
        '''
        self.nolearn_layers_method()
        self.nn = NeuralNet(layers=self.nolearn_layers, update=adam,
                update_learning_rate=0.0002)
        self.nn.initialize()

    def to_pickle(self, path):
        '''
        INPUT: Local path where pickle files will be stored
        OUTPUT: Two pickle files

        Pickles the Nolearn model as well as the mean image.
        '''
        joblib.dump(self.nn, '/home/ubuntu/vintage-classifier/pkls/nolearn_nn.pkl', compress=9)
        joblib.dump(self.mean_image, '/home/ubuntu/vintage-classifier/pkls/mean_image.pkl', compress=9)

if __name__ == "__main__":

    # Set max recursion depth in python to prevent errors
    sys.setrecursionlimit(10000)

    # Instanciate class
    model = LasagneToNolearn('/home/ubuntu/vintage-classifier/pkls/vgg_cnn_s.pkl')

    # Create dictionary of CNN layers for Lasagne
    model.lasagne_layers_method()

    # Build Lasagne CNN
    model.build_lasagne()

    # Extract necessary layers from Lasagne CNN
    model.extract_layers()

    # Build Nolearn CNN
    model.build_nolearn()

    # Save Nolearn CNN in Pickle format
    model.to_pickle('/home/ubuntu/vintage-classifier/pkls/')
