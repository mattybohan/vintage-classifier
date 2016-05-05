import matplotlib.pyplot as plt
import io
import skimage.transform
import urllib
import lasagne
import pickle
import numpy as np
from lasagne.utils import floatX
from sklearn.externals import joblib

def basic(url):
    '''
    INPUT: String
    OUTPUT: Ndarray, Ndarray

    Takes path to image file and returns 
    '''
    # Download/load image
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)

    # Resize to 256x256
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Resize to 224x224, taking center
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = np.copy(im).astype('uint8')
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    im = im[::-1, :, :]

    # Import mean image
    mean_image = joblib.load('/home/ubuntu/vintage-classifier/pkls/mean_image.pkl')
    im = im - mean_image
    return rawim, floatX(im[np.newaxis])
