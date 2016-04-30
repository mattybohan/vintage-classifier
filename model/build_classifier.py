import numpy as np
import pandas as pd
import time
import cPickle as pickle
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
from sklearn.svm import SVC
from sklearn.externals import joblib
from image_preprocess import basic
from lasagne.utils import floatX
from lasagne.updates import adam
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import train_test_split

class BuildModel(object):
    """


    """
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_nn(self):
        self.nn = joblib.load(self.folder_path + 'nolearn_nn.pkl')
        self.mean_image = joblib.load(self.folder_path + 'mean_image.pkl')     

    def selected_images(self):
        self.selected_items = pickle.load(open('/home/ubuntu/vintage-classifier/pkls/cleaned_data_list.pkl', "rb"))
 
    def process_images(self, path_to_folder, timer=False):
        self.features, self.labels, self.item_ids = process_folder(path_to_folder,
                self.selected_items, timer=False, threads=16)

    def chunk_features(self, iterable, chunk_size):
        for index in range(0, len(iterable), chunk_size):
            yield iterable[index:index+chunk_size]
    
    def vectorize_images(self):
        X = []
        for subset in self.chunk_features(self.features, 90):
            X.append(self.nn.predict_proba(subset))
        self.X = np.concatenate(X, axis=0)
        self.y = self.labels

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=0.30)
    
    def fit(self):
        self.model = SVC(C=10.0, kernel='rbf', gamma=0.0001)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, path):
        self.model.predict(path)

    def score(self):
        score = self.model.score(self.X_test, self.y_test)
        print 'Model Accuracy: ', score
 
    def build_dataframe(self):
        df1 = pd.DataFrame(self.X)
        df2 = pd.DataFrame(self.y, columns=['Label'])
        df3 = pd.DataFrame(self.item_ids, columns=['Item_IDs'])
        frames = [df1, df2, df3]
        self.df = pd.concat(frames, axis=1)

    def pickle_model(self, filename='classifier.pkl'):
        joblib.dump(self.model, self.folder_path + filename, compress=9)

    def pickle_dataframe(self, filename='dataframe.pkl'):
        joblib.dump(self.df, self.folder_path + filename, compress=9)

def process_one(filename):
    url = '/home/ubuntu/vintage-classifier/images/' + filename
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
    item_id = filename[4:-4]
    return (basic(url)[1], label, item_id)

def process_folder(folder_path, selected_files, timer=False, threads=16):
    if timer:
        start_time = time.time()
    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    selected = [f for f in filenames if f in selected_files]
    p = Pool(threads)
    result = p.map(process_one, selected)
    lists = map(list, zip(*result))
    features = np.concatenate(lists[0], axis=0)
    labels = np.array(lists[1])
    item_id = np.array(lists[2])
    if timer:
        print("--- %s seconds ---" % (time.time() - start_time))
    return (features, labels, item_id)


if __name__ == "__main__":

    path = '/home/ubuntu/vintage-classifier/pkls/'
    classifier = BuildModel(path)
    classifier.load_nn()
    classifier.selected_images()
    classifier.process_images('/home/ubuntu/vintage-classifier/images/', timer=True)
    classifier.vectorize_images()
    classifier.train_test_split()
    classifier.fit()
    classifier.build_dataframe()
    classifier.pickle_model()
    classifier.pickle_dataframe()
    print classifier.score()
