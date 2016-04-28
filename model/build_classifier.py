import numpy as np
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

class BuildClassifer(object):
    """


    """
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_nn(self):
        self.nn = joblib.load(self.folder_path + 'nolearn_nn.pkl')
        self.mean_image = joblib.load(self.folder_path + 'mean_image.pkl')     

    def selected_images(self):
        image_list = pickle.load(open('/home/ubuntu/vintage-classifier/pkls/cleaned_data_list.pkl', "rb"))
        self.selected_items = [item[4:-4] for item in image_list]
 
    def process_images(self, path_to_folder, timer=False):
        self.features, self.y, self.item_ids = process_folder(path_to_folder,
                timer=False, threads=16)
        print self.features.shape

    def chunk_features(self, iterable, chunk_size):
        for index in range(0, len(iterable), chunk_size):
            yield iterable[index:index+chunk_size]
    
    def vectorize_images(self):
        X = []
        for subset in self.chunk_features(self.features, 75):
            X.append(self.nn.predict_proba(subset))
        self.X = np.concatenate(X, axis=0)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=0.30)
	print self.X_train.shape
        print self.X_test.shape
	print self.y_train.shape
	print self.y_test.shape
    
    def fit(self):
        self.model = SVC(C=10.0, kernel='rbf', gamma=0.0001)
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        pass

    def score(self):
        score = self.model.score(self.X_test, self.y_test)
        print score

    def build_dataframe(path, start, end):

        df = pd.DataFrame()
        # Make list of all
        filenames = [f for f in listdir(path) if isfile(join(path, f))]

        for filename in filenames[start:end]:

            label_prob = process_one(path, filename)
            df = df.append({'label': label_prob[0], 'feature': label_prob[1]}, ignore_index=True)

        return df

    def pickle_model(self):
        pass

    def pickle_dataframe(self):
        pass

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

def process_folder(folder_path, timer=False, threads=16):
    if timer:
        start_time = time.time()
    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    p = Pool(threads)
    result = p.map(process_one, filenames)
    lists = map(list, zip(*result))
    features = np.concatenate(lists[0], axis=0)
    labels = np.array(lists[1])
    item_id = np.array(lists[2])
    if timer:
        print("--- %s seconds ---" % (time.time() - start_time))
    return (features, labels, item_id)


if __name__ == "__main__":

    path = '/home/ubuntu/vintage-classifier/pkls/'
    classifier = BuildClassifer(path)
    classifier.load_nn()
    classifier.selected_images()
    classifier.process_images('/home/ubuntu/vintage-classifier/images/', timer=True)
    classifier.vectorize_images()
    classifier.train_test_split()
    classifier.fit()
    print classifier.score()
