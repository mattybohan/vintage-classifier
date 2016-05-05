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
    This class takes in a Nolearn Convolutional Neural Network model stored
    in a pickle file and builds a SVM classifier and outputs it in pickle format.
    """

    def __init__(self, folder_path):
        '''
        INPUT: String
        OUTPUT: None

        Points to directory of pickle files containing Nolearn CNN and mean
        image data.
        '''
        self.folder_path = folder_path

    def load_nn(self):
        '''
        INPUT: None
        OUTPUT: None

        Loads Nolearn CNN and mean image data from pickle files.
        '''
        self.nn = joblib.load(self.folder_path + 'nolearn_nn.pkl')
        self.mean_image = joblib.load(self.folder_path + 'mean_image.pkl')

    def selected_images(self):
        '''
        INPUT: None
        OUTPUT: None

        Loads pickled LIST of filenames of images I hand selected out of the
        entire dataset.
        '''
        self.selected_items = pickle.load(open('/home/ubuntu/vintage-classifier/pkls/cleaned_data_list.pkl', "rb"))

    def process_images(self, path_to_folder, timer=False):
        '''
        INPUT: String, Boolean
        OUTPUT: None

        Processes images to extract features (ndarray), labels, and item_ids. Calls
        "process_folder" which is a multiprocessing function located outside of
        the class.
        '''
        self.features, self.labels, self.item_ids = process_folder(path_to_folder,
                self.selected_items, timer=False, threads=16)

    def chunk_features(self, iterable, chunk_size):
        '''
        INPUT: Ndarray, Int
        OUTPUT: List of Ndarrays

        Takes in list of feature numpy arrays and chunks them to a certain
        size to modulate the number of features vectorized at one time. This
        was necessary to solve GPU memory limitations of an AWS g2.2xlarge
        instance.
        '''
        for index in range(0, len(iterable), chunk_size):
            yield iterable[index:index+chunk_size]

    def vectorize_images(self):
        '''
        INPUT: None
        OUTPUT: None

        Vectorizes images by passing them through Nolearn CNN and compiles arrays
        of features (X) and labels (y).
        '''
        X = []
        for subset in self.chunk_features(self.features, 90):
            X.append(self.nn.predict_proba(subset))
        self.X = np.concatenate(X, axis=0)
        self.y = self.labels

    def train_test_split(self):
        '''
        INPUT: None
        OUTPUT: None

        Performs test-train split on dataset with 30 percent test data.
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=0.30)

    def fit(self):
        '''
        INPUT: None
        OUTPUT: None

        Fits data to SVM model.
        '''
        self.model = SVC(C=10.0, kernel='rbf', gamma=0.0001)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, path):
        '''
        INPUT: String
        OUTPUT: None

        Takes in image vector of shape (1,4096) and returns prediction.
        '''
        self.model.predict(path)

    def score(self):
        '''
        INPUT: None
        OUTPUT: None

        Scores model based on previously split test data. Must run
        train_test_split previously.
        '''
        score = self.model.score(self.X_test, self.y_test)
        print 'Model Accuracy: ', score

    def build_dataframe(self):
        '''
        INPUT: None
        OUTPUT: None

        Builds Pandas DataFrame of following columns: Features, Labels, Item_IDs
        '''
        df1 = pd.DataFrame(self.X)
        df2 = pd.DataFrame(self.y, columns=['Label'])
        df3 = pd.DataFrame(self.item_ids, columns=['Item_IDs'])
        frames = [df1, df2, df3]
        self.df = pd.concat(frames, axis=1)

    def pickle_model(self, filename='classifier.pkl'):
        '''
        INPUT: String
        OUTPUT: Pickle

        Writes pickle file containing SVM model.
        '''
        joblib.dump(self.model, self.folder_path + filename, compress=9)

    def pickle_dataframe(self, filename='dataframe.pkl'):
        '''
        INPUT: String
        OUTPUT: Pickle

        Writes pickle file containing Pandas DataFrame
        '''
        joblib.dump(self.df, self.folder_path + filename, compress=9)

def process_one(filename):
    '''
    INPUT: String
    OUTPUT: Tuple

    Given a filename by process_folder, returns a tuple containing ndarray
    of image data, label, and item_id. Extracts label and item_id information
    that is encoded in the filename.
    '''
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
    '''
    INPUT: String, List, Boolean, Int
    OUTPUT: Tuple

    Multiprocessing function that goes through folder of image files and
    returns ordered ndarrays containing the features, labels, and item_ids
    of all photos.
    '''
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

    # Path of pickle files
    path = '/home/ubuntu/vintage-classifier/pkls/'

    # Instanciate class
    classifier = BuildModel(path)

    # Load Nolearn Convolutional Neural Network
    classifier.load_nn()

    # Import list of selected images
    classifier.selected_images()

    # Process images
    classifier.process_images('/home/ubuntu/vintage-classifier/images/', timer=True)

    # Vectorize images
    classifier.vectorize_images()

    # Split data into training set and testing set
    classifier.train_test_split()

    # Fit data to SVM model
    classifier.fit()

    # Construct DataFrame containing all
    classifier.build_dataframe()

    # Pickle SVM classifier and DataFrame
    classifier.pickle_model()
    classifier.pickle_dataframe()

    print classifier.score()
