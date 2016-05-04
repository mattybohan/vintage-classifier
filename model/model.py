import numpy as np
import pandas as pd
from sklearn.externals import joblib
from image_preprocess import basic
from sklearn.metrics.pairwise import cosine_similarity


class DeployModel(object):

    """This class deploys a 
    """

    def __init__(self):
        self.nn = joblib.load('/home/ubuntu/vintage-classifier/pkls/nolearn_nn.pkl')
        self.classifier = joblib.load('/home/ubuntu/vintage-classifier/pkls/classifier.pkl')
        self.df = joblib.load('/home/ubuntu/vintage-classifier/pkls/dataframe.pkl')
        self.X = self.df.drop(['Label', 'Item_IDs'], axis=1).values
        self.y = self.df['Label']
        self.item = self.df['Item_IDs']

    def predict(self, path):
        self.preprocessed = basic(path)[1]
        self.vector = self.nn.predict_proba(self.preprocessed)
        self.prediction = self.classifier.predict(self.vector) # finish predict function in build_classifier
        print self.prediction

    def find_similar(self):
        top3 = cosine_similarity(self.X, self.vector).argsort(axis=0)[-3:][::1].reshape((3))
        self.similar = list(self.df.iloc[top3]['Item_IDs'])
        print self.similar
