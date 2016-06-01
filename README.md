# Dress Code

1.  Classify a vintage dress by decade
2.  Find similar items on Etsy

## 1. Overview

  The initial objective of Dress Code was to see if style–specifically
  vintage–could be featurized using computer vision techniques, allowing for
  classifying vintage dresses by decade: 1950s, 1960s, 1970s, 1980s,
  or 1990s. This is the sort of classification someone knowledgeable of fashion
  might make while picking items ("popping tags") at a thrift shop. A pretrained
  Convolutional Neural Network was employed to featurize the images collected
  from Etsy with classification of feature vectors by traditional
  classification models.

  The resulting model was successful in classification, obtaining a maximum
  of 87% accuracy in the binary case and 53% with all five classes. With the
  success of featurizing style with the CNN, a recommender was developed to
  that returns dresses that are similar in shape, pattern, and color–the
  visual features the model is capable of recognizing.  

  A web application was developed that allows users to:

  * Classify an image of a vintage dress by decade
  * See recommendation of similar dresses found on Etsy

  Check it out: http://dress-code.tech


## 2. Details

![alt text](https://github.com/mattybohan/vintage-classifier/blob/master/images/pipeline.jpg "Pipeline")

#### Scraping

Scraping was performed using BeautifulSoup.

#### Image Preprocessing

Image precessing included loading images with scikit-image and manual cropping.
Additionally, Haar Cascades detection implemented in OpenCV was used for detecting
facings so they could be removed. Ultimately, the images were resized to
224x224.

#### Featurize

Image featurizing was achieved using the VGG_CNN_S Convolutional Neural Network,
(CNN) pretrained on ImageNet (1000 classes). VGG_CNN_S was trained by the Visual
Geometry Group at Oxford Univeristy. More information on this CNN can be
found elsewhere:

    The Devil is in the Details: An evaluation of recent feature encoding methods
    K. Chatfield, V. Lempitsky, A. Vedaldi and A. Zisserman, In Proc. BMVC, 2011.
http://www.robots.ox.ac.uk/~vgg/research/deep_eval/

The model was implemented in Nolearn, a Lasagne wrapper. The output layer
(1000 dimensions) was removed and the output of the last DenseLayer (4096
dimensional) were used as feature vectors.

Nolearn was employed instead of Lasagne as it led to a 20 fold speedup.
An initial Lasagne model took over 60 minutes to vectorize 5000 images on an
AWS EC2 g2.2xlarge instance running in GPU mode with Nvidia CUDA, while
the Nolearn model took just six minutes. Further optimization was achieved using
multiprocessing, reducing the total time to three minutes.

T-distributed stochastic neighbor embedding was used to visualize the vectors:

![alt text](https://github.com/mattybohan/vintage-classifier/blob/master/images/cnn_embed_2k.jpg "t-SNE")

#### Classify

Support Vector Machine and Random Forest models were used to classify
the feature vectors. The following binary accuracies were achieved using SVM:

![alt text](https://github.com/mattybohan/vintage-classifier/blob/master/images/accuracy.png "Accuracy")


## 3. Code

#### Scrape

**scrape.py** is a custom web scraper that extracts images, labels, and other relavent
information from Etsy shops.

#### Model

**build_nn.py** builds a Nolearn CNN from pickled weights and biases formatted
                for Lasagne.

**build_classifier.py** builds a SVM classifier.

**model.py** is used to load the models (Nolearn, SVM) on initial launch of the
                web application.

**image_preprocess.py** processes raw images files into numpy arrays.

#### App

**app.py** is a Flask web application.



## 4. Installation

The following modules are required:

1. Lasagne
2. Nolearn
3. Nvidia CUDA
4. Sklearn
5. Numpy
6. Pandas
7. OpenCV
8. BeautifulSoup
