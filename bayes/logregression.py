#Abbie Jones
#Assignment 4
#CS 594
#Naive Bayes Classification and Logistic Regression
#May 9th, 2017

import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.svm as svm
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import random
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.naive_bayes import GaussianNB
import math

data = pd.read_csv('spambase.data', header=None, index_col=57) #read in data

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5) #split dat
X_train.shape, X_test.shape, y_train.shape, y_test.shape

numTrain = len(X_train.index)
numTest = len(X_test.index)


