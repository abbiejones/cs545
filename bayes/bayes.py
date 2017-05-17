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

indicesTrain = X_train.index.get_values() 

spamTrain = np.count_nonzero(indicesTrain)

spamTrain = float(spamTrain)/float(numTrain)
notSpamTrain = 1 - spamTrain

print(spamTrain)
print(notSpamTrain)

meanSpamTrain = []
meanNotSpamTrain = []
stdSpamTrain = []
stdNotSpamTrain = []

for x in range(0,57):
    if (indicesTrain[x] == 1): 
        meanSpamTrain.append((X_train[[x]].mean()).item())
        stdSpamTrain.append((X_train[[x]].std()).item())
    else:
        meanNotSpamTrain.append((X_train[[x]].mean()).item())
        stdNotSpamTrain.append((X_train[[x]].std()).item())
    

predictedTest = []
for x in range(len(meanSpamTrain)):
    spam = []
    for y in range(0,57):
        gauss = float((1 / ((math.sqrt(float(2 * (math.pi)))) * stdSpamTrain[y])))
        gauss2 = float(math.exp(math.pow(X_test.iloc[x][y] - meanSpamTrain[y],2)/(2 * math.pow(stdSpamTrain[y],2))))
        gauss = float(gauss) * float(gauss2)

        gaussNot = float((1 / ((math.sqrt(float(2 * (math.pi)))) * stdNotSpamTrain[y])))
        gaussNot2 = float(math.exp(math.pow(X_test.iloc[x][y] - meanNotSpamTrain[y],2)/(2 * math.pow(stdNotSpamTrain[y],2))))
        gaussNot = float(gaussNot) * float(gaussNot2)

        spam.append(gauss)
        spam.append(gaussNot)

        spam.append(gauss)
        spam.append(gaussNot)


    spam = np.asarray(spam)
    notSpam = np.asarray(notSpam)

    spam = np.sum(np.log(spam))
    notSpam = np.sum(np.log(notSpam))







    

    

