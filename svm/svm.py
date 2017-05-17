#Abbie Jones
#Assignment 3
#CS 594
#Support Vector Machine
#May 9th, 2017
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.svm as svm
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import random
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

#Experiment 1

data = pd.read_csv('spambase.data', header=None, index_col=57) #read in data

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5) #split dat
X_train.shape, X_test.shape, y_train.shape, y_test.shape

scaler = sklearn.preprocessing.StandardScaler().fit(X_train)    #scale data

X_train = pd.DataFrame(scaler.transform(X_train), index=y_train)  

X_test = pd.DataFrame(scaler.transform(X_test), index=y_test)

clf = svm.SVC(kernel='linear')  #create linear support vector
y_test_predicted = clf.fit(X_train,y_train).predict(X_test) #train and predict

accuracy = accuracy_score(y_test,y_test_predicted)
precision = precision_score(y_test,y_test_predicted)
recall = recall_score(y_test,y_test_predicted)

print("{0}".format(accuracy))
print("{0}".format(precision))
print("{0}".format(recall))

fpr, tpr, thresholds = roc_curve(y_test, y_test_predicted)  #create roc curve
roc_auc = auc(fpr, tpr)

#plot roc curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Experiment 2

highestWeights = [] #array to hold list of 56 highest weighted characteristics
weights = clf.coef_ #weights from clf
weights.setflags(write=1)   #make this writeable
for x in range(1,57):
    highestWeights.append((np.argmax(weights))) #find highest weight
    weights[0,np.argmax(weights)] = -100    #arbitrarily set that weight to a really high value so it's ignored the next iteration through for loop

print("highest weights: {0}".format(highestWeights))

exp2Accuracy = []   #array to hold accuracies per experiment
trainMod = pd.DataFrame(index=y_train)  #create dataframe to keep appending to training data (need to retrain for each set)
testMod = pd.DataFrame(index=y_test)    #create dataframe to keep appending to testing data
for x in range(0,56): 
    train = pd.DataFrame(X_train[highestWeights[x]])    #find column with highest weight in training data
    test = pd.DataFrame(X_test[highestWeights[x]])  #find column with hihgest weight in testing data
    trainMod = pd.concat([trainMod,train],axis=1)   #append to our trainMod data frame
    testMod = pd.concat([testMod,test],axis=1)  #append to our testMod data frame
    clf = svm.SVC(kernel='linear')  #set up classifier
    y_test_predicted = clf.fit(trainMod,y_train).predict(testMod)   #train and test!
    accuracy = accuracy_score(y_test,y_test_predicted)  #find accuracy
    exp2Accuracy.append(accuracy)       #append accuracy
print(exp2Accuracy)

#Experiment 3

exp3Accuracy = []   #array to hold accuracies per experiment
trainMod = pd.DataFrame(index=y_train)
testMod = pd.DataFrame(index=y_test)
randomWeights = random.sample(range(57),56) #create array of randomly chosen characteristics
for x in range(0,56): 
    train = pd.DataFrame(X_train[randomWeights[x]]) #see experiment 2 for information about everything else
    test = pd.DataFrame(X_test[randomWeights[x]])
    trainMod = pd.concat([trainMod,train],axis=1)
    testMod = pd.concat([testMod,test],axis=1)
    clf = svm.SVC(kernel='linear')
    y_test_predicted = clf.fit(trainMod,y_train).predict(testMod)
    accuracy = accuracy_score(y_test,y_test_predicted)
    exp3Accuracy.append(accuracy)
print(exp3Accuracy)


       
        

