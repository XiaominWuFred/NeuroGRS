'''
 author: xiaomin wu
 date: 1/16/2020
 '''
from Location import getLocation
import numpy as np

from DataLoader import DataLoader as DL;
from metrics import metrics as M;
import NNet;
import tensorflow as tf;
from tensorflow import keras;
from SVM import SVM;
import matplotlib.pyplot as plt
from keras.regularizers import l1
from keras.regularizers import l1_l2
from keras.regularizers import l2
from keras import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Reshape
from keras.constraints import NonNeg
from keras.layers.core import Dropout

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from docutils.nodes import row

from ReadDataCNN import balanceDataCNN
from sklearn import svm
from GroupDataDim import slidingWindow
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
from Location import matchX
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from compactImg import combined

def computeSortedNeuron(locationFilePath, X):
    location = getLocation(locationFilePath)
    sumLoca = np.sum(location,axis = 1)
    sumLoca = sumLoca.reshape((1,sumLoca.shape[0]))
    
    newX = X.reshape((X.shape[0],273))
    arr = np.concatenate((sumLoca,newX),axis = 0)
    arr = arr[:,arr[0].argsort()]
    sr = arr[1:arr.shape[0],:]
    return sr

