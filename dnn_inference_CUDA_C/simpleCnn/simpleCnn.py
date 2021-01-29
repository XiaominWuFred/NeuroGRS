'''
 author: xiaomin wu
 date: 1/16/2020
 '''
#import cv2 as c
import numpy as np;
from DataLoader import DataLoader as DL;
#from metrics import metrics as M;
#import NNet;
import tensorflow as tf;
from tensorflow import keras;
#from SVM import SVM;
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
#from docutils.nodes import row

from ReadDataCNN import balanceDataCNN
from sklearn import svm
from GroupDataDim import slidingWindow
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
from Location import getLocation
from Location import matchX
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from compactImg import combined

from weightCutor import weightCutorWM,weightCutorRandom,weightCutor_convWM,weightCutor_convRandom,weightCutor_CD1006WM,weightCutor_CD1006Random,weightCutor_CD1005Random,weightCutor_CD1005WM,weightCutor_CD1004WM,weightCutor_CD1004Random

from weightCutor import weightCutorIn,weightCutorInconv
from weightCutor import weightShare,weightShareconv
import sys
from keras import backend as K
from keras.models import model_from_json
import csv
import os

files = [
    ['1004_1', 0, 1, 1],

];

#1=mlp 2=cnn 3=cnn 4=mlp

model = 0
modtp = 'cnn'
folder = 'extractParas_2conv2'

#os.system("cd ../"+folder+"/conv1out\nrm *")
#os.system("cd ../"+folder+"/maxPoolout/\nrm *")

CNNDim = 273
version = 'NNN2.1'

train = 0.6;
val = 0.1;
test = 0.3;

loader = DL();

i = 0;
dataSet = [];
while (i != len(files)):
    data, length = loader.extractBalanceDataShuffled(files[i][0], files[i][1], files[i][2], files[i][3]);
    print('data length:', length, '\n');
    dataSet.append(data);
    i = i + 1;

    re = dataSet[0];
if (len(dataSet) == 1):
    pass;
else:
    i = 1;
    while (i != len(dataSet)):
        re = loader.combineTwo(re, dataSet[i]);
        i = i + 1;

X, Y = loader.splitXY(re);

if modtp == 'cnn':

    if (CNNDim == 273):
        X = combined(CNNDim, '../dataset/1004/CELLXY_1004.csv', X)  # 140 for 1005 114 for 1006 273 for 1004
    if (CNNDim == 114):
        X = combined(CNNDim, '../dataset/1004/CELLXY_1006.csv', X)  # 140 for 1005 114 for 1006 273 for 1004
    if (CNNDim == 140):
        X = combined(CNNDim, '../dataset/1004/CELLXY_1005.csv', X)  # 140 for 1005 114 for 1006 273 for 1004

if modtp == 'cnn':
    X = X.reshape((X.shape[0],X.shape[1],X.shape[2],1))
if modtp == 'mlp':
    pass

N = int(X.shape[0]);
if modtp == 'cnn':
    D = int(X.shape[1]);
    D1 = int(X.shape[2]);

C = int(np.max(Y) + 1);

if modtp == 'cnn':
    print("data total length:", N, "data dimension:", D, "data dimension2: ", D1, "data classes:", C);
sampleSize = X.shape[0];
# k_fold
k = 10
partial = 1 / k
X_parts = []
Y_parts = []
X_test = None
Y_test = None
X_test = (X[np.arange(int(sampleSize * partial)), :])

for i in range(1,k):
    X_parts.append(X[np.arange(int(sampleSize * i*partial), int(sampleSize * ((i+1) * partial)))])

Y_test = (Y[np.arange(int(sampleSize * partial))])
print("test samples: "+str(len(Y_test)))
for i in range(1,k):
    Y_parts.append(Y[np.arange(int(sampleSize * i*partial), int(sampleSize * ((i+1) * partial)))])

trainX = []
trainY = []

X_val = None
Y_val = None
X_train = None
Y_train = None
X_val = None
Y_val = None


def concaElement(train):
    conca = []
    for i in range(len(train)):
        if (i == 0):
            conca = train[i]

        else:
            conca = np.concatenate((conca, train[i]), 0)
    return conca


result = {};
sksvm_his = [];
svm_his = [];
lda_his = [];
nn1_his = [];
nn2_his = [];
CNN2D_his = [];
CNN1D_his = [];

for i in range(1):
    trainX = []
    trainY = []
    for j in range(k - 1):
        if (j == i):
            X_val = X_parts[j]
            Y_val = Y_parts[j]
        else:
            trainX.append(X_parts[j])
            trainY.append(Y_parts[j])
    X_train = concaElement(trainX)
    Y_train = concaElement(trainY)

    # data ratio, 1's percent
    train_ratio = np.sum(Y_train) / int(sampleSize * 0.9);
    test_ratio = np.sum(Y_test) / int(sampleSize * 0.1);
    print(train_ratio, test_ratio);

    # model create, train and test:
    if modtp == 'cnn':
        Y_trainConv = np.zeros((Y_train.shape[0], 2));
        Y_trainConv[np.arange(Y_train.shape[0]), Y_train.astype(int)] = 1;
        # Y_trainConv = Y_trainConv.astype(int);
        Y_testConv = np.zeros((Y_test.shape[0], 2));
        Y_testConv[np.arange(Y_test.shape[0]), Y_test.astype(int)] = 1;
        # Y_testConv = Y_testConv.astype(int);
        Y_valConv = np.zeros((Y_val.shape[0], 2));
        Y_valConv[np.arange(Y_val.shape[0]), Y_val.astype(int)] = 1;
    if modtp == 'mlp':
        pass


    dropoutDecay = 0.95
    DORate = 0.5
    DORate_g = 0.5
    DORate_r = 0.5
    # model comparing:
    ###############model1

    conv_net = Sequential()

    # convolution layer 1
    if model == 0:
        conv_net.add(Conv2D(32, (2, 2), activation='relu', input_shape=(D, D1, 1), name="conv1")) #was 32 filter
        # fully connected
        conv_net.add(MaxPooling2D(pool_size=(2,2),name="maxpool"))
        conv_net.add(Conv2D(16, (2, 2), activation='relu', name="conv2"))
        conv_net.add(Flatten(name="flatten"))
        conv_net.add(Dense(32, activation='relu', use_bias=True, name="dense1")) #was 16 nodes
        conv_net.add(Activation('relu'))
        conv_net.add(Dense(2, activation='softmax', use_bias=True, name='dense2'))
        conv_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        conv_net.fit(X_train, Y_trainConv, batch_size=32, epochs=150, verbose=0)
    #original model MLP in compact DNN paper
    if model == 1:
        conv_net.add(Dense(2,activation='relu',use_bias=True,name='dense1',input_dim = 273))
        conv_net.add(Dropout(0.5))
        conv_net.add(Dense(2, activation='relu', use_bias=True, name="dense2")) #was 16 nodes
        conv_net.add(Dropout(0.5))
        conv_net.add(Dense(2, activation='relu', use_bias=True, name="dense3")) #was 16 nodes
        conv_net.add(Dropout(0.5))
        conv_net.add(Dense(2, activation='softmax', use_bias=True, name="dense4")) #was 16 nodes
        conv_net.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        conv_net.fit(X_train, Y_train, batch_size=32, epochs=150, verbose=0)

    #original model CNN in compact DNN paper
    if model == 2:
        conv_net.add(Conv2D(2, (2, 2), activation='relu', input_shape=(D, D1, 1), name="conv1"))  # was 32 filter
        # fully connected
        conv_net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool"))
        conv_net.add(Conv2D(5, (2, 2), activation='relu', input_shape=(D, D1, 1), name="conv2"))  # was 32 filter
        conv_net.add(Flatten(name="flatten"))
        conv_net.add(Dense(2,activation='relu',use_bias=True,name='dense1',input_dim = 273))
        conv_net.add(Dense(2, activation='relu', use_bias=True, name="dense2"))
        conv_net.add(Dense(2, activation='relu', use_bias=True, name="dense3"))
        conv_net.add(Dense(2, activation='softmax', use_bias=True, name="dense4"))
        conv_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        conv_net.fit(X_train, Y_trainConv, batch_size=32, epochs=150, verbose=0)
    if model == 3:
        conv_net.add(Conv2D(2, (2, 2), activation='relu', input_shape=(D, D1, 1), name="conv1"))  # was 32 filter
        # fully connected
        conv_net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool"))
        conv_net.add(Conv2D(14, (2, 2), activation='relu', input_shape=(D, D1, 1), name="conv2"))  # was 32 filter
        conv_net.add(Flatten(name="flatten"))
        conv_net.add(Dense(2, activation='relu', use_bias=True, name='dense1', input_dim=273))
        conv_net.add(Dense(2, activation='softmax', use_bias=True, name="dense2"))
        conv_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        conv_net.fit(X_train, Y_trainConv, batch_size=32, epochs=150, verbose=0)
    if model == 4:
        conv_net.add(Dense(32,activation='relu',use_bias=True,name='dense1',input_dim = 273))
        conv_net.add(Dropout(0.5))
        conv_net.add(Dense(2, activation='softmax', use_bias=True, name="dense2")) #was 16 nodes
        conv_net.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        conv_net.fit(X_train, Y_train, batch_size=32, epochs=150, verbose=0)



    CNN2D_his_val = []
    test_acc_his = []
    test_acc_his_r = []
    test_acc_his_g = []
    val_acc_his = []
    val_acc_his_r = []
    val_acc_his_g = []

    if modtp == 'cnn':
        score = conv_net.evaluate(X_val, Y_valConv, verbose=0)
    else:
        score = conv_net.evaluate(X_val, Y_val, verbose=0)
    print("Val ACC: ", score[1])
    acc_val = score[1]
    CNN2D_his_val.append(acc_val)
    val_acc_his.append(acc_val)
    val_acc_his_g.append(acc_val)
    val_acc_his_r.append(acc_val)
    originalACC_val = acc_val

    CNN2D_his_test = []

    if modtp == 'cnn':
        score = conv_net.evaluate(X_test, Y_testConv, verbose=0)
    else:
        score = conv_net.evaluate(X_test, Y_test, verbose=0)
    acc_test = score[1]
    print("Test ACC: ", acc_test)

    with open("../"+folder+"/accTest.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([str(acc_test)])

    #observe output of first layer
    from keras.models import Model
    import extraCnnWeights
    if model == 0:
        layer_name = 'conv1'
        intermediate_layer_model = Model(inputs=conv_net.input,
                                         outputs=conv_net.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(X_test)

        layer_name = 'conv2'
        intermediate_layer_model2 = Model(inputs=conv_net.input,
                                         outputs=conv_net.get_layer(layer_name).output)
        intermediate_output2 = intermediate_layer_model2.predict(X_test)

        layer_name = 'dense1'
        dense_layer_model = Model(inputs=conv_net.input,
                                         outputs=conv_net.get_layer(layer_name).output)
        dense_output = dense_layer_model.predict(X_test)
        layer_name = 'dense2'
        dense_layer2_model = Model(inputs=conv_net.input,
                                  outputs=conv_net.get_layer(layer_name).output)
        dense2_output = dense_layer2_model.predict(X_test)
        layer_name = 'maxpool'
        maxpool_model = Model(inputs=conv_net.input,
                                  outputs=conv_net.get_layer(layer_name).output)
        maxpool_output = maxpool_model.predict(X_test)
        layer_name = 'flatten'
        flatten_model = Model(inputs=conv_net.input,
                                  outputs=conv_net.get_layer(layer_name).output)
        flatten_output = flatten_model.predict(X_test)
        flattenin_model = Model(inputs=conv_net.input,
                                  outputs=conv_net.get_layer(layer_name).input)
        flattenin_output = flattenin_model.predict(X_test)
        #getting weight info
        lw = conv_net.get_layer("conv1").get_weights()
        lw2 = conv_net.get_layer("conv2").get_weights()
        dw1 = conv_net.get_layer("dense1").get_weights()
        dw2 = conv_net.get_layer("dense2").get_weights()

        
        exr = extraCnnWeights.ExtraCnnWeights(folder)
        exr.extractCnnLayerW(lw,'cnn1')
        exr.extractCnnLayerW(lw2, 'cnn2')
        exr.extractDenseLayerW(dw1,'dense1')
        exr.extractDenseLayerW(dw2, 'dense2')
        exr.extractConv1Result(intermediate_output,'conv1')
        exr.extractConv2Result(intermediate_output2, 'conv2')

        exr.extractMaxPoolResult(maxpool_output,'maxpool')
        exr.extractdenseResult(dense_output,'dense1')
        exr.extractdenseResult(dense2_output, 'dense2')
        exr.extractFlattenResult(flatten_output, 'flatten')
        exr.extractTestXcnn(X_test)
        exr.extractTestY(Y_test)
    if model == 1:
        dw1 = conv_net.get_layer("dense1").get_weights()
        dw2 = conv_net.get_layer("dense2").get_weights()
        dw3 = conv_net.get_layer("dense3").get_weights()
        dw4 = conv_net.get_layer("dense4").get_weights()

        exr = extraCnnWeights.ExtraCnnWeights(folder)
        exr.extractDenseLayerW(dw1,'dense1')
        exr.extractDenseLayerW(dw2, 'dense2')
        exr.extractDenseLayerW(dw3,'dense3')
        exr.extractDenseLayerW(dw4, 'dense4')
        exr.extractTestXdense(X_test)
        exr.extractTestY(Y_test)
    if model == 2:
        #get outputs of each layer
        layer_name = 'conv1'
        intermediate_layer_model1 = Model(inputs=conv_net.input,
                                         outputs=conv_net.get_layer(layer_name).output)
        intermediate_output1 = intermediate_layer_model1.predict(X_test)

        layer_name = 'conv2'
        intermediate_layer_model = Model(inputs=conv_net.input,
                                         outputs=conv_net.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(X_test)

        layer_name = 'dense1'
        dense_layer_model = Model(inputs=conv_net.input,
                                  outputs=conv_net.get_layer(layer_name).output)
        dense_output = dense_layer_model.predict(X_test)

        #get weights
        lw1 = conv_net.get_layer("conv1").get_weights()
        lw2 = conv_net.get_layer("conv2").get_weights()
        dw1 = conv_net.get_layer("dense1").get_weights()
        dw2 = conv_net.get_layer("dense2").get_weights()
        dw3 = conv_net.get_layer("dense3").get_weights()
        dw4 = conv_net.get_layer("dense4").get_weights()

        exr = extraCnnWeights.ExtraCnnWeights(folder)
        exr.extractCnnLayerW(lw1, 'cnn1')
        exr.extractCnnLayerW(lw2, 'cnn2')
        exr.extractDenseLayerW(dw1, 'dense1')
        exr.extractDenseLayerW(dw2, 'dense2')
        exr.extractDenseLayerW(dw3, 'dense3')
        exr.extractDenseLayerW(dw4, 'dense4')
        exr.extractTestXcnn(X_test)
        exr.extractTestY(Y_test)
    if model == 3:
        lw1 = conv_net.get_layer("conv1").get_weights()
        lw2 = conv_net.get_layer("conv2").get_weights()
        dw1 = conv_net.get_layer("dense1").get_weights()
        dw2 = conv_net.get_layer("dense2").get_weights()

        exr = extraCnnWeights.ExtraCnnWeights(folder)
        exr.extractCnnLayerW(lw1, 'cnn1')
        exr.extractCnnLayerW(lw2, 'cnn2')
        exr.extractDenseLayerW(dw1, 'dense1')
        exr.extractDenseLayerW(dw2, 'dense2')
        exr.extractTestXcnn(X_test)
        exr.extractTestY(Y_test)
    if model == 4:
        dw1 = conv_net.get_layer("dense1").get_weights()
        dw2 = conv_net.get_layer("dense2").get_weights()

        exr = extraCnnWeights.ExtraCnnWeights(folder)
        exr.extractDenseLayerW(dw1,'dense1')
        exr.extractDenseLayerW(dw2, 'dense2')
        exr.extractTestXdense(X_test)
        exr.extractTestY(Y_test)


    print('done')
