'''
class ReadDataCNN
author: Xiaomin Wu

Loading data operations for 1DCNN
'''
import numpy as np
import pandas as pd
import csv

def readFiles(files):
    datas = None;
    for file in files:
        data = pd.read_csv(file);
        if datas is None:
            datas = data;
        else:
            datas = np.concatenate((datas,data),axis = 1);
    return datas.T;

def loadData(path):
    data = [];
   
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #i = 0;
        for row in spamreader:
            #if(i!=0):
            data.append(list(map(float,row[0].split(','))));
            #i =i+1;
        data = np.array(data);
                     
        #return data<.csv>[rows][columns]
        
        
    return data;


#data loader for TRACES_1004_1_1 and BEHAVIOR_1004_1_1 CSV
def loadXY(X_path,Y_path):
    X = loadData(X_path);
    Y = loadData(Y_path);
    X = X.T;
    Y = Y[:,2]; #use only # behaviral variable
    
    return X,Y;

def LoadSeries(PathAry):
    i = 0;
    data = [];
    while(i != len(PathAry)):
        trace = PathAry[i];
        behavior = PathAry[i+1];
        i = i +2;
        
        X,Y = loadXY(trace, behavior);
        Y = Y.reshape(Y.shape[0],1)
        data.append(np.concatenate((X,Y),axis = 1))       
    return data

def FileSeries(str1,s,e):
    pathAry =[];
    for i in range(s,e):
        pathAry.append(''.join(['TRACES_',str1,'_',str(i+1),'.csv']));
        pathAry.append(''.join(['BEHAVIOR_',str1,'_',str(i+1),'.csv']));

    data = LoadSeries(pathAry);
    return data;

def readData(files):
    data = [];
    for file in files:
        a = FileSeries(file[0], file[1],file[2]);
        data.append(a)
        
    oneData = None
    for aa in data:
        for aaa in aa:
            if oneData is None:
                oneData = aaa;
            else:
                oneData = np.concatenate((oneData,aaa),axis = 0);
    
    
    return oneData, len(oneData);

def separateTime(datas, timeStep):
    #data:(#,time#,dimensionData)
    #label:(#,label)
    #separate XY
    X = datas[:, 0:272]  #274 dimension
    Y = datas[:,273]
    
    Xgroup = [];
    Ygroup = [];
    i = 0;
    while (i < datas.shape[0]):
        Xgroup.append(X[i:i+timeStep])
        Ygroup.append(Y[i:i+timeStep])
        i = i + timeStep
    
    #select Y label
    Ydecision = np.sum(Ygroup,axis = 1);
    label = np.zeros((Ydecision.shape[0],1))
    bound = timeStep/2;
    label[Ydecision>=bound] = 1;
    
    return Xgroup,label;
    
    #separate data into timeSteps
    
    #select label in time step
    
    #group corresponding datas

#files = [['1004_1', 0,3],['1004_2', 0,3]];

#['1004_1', 0,3],
#['1004_2', 0,3],
#['1004_3', 0,2],
#['1004_3', 2,3],
#['1004_4', 0,3],
#['1004_5', 0,3]
def LoadXYForCNN(files, timeStep):    
    datas,_ = readData(files);
    X,Y = separateTime(datas, timeStep);
    return X, Y;

def Extract(X,Y):
    
    Sum = np.sum(Y);
    if(Sum >= Y.shape[0]):
        target = 0;
    else:
        target = 1;
    
    iterator = 0;
    ExtractX = [];
    otherX = [];
    ExtractY = [];
    otherY = [];
    while (iterator != len(Y)):
        if(Y[iterator] == target):
            ExtractX.append(X[iterator])
            ExtractY.append(Y[iterator])
        else:
            otherX.append(X[iterator])
            otherY.append(Y[iterator])
        iterator+=1
    num = len(ExtractX);
    otherX = np.array(otherX)
    otherY = np.array(otherY)
    mask = np.random.choice(otherX.shape[0], num);
    oppositeX = otherX[mask];
    oppositeY = otherY[mask];
    
    reX = np.concatenate((ExtractX,oppositeX),axis = 0);
    reY = np.concatenate((ExtractY,oppositeY),axis = 0);
    CforShuffle = [];
    for i  in range(reX.shape[0]):
        CforShuffle.append([reX[i],reY[i]])
        
    for i in range(5):
        np.random.shuffle(CforShuffle);
    
    re = np.array(CforShuffle);
    reX = re[:,0];
    reY = re[:,1];
    return reX, reY;

def balanceDataCNN(files,timeStep):
    X,Y = LoadXYForCNN(files,timeStep)
    X,Y = Extract(X, Y)

    newY = [];
    for each in Y:
        newY.append(each[0])
    Y = np.array(newY)
    return X, Y;