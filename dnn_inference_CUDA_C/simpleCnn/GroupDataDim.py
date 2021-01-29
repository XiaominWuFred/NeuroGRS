'''
 author: xiaomin wu
 date: 1/16/2020
 '''
import numpy as np

def correlationScore(X1,X2):
    return np.corrcoef(X1, X2);
    
def slidingWindow(X, sliding = 5):
    
    newDataSet = [];
    newData = [];
    newGroup = None; #(sliding, subdimention)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1] - sliding):
            newGroup = X[i,np.arange(j,j+sliding)]
            newGroup = np.array(newGroup)
            newData.append(newGroup)
        
        newData = np.array(newData)
       
        newDataSet.append(newData)
        newData = [];
    
    newDataSet = np.array(newDataSet)    
    return newDataSet;