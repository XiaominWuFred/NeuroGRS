'''
 author: xiaomin wu
 date: 1/16/2020
 '''
import pandas as pd;
import csv;
import numpy as np
#import cv2 as c
import matplotlib.pyplot as plt

def getLocation(path):
    data = [];
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #i = 0;
        for row in spamreader:
            data.append(list(map(float,row[0].split(','))));       
        data = np.array(data);
                        
    return data;

def matchX(X,location):
    
    newX = [];
    for eachX in X:
        ary = np.zeros((308,331))
        for i in range(len(location)):
            roundL = np.round(location[i])
            ary[int(roundL[0])-1][int(roundL[1])-1] = eachX[i]
        
        newX.append(ary)
    
    newX = np.array(newX)
    newX = newX.reshape((X.shape[0],308,331,1))
    return newX

'''
#test
X = np.full((2,273),255)

location = getLocation('CELLXY_1004.csv')
nX = matchX(X, location)
plt.imshow(nX[0])
plt.show()
'''