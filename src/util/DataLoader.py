################################################################################
# @ddblock_begin copyright
# -------------------------------------------------------------------------
# Copyright (c) 2017-2020
# UMB-UMD Neuromodulation Research Group,
# University of Maryland at Baltimore, and 
# University of Maryland at College Park. 
# 
# All rights reserved.
# 
# IN NO EVENT SHALL THE UNIVERSITY OF MARYLAND BALTIMORE
# OR UNIVERSITY OF MARYLAND COLLEGE PARK BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
# ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
# THE UNIVERSITY OF MARYLAND HAS BEEN ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# 
# THE UNIVERSITY OF MARYLAND SPECIFICALLY DISCLAIMS ANY WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE
# PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
# MARYLAND HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.DE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.
# -------------------------------------------------------------------------

# @ddblock_end copyright
################################################################################

'''
class DataLoader
author: Xiaomin Wu
Functions:
__init__(self,path): the constructor, path will be the csv file path start from the directory of this program file
loadData(self): load data from csv file. return X,Y. X contain input data in shape (N,D). Y contain output label standard in shape (N,1). 
    here N means number of sample, D means dimentions. 
'''

import csv
import numpy as np

class DataLoader(object):
    
    def __init__(self,dir,testRatio=0.1):
        self.dir = dir
        self.SEED = 0
        self.testRatio = testRatio
        self.testX = None
        self.testY = None
        
    def loadData(self,path):
        data = []
       
        with open(self.dir+path, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            #i = 0
            for row in spamreader:
                #if(i!=0):
                data.append(list(map(float,row[0].split(','))))
                #i =i+1
            data = np.array(data)
                         
            #return data<.csv>[rows][columns]
            
            
        return data

    
    #data loader for TRACES_1004_1_1 and BEHAVIOR_1004_1_1 CSV
    def loadXY(self,X_path,Y_path):
        X = self.loadData(X_path)
        Y = self.loadData(Y_path)
        X = X.T
        Y = Y[:,2] #use only first behaviral variable
        Y = Y.reshape(len(Y), 1)
        print('label 1 amount:'+str(np.sum(Y)))
        print('label 0 amount:'+str(len(Y)-np.sum(Y)))

        xy = np.concatenate((X, Y), 1)
        np.random.seed(self.SEED)
        np.random.shuffle(xy)
        X,Y = self.splitXY(xy)
        self.testX = X[int((1-self.testRatio)*len(X)):len(X)]
        self.testY = Y[int((1-self.testRatio)*len(Y)):len(Y)]
        Xremain = X[0:int((1-self.testRatio)*len(X))]
        Yremain = Y[0:int((1-self.testRatio)*len(Y))]
        return Xremain,Yremain
    
    def extractOne(self,X,Y,label):
        N = X.shape[0]
        D = X.shape[1]
        ExtractResult = []
        Y = Y.reshape(N,1)
        xy = np.concatenate((X,Y), 1)
        
        for row in range(N):
            last = xy[row][D]
            if (last == label):
                ExtractResult.append(np.array(xy[row]))
        
        ExtractResult = np.array(ExtractResult)
        return ExtractResult, len(ExtractResult)
        
    def extractFiles(self,PathAry,label):
        i = 0
        extract = []
        while(i != len(PathAry)):
            trace = PathAry[i]
            behavior = PathAry[i+1]
            i = i +2
            #original 3000 data samples
            X,Y = self.loadXY(trace, behavior)
            exre,length = self.extractOne(X, Y,label)
            extract.append(exre)
            print("Extract ",length,"data with label:",label,"from file:",trace)
        
        result = None
        if(len(extract) - 1 == 0):
            result = extract[0]
        else:
            for j in range(len(extract) - 1):
                extract[j+1] = np.concatenate((extract[j],extract[j+1]),0)
                result = extract[j+1]
            
        return result, len(result)
    
    def extractFileSeries(self,str1,s,e,label):
        pathAry =[]
        if s == e:
            pathAry.append(''.join(['TRACES_1000_0_0.csv']))
            pathAry.append(''.join(['BEHAVIOR_1000_0_0.csv']))
        else:
            for i in range(s,e):
                pathAry.append(''.join(['TRACES_',str1,'_',str(i+1),'.csv']))
                pathAry.append(''.join(['BEHAVIOR_',str1,'_',str(i+1),'.csv']))

        a,b = self.extractFiles(pathAry,label)
        return a,b
    
    def extractBalanceData(self,str1,s,e,focus):
        a,mask = self.extractFileSeries(str1, s,e, focus)
        c = None
        if(focus == 1):
            c,_ = self.extractFileSeries(str1, s,e, 0)
        else:
            c,_ = self.extractFileSeries(str1, s,e, 1)
        np.random.seed(self.SEED)
        np.random.shuffle(c)
        b = c[0:mask]
        re = np.concatenate((a,b),0)
        
        return re, len(re)
    
    def extractBalanceDataShuffled(self,str1,s,e,focus):
        data,length = self.extractBalanceData(str1, s,e, focus)
        for i in range(10):
            np.random.seed(self.SEED)
            np.random.shuffle(data)
       
        return data,length
    
    def combineTwo(self,X,Y):
        re = np.concatenate((X,Y),0)
        
        return re
    
    def splitXY(self, XY):
        X = XY[:,np.arange(XY.shape[1] - 1)]
        Y = XY[:,XY.shape[1]-1]
        
        return X,Y
        
            
'''            
loader = DataLoader('999_training.csv')

X,Y = loader.loadData()

print(X,Y)
'''
