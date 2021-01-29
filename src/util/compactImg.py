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

import numpy as np;
from Location import getLocation

def findDim(featureNum):
    sqrt = np.sqrt(featureNum)
    sqrt = int(sqrt)
    if(np.square(sqrt) < featureNum):
        sqrt = sqrt+1
    dimAry = np.zeros((sqrt,sqrt))
    return dimAry

def reSortLocation(sumLoca,eachX):
    eachX = eachX.reshape((eachX.shape[0],1))
    arr = np.concatenate((sumLoca,eachX),axis = 1)
    arr = arr[arr[:,0].argsort()]
    sr = arr[:,1]
    return sr

def matchLR(featureNum,dimAry,x,y,sortedLoca, i):
    staticX = x
    staticY = y
    yi = 0
    xi = 0
    while(1):
        
        if(yi < y):
            if(i < featureNum):
                dimAry[staticX,yi] = sortedLoca[i]
            i = i + 1
            yi = yi + 1
        else:
            break
        
        if(xi < x):
            if(i < featureNum):
                dimAry[xi,staticY] = sortedLoca[i]
            i = i + 1
            xi = xi + 1
    if(i < featureNum):
        dimAry[staticX,staticY] = sortedLoca[i]
        i = i + 1
    return dimAry, i

def rMatch(featureNum,sortedLoca, dimAry):
    ary = np.zeros(dimAry.shape)
    i=0
    for j in range(dimAry.shape[0]):
        ary,i = matchLR(featureNum,ary, j, j, sortedLoca, i)
    return ary

def combined(featureNum,path,dir,X):
    dimAry = findDim(featureNum)
    location = getLocation(path,dir)
    sumLoca = np.sum(location,axis = 1)
    sumLoca = sumLoca.reshape((sumLoca.shape[0],1))
    newdata = []
    for eachX in X:
        sortedLoca = reSortLocation(sumLoca,eachX)
        newData2D = rMatch(featureNum,sortedLoca,dimAry)
        newdata.append(newData2D)
    newdata = np.array(newdata)
    return newdata
       
#test
'''
files = [
       ['1004_1', 0,1, 1],

         ];


train = 0.5;
val = 0.2;
test = 0.3;

#model select
select = 'compare'; #'SVM' 'NNKeras1' 'NNself' 'NNKeras2' 'LDA' 'CONV1D' 'CONV1D1' 'skSVM' 'compare' '2DCNN'
AverageTimes = 1; #for compare only
#load data (shuffled)
Metr = M();
loader = DL();
   
i = 0;
dataSet = [];
while (i!=len(files)):
    data,length = loader.extractBalanceDataShuffled(files[i][0],files[i][1],files[i][2],files[i][3]);
    print('data length:', length,'\n');
    dataSet.append(data);
    i = i +1;
    
    re = dataSet[0];
if(len(dataSet) == 1):
    pass;
else:
    i = 1;
    while (i != len(dataSet)):
        re = loader.combineTwo(re, dataSet[i]);
        i = i + 1;

X,Y = loader.splitXY(re);

newData2D = combined(273, 'CELLXY_1004.csv',X)
print(newData2D)
'''