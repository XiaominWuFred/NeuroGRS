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

import numpy as np
from DataLoader import DataLoader as DL
from compactImg import combined
import sys
sys.path.append("../../graphGen/util")
from extraCnnWeights import ExtraCnnWeights
import os
import copy


class datasetPre(object):
    #inputs:
    #v: version to diff each run
    #set: for choosing dataset in our application case
    def __init__(self,args=None):
        #required variables:
        self.mlptrainX = None
        self.mlptrainY = None
        self.mlpvalX = None
        self.mlpvalY = None
        self.mlptestX = None
        self.mlptestY = None
        self.cnntrainX = None
        self.cnntrainY = None
        self.cnnvalX = None
        self.cnnvalY = None
        self.cnntestX = None
        self.cnntestY = None
        self.file_name = None
        self.D_Row = None
        self.D_Column = None

        #custom variable according to needs
        self.set = args[1]
        self.ds = None
        self.version = args[0]
        self.N = None
        self.C = None

        #needed instruction
        self.run()
        self.testDatasetSave()

    def run(self):
        if (self.set == ('04e1')):
            #print("chosen 04e1\n")
            files = [['1004_1', 0, 1, 1], ]
            self.file_name = '_1004_e1_' + self.version
            CNNDim = 273
            self.ds = 'm04e1'
        if (self.set == ('04e2')):
            files = [['1004_1', 1, 2, 1], ]
            self.file_name = '_1004_e2_' + self.version
            CNNDim = 273
            self.ds = 'm04e2'
        if (self.set == ('04e3')):
            files = [['1004_1', 2, 3, 1], ]
            self.file_name = '_1004_e3_' + self.version
            CNNDim = 273
            self.ds = 'm04e3'
        if (self.set == ('05e1')):
            files = [['1005_1', 0, 1, 1], ]
            self.file_name = '_1005_e1_' + self.version
            CNNDim = 140
            self.ds = 'm05e1'
        if (self.set == ('05e2')):
            files = [['1005_1', 1, 2, 1], ]
            self.file_name = '_1005_e2_' + self.version
            CNNDim = 140
            self.ds = 'm05e2'
        if (self.set == ('05e3')):
            files = [['1005_1', 2, 3, 1], ]
            self.file_name = '_1005_e3_' + self.version
            CNNDim = 140
            self.ds = 'm05e3'
        if (self.set == ('06e1')):
            files = [['1006_1', 0, 1, 1], ]
            self.file_name = '_1006_e1_' + self.version
            CNNDim = 114
            self.ds = 'm06e1'
        if (self.set == ('06e2')):
            files = [['1006_1', 1, 2, 1], ]
            self.file_name = '_1006_e2_' + self.version
            CNNDim = 114
            self.ds = 'm06e2'
        if (self.set == ('06e3')):
            files = [['1006_1', 2, 3, 1], ]
            self.file_name = '_1006_e3_' + self.version
            CNNDim = 114
            self.ds = 'm06e3'
        if (self.set == ('simu')):
            files = [['1000_0', 0, 0, 1], ]
            self.file_name = '_SIMU_' + self.version
            CNNDim = 100
            self.ds = 'simu'

        day = {}
        day['1004_1'] = 'Mouse 1004 Day1'
        day['1004_2'] = 'Mouse 1004 Day2'
        day['1004_3'] = 'Mouse 1004 Day3'
        day['1004_4'] = 'Mouse 1004 Day4'
        day['1004_5'] = 'Mouse 1004 Day5'
        day['1005_1'] = 'Mouse 1005 Day1'
        day['1005_2'] = 'Mouse 1005 Day2'
        day['1005_3'] = 'Mouse 1005 Day3'
        day['1005_4'] = 'Mouse 1005 Day4'
        day['1005_5'] = 'Mouse 1005 Day5'
        day['1006_1'] = 'Mouse 1006 Day1'
        day['1006_2'] = 'Mouse 1006 Day2'
        day['1006_3'] = 'Mouse 1006 Day3'
        day['1006_4'] = 'Mouse 1006 Day4'
        day['1006_5'] = 'Mouse 1006 Day5'
        day['1000_0'] = 'Mouse Simu'

        Experiment = {}
        Experiment['1'] = 'E1'
        Experiment['2'] = 'E2'
        Experiment['3'] = 'E3'
        Experiment['4'] = 'E4'
        Experiment['5'] = 'E5'

        testRatio = 1/3000#0.1
        loader = DL('../../dataset/',testRatio)
        i = 0
        dataSet = []
        while (i != len(files)):
            data, length = loader.extractBalanceDataShuffled(files[i][0], files[i][1], files[i][2], files[i][3])
            print('data length:', length)
            dataSet.append(data)
            i = i + 1

            re = dataSet[0]
        if (len(dataSet) == 1):
            #only one dataset
            pass
        else:
            #combine multiple datasets
            i = 1
            while (i != len(dataSet)):
                re = loader.combineTwo(re, dataSet[i])
                i = i + 1

        X, Y = loader.splitXY(re)
        X_test = copy.deepcopy(loader.testX)
        Y_test = copy.deepcopy(loader.testY)
        print("Test dataset Ratio: label 1 : label 0 == "+str(int(np.sum(Y_test)))+" : "
              +str(len(Y_test)-int(np.sum(Y_test))))


        if (CNNDim == 273):
            Xconv = combined(CNNDim, 'CELLXY_1004.csv','../../dataset/', X)
            Xconv_test = combined(CNNDim, 'CELLXY_1004.csv','../../dataset/', X_test)
        if (CNNDim == 114):
            Xconv = combined(CNNDim, 'CELLXY_1006.csv', '../../dataset/',X)
            Xconv_test = combined(CNNDim, 'CELLXY_1006.csv', '../../dataset/',X_test)
        if (CNNDim == 140):
            Xconv = combined(CNNDim, 'CELLXY_1005.csv', '../../dataset/',X)
            Xconv_test = combined(CNNDim, 'CELLXY_1005.csv', '../../dataset/',X_test)
        if (CNNDim == 100):
            Xconv = combined(CNNDim, 'CELLXY_1000.csv', '../../dataset/', X)
            Xconv_test = combined(CNNDim, 'CELLXY_1000.csv', '../../dataset/', X_test)

        Xconv = Xconv.reshape((Xconv.shape[0], Xconv.shape[1], Xconv.shape[2], 1))
        Xconv_test = Xconv_test.reshape((Xconv_test.shape[0], Xconv_test.shape[1], Xconv_test.shape[2], 1))

        # data description:

        self.N = int(Xconv.shape[0])
        self.D_Row = int(Xconv.shape[1])
        self.D_Column = int(Xconv.shape[2])
        self.C = int(np.max(Y) + 1)
        print("data total length:", self.N, "data dimension:", self.D_Row, "data dimension2: ", self.D_Column, "data classes:", self.C)
        sampleSize = Xconv.shape[0]
        train = 0.8
        val = 0.2

        self.mlptrainX = X[np.arange(int(sampleSize * train)), :]
        self.mlpvalX = X[np.arange(int(sampleSize * train), int(sampleSize * (train + val)))]
        self.mlptestX = loader.testX
        self.mlptrainY = Y[np.arange(int(sampleSize * train))]
        self.mlpvalY = Y[np.arange(int(sampleSize * train), int(sampleSize * (train + val)))]
        self.mlptestY = loader.testY

        self.cnntrainX = Xconv[np.arange(int(sampleSize * train)), :]
        self.cnnvalX = Xconv[np.arange(int(sampleSize * train), int(sampleSize * (train + val)))]
        self.cnntestX = Xconv_test
        Y_train = copy.deepcopy(Y[np.arange(int(sampleSize * train))])
        Y_val =  copy.deepcopy(Y[np.arange(int(sampleSize * train), int(sampleSize * (train + val)))])

        self.cnntrainY = np.zeros((Y_train.shape[0], 2))
        self.cnntrainY[np.arange(Y_train.shape[0]), Y_train.astype(int)] = 1

        self.cnnvalY = np.zeros((Y_val.shape[0], 2))
        self.cnnvalY[np.arange(Y_val.shape[0]), Y_val.astype(int)] = 1

        self.cnntestY = np.zeros((Y_test.shape[0],2))
        self.cnntestY[np.arange(Y_test.shape[0]),Y_test.astype(int)] = 1

#fixed function:
    def testDatasetSave(self):
        folder = '../graphGen/extractedParas'
        os.system("cd ../"+folder+"\nrm *")
        exr = ExtraCnnWeights(folder)
        exr.extractTestXcnn(self.cnntestX)
        exr.extractTestXdense(self.mlptestX)
        exr.extractTestY(self.mlptestY)

        #save sample size:



if __name__ == "__main__":
    dataPre = datasetPre([sys.argv[1],sys.argv[2]])
    print('Data extraction done for: '+sys.argv[1]+sys.argv[2])