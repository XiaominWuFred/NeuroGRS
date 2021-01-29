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

from keras.layers import Dense,Conv2D
from weightCutor import weightCutorIn,weightCutorInconv
from common import *
from training import training
from modelPre import modelPre
from datasetPre import datasetPre
from modelstats import modelstatsNonAproximat
import numpy as np
import copy
from tfRefresh import loadGRSModel,saveWeights,loadWeights


def pruningWeightChannel(model, layerName,thrs):
    lw = model.get_layer(layerName).get_weights()
    newW,deleteWN,totalWN,remainWN = weightCutorIn(lw, thrs)
    model.get_layer(layerName).set_weights(newW)
    return model,deleteWN,totalWN,remainWN


def pruningWeightChannelconv(model, layerName,thrs):
    lw = model.get_layer(layerName).get_weights()
    newW,deleteWN,totalWN,remainWN = weightCutorInconv(lw, thrs)
    model.get_layer(layerName).set_weights(newW)
    return model,deleteWN,totalWN,remainWN


class t(object):
    def __init__(self,filename,thrs=0.001,minsize=0.001,record=True):
        #inputs
        self.model_name = None
        self.file_name = filename
        self.model = None
        self.originalACC_val = None
        self.originalACC_test = None
        self.valX = None
        self.valY = None
        self.testX = None
        self.testY = None
        #variables:
        self.name = []
        self.layers = []
        self.originShape = []
        self.minShape = []
        self.shape = []
        self.shapeString = []
        self.tolerance = weightPruneTH
        self.thrs = thrs
        self.minsize = minsize
        self.record = record
        self.loss = None
        self.modelType = None
        #outputs:
        self.finalacctest = None
        self.finalaccval = None
        self.totalDelete = None
        self.paras = None
        self.flops = None

    def load(self,model,model_name,valX,valY,testX,testY,originalACC_val,originalACC_test):
        self.model_name = model_name
        self.model = loadGRSModel(model_name,self.file_name)
        self.originalACC_val = originalACC_val[len(originalACC_val)-1]
        self.originalACC_test = originalACC_test[len(originalACC_test)-1]
        self.valX = valX
        self.valY = valY
        self.testX = testX
        self.testY = testY
        #reset
        self.name = []
        self.layers = []
        self.originShape = []
        self.minShape = []
        self.shape = []
        self.shapeString = []

    def modelIdentify(self):
        for each in self.model.layers:
            if isinstance(each, (Conv2D, Dense)):
                self.layers.append(each)
                self.name.append(each.name)
        if isinstance(self.layers[0], Conv2D):
            self.loss = 'binary_crossentropy'
            self.modelType = 'cnn'
        else:
            self.loss = 'sparse_categorical_crossentropy'
            self.modelType = 'mlp'

    def run(self):
        self.modelIdentify()
        if self.record:
            F = open('../../outputs/runInfo/' + self.model_name+self.file_name + 'ChannelCut.txt', 'w')

        DN = np.zeros(len(self.layers))
        empty = 1
        acc_val = self.originalACC_val
        acc_test = self.originalACC_test

        #recompile model
        self.model.compile(optimizer='adam',
                           loss=self.loss,
                           metrics=['accuracy'])

        while (acc_val / self.originalACC_val) >= self.tolerance and empty != 0:
            #savedModel = copy.deepcopy(self.model)
            saveWeights(self.model)
            savedValACC = acc_val
            savedTestACC = acc_test

            empty = 0
            for i in range(len(self.layers)):
                if isinstance(self.layers[i],Dense):
                    self.model, DN[i], TN, RN = pruningWeightChannel(self.model, self.name[i], self.thrs)
                else:
                    self.model, DN[i], TN, RN = pruningWeightChannelconv(self.model, self.name[i], self.thrs)
                empty = RN | empty
                scores = self.model.evaluate(self.testX, self.testY, verbose=0)
                acc_test = scores[1]
                score = self.model.evaluate(self.valX, self.valY, verbose=0)
                acc_val = score[1]
                empty = empty | RN

                if (acc_val / self.originalACC_val) >= self.tolerance and empty != 0:
                    #savedModel = copy.deepcopy(self.model)
                    saveWeights(self.model)
                    savedValACC = acc_val
                    savedTestACC = acc_test
                    if self.record:
                        F.write('\nweight cancel acc: ' + str(acc_test) +
                                '\n val acc' + str(acc_val) +
                                '\n layer '+str(i)+' deleted weight #' + str(DN[i]) +
                                '\n layer '+str(i)+'  total weight #' + str(TN) +
                                '\n layer '+str(i)+'  remain weight #' + str(RN))
                else:
                    break
            self.thrs = self.thrs + self.minsize

        #self.model = savedModel
        loadWeights(self.model)
        self.finalaccval = savedValACC
        self.finalacctest = savedTestACC
        self.totalDelete = np.sum(DN)

        # save models
        model_json = self.model.to_json()
        with open('../../outputs/modelinfo/' + self.model_name + self.file_name + "model_T_pruned.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights('../../outputs/modelinfo/' + self.model_name + self.file_name + "model_T_weights.h5")

        if self.record:
            F.write("\noriginal val acc: " + str(self.originalACC_val))
            F.write("\nlast val acc: " + str(self.finalaccval))
            F.write("\noriginal test acc: " + str(self.originalACC_test))
            F.write("\nlast test acc: " + str(self.finalacctest))
            F.close()


        self.paras, self.flops = modelstatsNonAproximat('../../outputs/modelinfo/' + self.model_name + self.file_name + "model_GRS_pruned.json",
                                                        self.modelType)
        self.paras = self.paras - self.totalDelete
        self.flops = self.flops - 2*self.totalDelete

if __name__ == "__main__":
    dataPre = datasetPre('v','04e1')
    modelPre = modelPre(dataPre.D,dataPre.D1)
    training = training(modelPre.models[2],modelPre.model_name[2],dataPre.file_name,dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                        dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,dataPre.cnntrainX,dataPre.cnntrainY,
                        dataPre.cnnvalX,dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,savemodel = True)
    mlporcnn = training.run()
    if mlporcnn == 'mlp':
        t = t(modelPre.model_name[2],dataPre.file_name,modelPre.models[2],training.valacc,training.testacc,
                  dataPre.mlpvalX,dataPre.mlpvalY, dataPre.mlptestX, dataPre.mlptestY)
    else:
        t = t(modelPre.model_name[2], dataPre.file_name, modelPre.models[2], training.valacc, training.testacc,
                  dataPre.cnnvalX, dataPre.cnnvalY, dataPre.cnntestX, dataPre.cnntestY)

    t.run()
    print(t.finalacctest)
    print(t.finalaccval)
    print(t.totalDelete)
    print(t.paras)
    print(t.flops)
