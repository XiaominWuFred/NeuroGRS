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
import numpy as np
import copy
from weightCutor import weightShare,weightShareconv
from common import *
from modelPre import modelPre
from datasetPre import datasetPre
from training import training
from tfRefresh import saveWeights,loadWeights,loadTModel


def shareWeight(model, layerName, digi):
    lw = model.get_layer(layerName).get_weights()
    ori = np.unique(lw[0])

    uamt0 = np.unique(lw[0])
    uamt1 = np.unique(lw[1])
    unori = np.concatenate((uamt0, uamt1))
    ori = np.unique(unori)

    oriWeightAmount = ori.shape[0]
    newW, unique = weightShare(lw, digi)
    #print(unique)
    model.get_layer(layerName).set_weights(newW)
    weightAmount = unique.shape[0]
    return model, weightAmount, oriWeightAmount


def shareWeightconv(model, layerName, digi):
    lw = model.get_layer(layerName).get_weights()
    ori = np.unique(lw[0])

    uamt0 = np.unique(lw[0])
    uamt1 = np.unique(lw[1])
    unori = np.concatenate((uamt0, uamt1))
    ori = np.unique(unori)

    oriWeightAmount = ori.shape[0]
    newW, unique = weightShareconv(lw, digi)
    #print(unique)
    model.get_layer(layerName).set_weights(newW)
    weightAmount = unique.shape[0]
    return model, weightAmount, oriWeightAmount

class q(object):
    def __init__(self,filename,record=True):
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
        self.digi = 4
        self.name = []
        self.layers = []
        self.originShape = []
        self.minShape = []
        self.shape = []
        self.shapeString = []
        self.tolerance = weightShareTH
        self.record = record
        self.loss = None
        #outputs:
        self.finalacctest = None
        self.finalaccval = None
        self.originalParas = None
        self.finalParas = None

    def load(self,model,model_name,valX,valY,testX,testY,originalACC_val,originalACC_test):
        self.model_name = model_name
        self.model = loadTModel(model_name,self.file_name)
        self.originalACC_val = originalACC_val
        self.originalACC_test = originalACC_test
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
        else:
            self.loss = 'sparse_categorical_crossentropy'


    def run(self):
        self.modelIdentify()

        #recompile model
        self.model.compile(optimizer='adam',
                           loss=self.loss,
                           metrics=['accuracy'])

        if self.record:
            F = open('../../outputs/runInfo/'+self.model_name + self.file_name + '_weightsharing.txt', 'w')

        acc_val = self.originalACC_val
        acc_test = self.originalACC_test
        while acc_val/self.originalACC_val > self.tolerance and self.digi > 0:
            #savedModel = copy.deepcopy(self.model)
            saveWeights(self.model)
            savedValACC = acc_val
            savedTestACC = acc_test
            for i in range(len(self.layers)):
                if isinstance(self.layers[i], Dense):
                    self.model, weightAmount, oriWA = shareWeight(self.model, self.name[i], self.digi)
                else:
                    self.model, weightAmount, oriWA = shareWeightconv(self.model, self.name[i], self.digi)

                scores = self.model.evaluate(self.testX, self.testY, verbose=0)
                acc_test = scores[1]
                score = self.model.evaluate(self.valX, self.valY, verbose=0)
                acc_val = score[1]

                if acc_val/self.originalACC_val > self.tolerance:
                    #savedModel = copy.deepcopy(self.model)
                    saveWeights(self.model)
                    savedValACC = acc_val
                    savedTestACC = acc_test
                    if self.record:
                        F.write("\n"+self.name[i])
                        F.write("\noriginal test acc : " + str(self.originalACC_test))
                        F.write("\noriginal val acc : " + str(self.originalACC_val))
                        F.write("\nafter test acc: " + str(acc_test))
                        F.write("\nafter val acc: " + str(acc_val))
                        F.write("\nweights parameters before: " + str(oriWA))
                        F.write("\nshared weights para amount: " + str(weightAmount))
                        F.write("\nweights digits: "+str(self.digi))
                else:
                    break

            self.digi = self.digi - 1

        #self.model = savedModel
        loadWeights(self.model)
        self.finalaccval = savedValACC
        self.finalacctest = savedTestACC
        if self.digi == -1:
            self.digi = 0
        else:
            self.digi = self.digi + 2

        # total paras:
        totalParas = []
        for i in range(len(self.layers)):
            w = self.model.get_layer(self.name[i]).get_weights()
            for each in w:
                totalParas = np.append(totalParas,each)
        self.originalParas = totalParas.shape[0]
        uniqueW = np.unique(totalParas)
        self.finalParas = uniqueW.shape[0]
        F.write("\nFinal parameters' digits: " + str(self.digi))
        F.write("\nFinal shared weights para amount: " + str(self.finalParas))
        F.close()

        # save models
        model_json = self.model.to_json()
        with open('../../outputs/modelinfo/' + self.model_name + self.file_name + "model_TQ_pruned.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights('../../outputs/modelinfo/' + self.model_name + self.file_name + "model_TQ_weights.h5")
        print("Saved TQ model to disk")

if __name__ == "__main__":
    dataPre = datasetPre('v','04e1')
    modelPre = modelPre(dataPre.D,dataPre.D1)
    training = training(modelPre.models[2],modelPre.model_name[2],dataPre.file_name,dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                        dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,dataPre.cnntrainX,dataPre.cnntrainY,
                        dataPre.cnnvalX,dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,savemodel = True)
    mlporcnn = training.run()
    if mlporcnn == 'mlp':
        q = q(modelPre.model_name[2],dataPre.file_name,modelPre.models[2],training.valacc,training.testacc,
                  dataPre.mlpvalX,dataPre.mlpvalY, dataPre.mlptestX, dataPre.mlptestY)
    else:
        q = q(modelPre.model_name[2], dataPre.file_name, modelPre.models[2], training.valacc, training.testacc,
                  dataPre.cnnvalX, dataPre.cnnvalY, dataPre.cnntestX, dataPre.cnntestY)

    q.run()
    print(q.originalParas)
    print(q.finalParas)
    print(q.finalacctest)
