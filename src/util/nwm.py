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

from grs import pruningSingleLayer,plotTrend
from tfRefresh import refreshTF,loadTrainedModel
from keras.layers import Dense,Conv2D
import numpy as np
from common import *
import copy
from modelPre import modelPre
from datasetPre import datasetPre
from training import training
from modelstats import *

class nwm(object):
    def __init__(self,filename,smallestUnit,type='nwm',plot=True):
        #inputs
        self.model_name = None
        self.file_name = filename
        self.model = None
        self.originalACC_val = None
        self.originalACC_test = None
        self.min = smallestUnit
        self.trainX = None
        self.trainY = None
        self.valX = None
        self.valY = None
        self.testX = None
        self.testY = None
        #variables:
        self.type = type
        self.plot = plot
        self.originShape = []
        self.minShape = []
        self.shape = []
        self.shapeString = []
        self.tolerance = StructurePruneTH
        #outputs:
        self.testacchis = None
        self.valacchis = None
        self.paras = None
        self.flops = None

    def load(self,modelname,trainX,trainY,valX,valY,testX,testY,originalACC_val,originalACC_test):
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.testX = testX
        self.testY = testY
        self.originalACC_val = originalACC_val
        self.originalACC_test = originalACC_test
        self.model_name = modelname
        self.model = loadTrainedModel(self.model_name,self.file_name)
        self.testacchis = [originalACC_test]
        self.valacchis = [originalACC_val]
        #reset:
        self.originShape = []
        self.minShape = []
        self.shape = []
        self.shapeString = []

    def modelIdentify(self):
        for each in self.model.layers:
            if isinstance(each, Dense):
                self.originShape.append(each.units)
                self.shape.append(each.units)
                self.minShape.append(self.min)
            if isinstance(each, Conv2D):
                self.originShape.append(each.filters)
                self.shape.append(each.filters)
                self.minShape.append(self.min)

    def run(self):
        random = False
        self.modelIdentify()
        finetuningACC = self.originalACC_val
        self.shapeString.append('X'.join(np.array(self.originShape).astype(str)))
        for i in range(len(self.originShape)):

            while (finetuningACC / self.originalACC_val) > self.tolerance:
                if self.shape[i] > self.minShape[i]:
                    #refresh tf graph
                    self.model = refreshTF(self.model,self.model_name,self.file_name)
                    #DORate = DORate * dropoutDecay
                    print("pruning DENSE #"+str(i))
                    savedModel = copy.deepcopy(self.model)
                    savedVal = finetuningACC
                    self.model,finetuningACC,self.shape[i] = pruningSingleLayer(i,self.model, random,
                                                                                  self.trainX,self.trainY,
                                                                                  self.valX,self.valY)

                    if (finetuningACC / self.originalACC_val) > self.tolerance:
                        self.valacchis.append(finetuningACC)
                        self.shapeString.append('X'.join(np.array(self.shape).astype(str)))
                        print("pruned shape:", self.shapeString[len(self.shapeString) - 1])
                        score = self.model.evaluate(self.testX, self.testY, verbose=0)
                        print("Pruned Test set ACC: ", score[1])
                        self.testacchis.append(score[1])
                    else:
                        print("out of threshold")
                        self.model = savedModel
                        finetuningACC = savedVal
                        self.shape[i] = self.shape[i] + 1
                        break
                else:
                    break

        if self.plot:
            plotTrend(self.testacchis,self.valacchis,self.shapeString,self.file_name,self.model_name,self.type)

        #save Natrual results
        Fr = open('../../outputs/modelinfo/' + self.model_name + self.file_name + 'ModelSum_NWM', 'w')
        Fr.write('Structured model after NWM pruning:\n')
        self.model.summary(print_fn=lambda x: Fr.write(x + '\n'))
        Fr.close()

        # save models
        model_json = self.model.to_json()
        with open('../../outputs/modelinfo/' + self.model_name + self.file_name + "model_NWM_pruned.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights('../../outputs/modelinfo/' + self.model_name + self.file_name + "model_NWM_weights.h5")
        print("Saved nwm model to disk")

        self.paras, self.flops = modelstats(self.model)

        print('Neuron pruning done')


if __name__ == "__main__":
    dataPre = datasetPre(['v','04e1'])
    modelPre = modelPre(dataPre.D_Row,dataPre.D_Column)
    training = training(dataPre.file_name,dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                        dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,dataPre.cnntrainX,dataPre.cnntrainY,
                        dataPre.cnnvalX,dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,savemodel = True)
    training.load(modelPre.model_name[2])
    training.run()
    mlporcnn = training.cnnormlp


    grs = nwm(dataPre.file_name,2,'rrs',True)

    if mlporcnn == 'mlp':
        grs.load(training.model,training.model_name,
                  dataPre.mlptrainX, dataPre.mlptrainY, dataPre.mlpvalX,dataPre.mlpvalY, dataPre.mlptestX,
                 dataPre.mlptestY,training.valacc,training.testacc)
    else:
        grs.load(training.model,training.model_name,
                  dataPre.cnntrainX, dataPre.cnntrainY,dataPre.cnnvalX, dataPre.cnnvalY, dataPre.cnntestX,
                 dataPre.cnntestY,training.valacc, training.testacc)

    grs.run()
    print(grs.shapeString)
    print(grs.testacchis)
    print(grs.valacchis)
    print(training.paras)
    print(grs.paras)
    print(grs.flops)