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
from weightCutor import weightCutorRandom, weightCutorWM, weightCutor_convRandom,weightCutor_CDRandom,weightCutor_CDWM,weightCutor_convWM
from keras.models import model_from_json
import numpy as np
from common import *
from tfRefresh import refreshTF,loadTrainedModel
import copy
import matplotlib.pyplot as plt
from modelPre import modelPre
from datasetPre import datasetPre
from training import training
from modelstats import modelstatsNonAproximat


def pruningSingleLayer(layerIndx, model, random, trainX, trainY, valX, valY):
    oldWeights = []
    type = []
    name = []
    layers = []
    loss = None
    for each in model.layers:
        if isinstance(each, (Conv2D, Dense)):
            oldWeights.append(each.get_weights())
            name.append(each.name)
            layers.append(each)
            if isinstance(each, Conv2D):
                type.append('conv')
            else:
                type.append('dense')

    if type[layerIndx] == type[layerIndx + 1]: #has problem with a final layer, out of index range problem
        if type[layerIndx] == 'dense':
            convOrDense = 'dense'
        else:
            convOrDense = 'conv'
    else:
        convOrDense = 'CD'


    if 'conv' in type:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'

    if random:
        if convOrDense == 'dense':
            oldWeights[layerIndx], oldWeights[layerIndx + 1], amount = weightCutorRandom(oldWeights[layerIndx],
                                                                                         oldWeights[layerIndx + 1])
        elif convOrDense == 'conv':
            oldWeights[layerIndx], oldWeights[layerIndx + 1], amount = weightCutor_convRandom(oldWeights[layerIndx],
                                                                                              oldWeights[layerIndx + 1])
        elif convOrDense == 'CD':
            flattenSize = model.get_layer(name[layerIndx]).output_shape[1]
            flattenSize = flattenSize * flattenSize
            oldWeights[layerIndx], oldWeights[layerIndx + 1], amount = weightCutor_CDRandom(oldWeights[layerIndx],
                                                                                            oldWeights[layerIndx + 1],
                                                                                            flattenSize)

    else:
        if convOrDense == 'dense':
            oldWeights[layerIndx], oldWeights[layerIndx + 1], amount = weightCutorWM(oldWeights[layerIndx],
                                                                                     oldWeights[layerIndx + 1])
        elif convOrDense == 'conv':
            oldWeights[layerIndx], oldWeights[layerIndx + 1], amount = weightCutor_convWM(oldWeights[layerIndx],
                                                                                          oldWeights[layerIndx + 1])
        elif convOrDense == 'CD':
            flattenSize = model.get_layer(name[layerIndx]).output_shape[1]
            flattenSize = flattenSize * flattenSize
            oldWeights[layerIndx], oldWeights[layerIndx + 1], amount = weightCutor_CDWM(oldWeights[layerIndx],
                                                                                        oldWeights[layerIndx + 1],
                                                                                        flattenSize)

    # model place
    if convOrDense == 'dense':
        layers[layerIndx].units = amount

    else:
        layers[layerIndx].filters = amount
    model2 = model_from_json(model.to_json())
    model2.compile(optimizer='adam',
                   loss=loss,
                   metrics=['accuracy'])

    indx = 0
    for each in model2.layers:
        if isinstance(each, (Conv2D, Dense)):
            each.set_weights(oldWeights[indx])
            indx = indx + 1

    # retrain
    print("retrain:")
    model2.fit(trainX, trainY, batch_size=32, epochs=100, shuffle=False,verbose=0)

    scores = model2.evaluate(valX, valY, verbose=0)
    acc_val = scores[1]

    return model2, acc_val, amount

def plotTrend(testacchis,valacchis,shapeString,file_name,model_name,type):
    # plot greedy
    total_iterations = len(testacchis)
    xAxis = np.arange(total_iterations)
    labels = shapeString
    plt.figure(figsize=(18, 6))
    plt.plot(xAxis, testacchis, 'r--x', label='test_acc')
    plt.plot(xAxis, valacchis, 'g--x', label='val_acc')
    plt.xticks(xAxis, labels, rotation='vertical')
    plt.title(model_name+' Model Greedy Pruning Accuracy History on '+file_name)
    plt.xlabel(model_name+' hidden layer structures')
    plt.ylabel('Accuracy (X100 %)')
    plt.ylim(bottom=0.5, top=1)
    plt.yticks(np.arange(0.5, 1.05, 0.05))

    avgvalAcc_g = avgValAcc(valacchis)

    plt.annotate('average val_acc:' + str(avgvalAcc_g),
                 xy=(0, 0.6))

    plt.annotate('original_test_acc:\n' + str(np.round(testacchis[0], 3)) +
                 '\noriginal_val_acc\n' + str(np.round(valacchis[0], 3)),
                 xy=(0, testacchis[0] - 0.15))
    plt.annotate('final_test_acc:\n' + str(np.round(testacchis[total_iterations - 1], 3)) +
                 '\nfinal_val_acc:\n' + str(np.round(valacchis[total_iterations - 1], 3)),
                 xy=((total_iterations / 7) * 6, testacchis[total_iterations - 1] - 0.15))
    plt.tight_layout()
    plt.legend()
    # plt.show();
    if type == 'rrs':
        plt.savefig('../../outputs/plots/' + model_name + file_name + 'Random' + '.png')
    elif type == 'grs':
        plt.savefig('../../outputs/plots/' + model_name+file_name + 'Greedy' + '.png')
    elif type == 'nwm':
        plt.savefig('../../outputs/plots/' + model_name+file_name + 'NatrualWM' + '.png')
    else:
        plt.savefig('../../outputs/plots/' + model_name+file_name + 'unspecifiedMethod' + '.png')

class grs(object):
    def __init__(self,filename,smallestUnit,type,plot=True):
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
        self.modelType = None
        self.saveName = None
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
        if isinstance(self.model.layers[0], Conv2D):
            self.modelType = 'cnn'
        else:
            self.modelType = 'mlp'

    def run(self):
        if self.type == 'rrs':
            self.saveName = 'RRS'
        else:
            self.saveName = 'GRS'

        random = True
        self.modelIdentify()
        self.shapeString.append('X'.join(np.array(self.originShape).astype(str)))
        chosenP = self.originalACC_val
        # greedy
        while (chosenP / self.originalACC_val > self.tolerance):
            #refresh tensorflow graph
            self.model = refreshTF(self.model,self.model_name,self.file_name)

            tmpModels = []
            for i in range(len(self.originShape)):
                tmpModels.append(copy.deepcopy(self.model))

            tmpShape = copy.deepcopy(self.shape)
            Ps = []
            done = True
            for i in range(len(self.originShape)-1):
                if self.shape[i] > self.minShape[i]:
                    tmpModels[i], P1, tmpShape[i] = pruningSingleLayer(i, tmpModels[i], random,self.trainX,
                                                                            self.trainY,self.valX,self.valY)
                    Ps.append(P1)
                    done = False
                else:
                    Ps.append(0)

            #reach min shape
            if done:
                break

            if self.type == 'rrs':
                determine = np.max(Ps)
                if (determine != 0):
                    redo = 1
                    while (redo):
                        choice = np.random.choice(len(Ps), 1)
                        if (Ps[choice[0]] != 0):
                            redo = 0
                chosenP = Ps[choice[0]]
                positionP = choice[0]
            else:
                chosenP = np.max(Ps)
                positionP = np.argmax(Ps)

            if chosenP != 0 and chosenP / self.originalACC_val > self.tolerance:
                self.model = tmpModels[positionP]
                self.shape[positionP] = tmpShape[positionP]
                #DORate = DORate * dropoutDecay
                scores = self.model.evaluate(self.testX, self.testY, verbose=0)
                acc_test = scores[1]
                self.testacchis.append(acc_test)
                self.valacchis.append(chosenP)
                self.shapeString.append('X'.join(np.array(self.shape).astype(str)))
                print("pruned shape:", self.shapeString[len(self.shapeString) - 1])

        #save GRS results
        Fr = open('../../outputs/modelinfo/' + self.model_name + self.file_name + 'ModelSum_'+self.saveName+'', 'w')
        Fr.write('Structured model after '+self.saveName+' pruning:\n')
        self.model.summary(print_fn=lambda x: Fr.write(x + '\n'))
        Fr.close()

        # save models
        model_json = self.model.to_json()
        with open('../../outputs/modelinfo/' + self.model_name + self.file_name + 'model_'+self.saveName+'_pruned.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights('../../outputs/modelinfo/' + self.model_name + self.file_name + 'model_'+self.saveName+'_weights.h5')
        print("Saved greedy model to disk")

        #plot
        if self.plot:
            plotTrend(self.testacchis,self.valacchis,self.shapeString,self.file_name,self.model_name,self.type)

        self.paras, self.flops = modelstatsNonAproximat('../../outputs/modelinfo/' + self.model_name + self.file_name + 'model_'+self.saveName+'_pruned.json',
                                                        self.modelType)




if __name__ == "__main__":
    dataPre = datasetPre(['v','04e1'])
    modelPre = modelPre(dataPre.D_Row,dataPre.D_Column)
    training = training(dataPre.file_name,dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                        dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,dataPre.cnntrainX,dataPre.cnntrainY,
                        dataPre.cnnvalX,dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,savemodel = True)
    training.load(modelPre.model_name[2])
    training.run()
    mlporcnn = training.cnnormlp


    grs = grs(dataPre.file_name,2,'rrs',True)

    if mlporcnn == 'mlp':
        grs.load(training.model_name,
                  dataPre.mlptrainX, dataPre.mlptrainY, dataPre.mlpvalX,dataPre.mlpvalY, dataPre.mlptestX,
                 dataPre.mlptestY,training.valacc,training.testacc)
    else:
        grs.load(training.model_name,
                  dataPre.cnntrainX, dataPre.cnntrainY,dataPre.cnnvalX, dataPre.cnnvalY, dataPre.cnntestX,
                 dataPre.cnntestY,training.valacc, training.testacc)

    grs.run()
    print(grs.shapeString)
    print(grs.testacchis)
    print(grs.valacchis)
    print(training.paras)
    print(grs.paras)
    print(grs.flops)
