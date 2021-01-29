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

from keras.layers import Conv2D, Dense
from modelPre import modelPre
from datasetPre import datasetPre
from modelstats import modelstatsNonAproximat
from tfRefresh import *


class training(object):
    def __init__(self,filename,mlptrainX,mlptrainY,mlpvalX,mlpvalY,mlptestX,mlptestY,cnntrainX,cnntrainY,cnnvalX,cnnvalY,cnntestX,cnntestY,savemodel = True):
        #inputs
        self.mlptrainX = mlptrainX
        self.mlptrainY =  mlptrainY
        self.mlpvalX = mlpvalX
        self.mlpvalY = mlpvalY
        self.mlptestX = mlptestX
        self.mlptestY = mlptestY
        self.cnntrainX = cnntrainX
        self.cnntrainY = cnntrainY
        self.cnnvalX = cnnvalX
        self.cnnvalY = cnnvalY
        self.cnntestX = cnntestX
        self.cnntestY = cnntestY
        self.model = None
        self.file_name = filename
        self.model_name = None
        #parameters
        self.cnnormlp = None
        self.savemodel = savemodel
        #outputs
        self.testacc = None
        self.valacc = None
        self.paras = None
        self.flops = None

        #self.run()

    def saveModel(self):
        Fr = open('../../outputs/modelinfo/' + self.model_name+self.file_name + 'ModelSum_training', 'w')
        Fr.write('model before pruning:\n')
        self.model.summary(print_fn=lambda x: Fr.write(x + '\n'))
        Fr.close()

        model_json = self.model.to_json()
        with open('../../outputs/modelinfo/' + self.model_name+self.file_name + "model_original.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights('../../outputs/modelinfo/' + self.model_name+self.file_name + "model_original_weights.h5")
        print("Saved original model to disk")

    def load(self,modelname):
        self.model = loadModel(modelname)
        self.model_name = modelname

    def run(self):
        if isinstance(self.model.layers[0], Conv2D):
            print("cnn model found")
            self.cnnormlp = 'cnn'
        else:
            print("mlp model found")
            self.cnnormlp = 'mlp'


        if self.cnnormlp == 'cnn':
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model.fit(self.cnntrainX, self.cnntrainY, batch_size=32, epochs=150, shuffle=True,verbose=0)
            scores = self.model.evaluate(self.cnnvalX, self.cnnvalY, verbose=0)
            self.valacc = scores[1]
            scores = self.model.evaluate(self.cnntestX, self.cnntestY, verbose=0)
            self.testacc = scores[1]
        else:
            self.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            self.model.fit(self.mlptrainX, self.mlptrainY, batch_size=32, epochs=150, shuffle=True,verbose=0)
            scores = self.model.evaluate(self.mlpvalX, self.mlpvalY, verbose=0)
            self.valacc = scores[1]
            scores = self.model.evaluate(self.mlptestX, self.mlptestY, verbose=0)
            self.testacc = scores[1]

        if self.savemodel == True:
            self.saveModel()

        self.paras,self.flops = modelstatsNonAproximat('../../outputs/modelinfo/' + self.model_name+self.file_name + "model_original.json",
                                                       self.cnnormlp)



if __name__ == "__main__":
    dataPre = datasetPre('v','04e1')
    modelPre = modelPre(dataPre.D,dataPre.D1)
    training = training(dataPre.file_name,dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                        dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,dataPre.cnntrainX,dataPre.cnntrainY,
                        dataPre.cnnvalX,dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,savemodel = True)
    training.load(modelPre.models[3],modelPre.model_name[3])
    training.run()
    print(training.testacc)
    print(training.valacc)
    print(training.paras)
    print(training.flops)

