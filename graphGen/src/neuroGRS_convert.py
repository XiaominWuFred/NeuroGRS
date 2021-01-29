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

import sys
sys.path.append("../util")
from keras.layers import Dense,Conv2D,MaxPooling2D
from neuroGRS_loadPrunedModel import neuroGRS_loadPrunedModel
from neuroGRS_paraGen import neuroGRS_paraGen
from neuroGRS_graphGen import neuroGRS_graphGen
import csv
import os

class neuroGRS_convert:
    def __init__(self,modelname,filename,set,path='../../outputs/modelinfo/',Pruned=0):
        self.modelname = modelname # for both
        self.filename = filename
        self.path = path
        self.model = None

        #variables for model
        self.shape = [] # for .c file
        self.rawLayers = [] # for both
        self.DaCLayers = [] # for both
        self.DaCName = [] # for both
        self.layersAmount = None # for both
        self.type = None # for both
        self.D_Row = None # for .c file
        self.D_Column = None # for .c file
        self.sampleSize = None # for .c file
        self.sampleDim = None  # for .c file
        version = 'version'
        file = set
        os.system('cd ../../src/util/\npython3 datasetPre.py '+version+' '+file)

        self.convert(Pruned=Pruned)
        self.modelIdentify()
        self.paraGen = neuroGRS_paraGen(self.DaCLayers,
                                        self.model)
        #TODO:
        self.graphGen = neuroGRS_graphGen(self.modelname,
                                          self.rawLayers,
                                          self.DaCLayers,
                                          self.DaCName,
                                          self.type,
                                          self.layersAmount,
                                          self.sampleSize,
                                          self.sampleDim)
        self.graphGen.writeHeader(path='../../dnn_inference_CUDA_C/lide_c_dnnlayers/autoGenGraph/')
        self.graphGen.writeSource(path='../../dnn_inference_CUDA_C/lide_c_dnnlayers/autoGenGraph/')
        self.graphGen.writeMakeGraph(path='../../dnn_inference_CUDA_C/lide_c_dnnlayers/autoGenGraph/')
        self.graphGen.writeDriver(path='../../dnn_inference_CUDA_C/lide_c_dnnlayers/test_c/autoGenModel/',
                                  backPath='../../../../graphGen/extractedParas/')
        self.graphGen.writeDriver(path='../../dnn_inference_CUDA_C/lide_c_dnnlayers/test_cuda/autoGenModel/',
                                  backPath='../../../../graphGen/extractedParas/')
    def convert(self,Pruned):
        loader = neuroGRS_loadPrunedModel(self.modelname,
                                          self.filename,self.path)
        if Pruned:
            self.model = loader.model_P
        else:
            self.model = loader.model_O
        f = open("../extractedParas/sampleSize.csv", "r")
        self.sampleSize = int(f.read())
        f.close()

        f = open("../extractedParas/sampleDim.csv", "r")
        self.sampleDim = int(f.read())
        f.close()




    def modelIdentify(self):
        if isinstance(self.model.layers[0], Conv2D):
            print("cnn model found")
            self.type = 'cnn'
            self.D_Row = int(self.model.input.shape.dims[1])
            self.D_Column = int(self.model.input.shape.dims[2])
        else:
            print("mlp model found")
            self.type = 'mlp'

        for each in self.model.layers:
            if isinstance(each, (Dense,Conv2D,MaxPooling2D)):
                self.rawLayers.append(each)

            if isinstance(each, Dense):
                self.DaCLayers.append(each)
                self.DaCName.append(each.name)
                self.shape.append(each.units)

            if isinstance(each, Conv2D):
                self.shape.append(each.filters)
                self.DaCName.append(each.name)
                self.DaCLayers.append(each)



        self.layersAmount = len(self.rawLayers)



if __name__ == "__main__":
    model = sys.argv[1] #'cnnsingle'
    fileName = sys.argv[2] #'_1006_e1_V{0}seed0'
    set = sys.argv[3] #'06e1'
    path = sys.argv[4] #'../../../NeuroGRSoutputSave/outputs0602bw/regenModelinfo/'
    tmp = sys.argv[5]
    Pruned = int(tmp) #'0' original, '1' pruned

    converter = neuroGRS_convert(model, fileName, set,
                                 path=path, Pruned=Pruned)

    #os.system('python3 neuroGRS_convert.py cnnsingle _1006_e1_V{0}seed0 06e1 ../../../NeuroGRSoutputSave/outputs0602bw/regenModelinfo/ True)

    print("NeuronGRS_Convert.py done")