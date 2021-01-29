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

import pandas as pd
from common import *
from modelPre import modelPre
from datasetPre import datasetPre
from training import training
from grs import grs
from t import t
from q import q


class designEvaluation(object):
    def __init__(self,record=True):
        #inputs
        self.model_name = None
        self.file_name = None
        self.paras = None
        self.flops = None
        self.testacc = None
        #variables
        self.record = record
        #outputs
        self.results = {}

    def load(self,model,model_name,flops,paras,testacc,valacc):
        self.model = model
        self.model_name = model_name
        self.paras= paras
        self.flops = flops
        try:
            self.testacc = testacc[len(testacc)-1]
            self.valacc = valacc[len(valacc)-1]
        except:
            self.testacc = testacc
            self.valacc = valacc

    def run(self):
        self.results[self.model_name] = [self.model,self.paras,self.flops,self.testacc,self.valacc]


if __name__ == "__main__":
    dataPre = datasetPre('v','04e1')
    modelPre = modelPre(dataPre.D,dataPre.D1)
    training = training(modelPre.models[2],modelPre.model_name[2],dataPre.file_name,dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                        dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,dataPre.cnntrainX,dataPre.cnntrainY,
                        dataPre.cnnvalX,dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,savemodel = True)
    mlporcnn = training.run()

    if mlporcnn == 'mlp':
        grs = grs(modelPre.model_name[2],dataPre.file_name,modelPre.models[2],training.valacc,training.testacc,2,
                  dataPre.mlptrainX, dataPre.mlptrainY, dataPre.mlpvalX,dataPre.mlpvalY, dataPre.mlptestX, dataPre.mlptestY,'rrs',True)
    else:
        grs = grs(modelPre.model_name[2], dataPre.file_name, modelPre.models[2], training.valacc, training.testacc, 2,
                  dataPre.cnntrainX, dataPre.cnntrainY,dataPre.cnnvalX, dataPre.cnnvalY, dataPre.cnntestX, dataPre.cnntestY,'rrs',True)

    grs.run()

    if mlporcnn == 'mlp':
        t = t(grs.model_name,dataPre.file_name,grs.model,training.valacc,training.testacc,
                  dataPre.mlpvalX,dataPre.mlpvalY, dataPre.mlptestX, dataPre.mlptestY)
    else:
        t = t(grs.model_name, dataPre.file_name, grs.model, grs.valacchis[len(grs.valacchis)-1], grs.testacchis[len(grs.testacchis)-1],
                  dataPre.cnnvalX, dataPre.cnnvalY, dataPre.cnntestX, dataPre.cnntestY)

    t.run()

    if mlporcnn == 'mlp':
        q = q(t.model_name,dataPre.file_name,t.model,t.finalaccval,t.finalacctest,
                  dataPre.mlpvalX,dataPre.mlpvalY, dataPre.mlptestX, dataPre.mlptestY)
    else:
        q = q(t.model_name, dataPre.file_name, t.model, t.finalaccval, t.finalacctest,
                  dataPre.cnnvalX, dataPre.cnnvalY, dataPre.cnntestX, dataPre.cnntestY)

    q.run()

    eva = designEvaluation()
    #eva.load_grs(grs.model_name,grs.file_name,training.paras,grs.paras,training.flops,grs.flops,grs.testacchis,grs.valacchis,grs.shapeString)
    eva.load_tq(q.file_name,grs.model,q.model_name,training.flops,training.paras,grs.flops,grs.paras,
                grs.valacchis,grs.testacchis,t.flops,q.finalParas,q.finalaccval,q.finalacctest,grs.shapeString)
    eva.run()
