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


class statRecord(object):
    def __init__(self,record=True):
        #inputs
        self.model_name = None
        self.file_name = None
        self.paras_train = None
        self.paras_grs = None
        self.paras_tq = None
        self.flops_train = None
        self.flops_grs = None
        self.flops_tq = None
        self.testacc_grs = None
        self.testacc_tq = None
        self.valacc_grs = None
        self.valacc_tq = None
        self.shapes_grs = None
        self.shapes_rrs = None
        self.shapes_nwm = None
        #variables
        self.record = record
        self.testacc_train = None
        self.valacc_train = None
        self.valacc_rrs = None
        self.valacc_nwm = None
        #outputs
        #self.results = {}

    def load_grs(self,file_name,model_name,flops_train,paras_train,flops_grs,paras_grs,valacc_grs,testacc_grs,shapes):
        self.model_name = model_name
        self.file_name = file_name
        self.paras_train = paras_train
        self.paras_grs = paras_grs
        self.paras_tq = None
        self.flops_train = flops_train
        self.flops_grs = flops_grs
        self.flops_tq = None

        self.testacc_grs = testacc_grs
        self.testacc_tq = None

        self.valacc_grs = valacc_grs
        self.valacc_tq = None
        self.shapes_grs = shapes

    def load_tq(self,file_name,model_name,flops_train,paras_train,flops_grs,paras_grs,valacc_grs,testacc_grs,flops_tq,paras_tq,valacc_tq,testacc_tq,shapes):
        self.model_name = model_name
        self.file_name = file_name
        self.paras_train = paras_train
        self.paras_grs = paras_grs
        self.paras_tq = paras_tq
        self.flops_train = flops_train
        self.flops_grs = flops_grs
        self.flops_tq = flops_tq

        self.testacc_grs = testacc_grs
        self.testacc_tq = testacc_tq

        self.valacc_grs = valacc_grs
        self.valacc_tq = valacc_tq
        self.shapes_grs = shapes

    def run(self):
        self.testacc_train = self.testacc_grs[0]
        self.valacc_train = self.valacc_grs[0]
        '''
        if self.paras_tq is None:
            self.results[self.model_name] = [self.model,self.paras_grs,self.flops_grs,self.testacc_grs[len(self.testacc_grs)-1],self.valacc_grs[len(self.valacc_grs)-1],
                                             (self.paras_train-self.paras_grs)/self.paras_train,
                                             (self.flops_train-self.flops_grs)/self.flops_train,
                                             (self.testacc_train-self.testacc_grs[len(self.testacc_grs)-1])/self.testacc_train,
                                             (self.valacc_train-self.valacc_grs[len(self.valacc_grs)-1])/self.valacc_train]
        else:
            self.results[self.model_name] = [self.model,self.paras_tq,self.flops_tq,self.testacc_tq,self.valacc_tq,
                                             (self.paras_train-self.paras_tq)/self.paras_train,
                                             (self.flops_train-self.flops_tq)/self.flops_train,
                                             (self.testacc_train-self.testacc_tq)/self.testacc_train,
                                             (self.valacc_train-self.valacc_tq)/self.valacc_train]
        '''

        if self.record:
            oriStruc_save = self.shapes_grs[0]
            finalStruc_save = self.shapes_grs[len(self.shapes_grs) - 1]

            if self.testacc_train * self.valacc_train != 0:
                if self.paras_tq is None:
                    df = pd.DataFrame({'GRS_TestAcc(lost%)': [str(round(self.testacc_grs[len(self.testacc_grs)-1], 3)) + '(' + str(
                        (self.testacc_train - self.testacc_grs[len(self.testacc_grs)-1]) * 100 / self.testacc_train) + '%)'],
                                       'GRS_ValAcc(lost%)': [str(round(self.valacc_grs[len(self.valacc_grs)-1], 3)) + '(' + str(
                                           (self.valacc_train - self.valacc_grs[len(self.valacc_grs)-1]) * 100 / self.valacc_train) + '%)'],
                                       'GRS_FLOPs(reduced%)': [str(round(self.flops_grs, 3)) + '(' + str(
                                           (self.flops_train - self.flops_grs) * 100 / self.flops_train) + '%)'],
                                       'GRS_Paras(reduced%)': [str(round(self.paras_grs, 3)) + '(' + str(
                                           (self.paras_train - self.paras_grs) * 100 / self.paras_train) + '%)'],
                                       'Original structure': [oriStruc_save],
                                       'GRS_found structure': [finalStruc_save]
                                       })
                else:
                    df = pd.DataFrame({'GRS_TestAcc(lost%)': [str(round(self.testacc_grs[len(self.testacc_grs)-1], 3)) + '(' + str((self.testacc_train-self.testacc_grs[len(self.testacc_grs)-1])*100/self.testacc_train) + '%)'],
                                       'GRS_ValAcc(lost%)': [str(round(self.valacc_grs[len(self.valacc_grs)-1], 3)) + '(' + str((self.valacc_train-self.valacc_grs[len(self.valacc_grs)-1])*100/self.valacc_train) + '%)'],
                                       'GRS_FLOPs(reduced%)': [str(round(self.flops_grs, 3)) + '(' + str((self.flops_train-self.flops_grs)*100/self.flops_train) + '%)'],
                                       'GRS_Paras(reduced%)': [str(round(self.paras_grs, 3)) + '(' + str((self.paras_train-self.paras_grs)*100/self.paras_train) + '%)'],

                                       'TQ_TestAcc(lost%)': [str(round(self.testacc_tq, 3)) + '(' + str((self.testacc_train-self.testacc_tq)*100/self.testacc_train) + '%)'],
                                       'TQ_ValAcc(lost%)': [str(round(self.valacc_tq, 3)) + '(' + str((self.valacc_train-self.valacc_tq)*100/self.valacc_train) + '%)'],
                                       'TQ_FLOPs(reduced%)': [str(round(self.flops_tq, 3)) + '(' + str((self.flops_train-self.flops_tq)*100/self.flops_train) + '%)'],
                                       'TQ_Paras(reduced%)': [str(round(self.paras_tq, 3)) + '(' + str((self.paras_train-self.paras_tq)*100/self.paras_train) + '%)'],
                                       'Original structure': [oriStruc_save],
                                       'GRS_found structure': [finalStruc_save]
                                       })
                df.to_csv('../../outputs/prunedM/' + self.model_name+self.file_name + '_designEvaluation.csv')

    def loadShapeComp(self,file_name,model_name,shapes_grs,shapes_rrs,shapes_nwm,valacc_grs,valacc_rrs,valacc_nwm):
        self.file_name = file_name
        self.model_name = model_name
        self.shapes_grs = shapes_grs
        self.shapes_rrs = shapes_rrs
        self.shapes_nwm = shapes_nwm
        self.valacc_grs = valacc_grs
        self.valacc_rrs = valacc_rrs
        self.valacc_nwm = valacc_nwm

    def runShapeComparison(self):
        if self.shapes_rrs is not None and self.shapes_nwm is not None:
            avgvalAcc_g = avgValAcc(self.valacc_grs)
            avgvalAcc_r = avgValAcc(self.valacc_rrs)
            avgvalAcc_n = avgValAcc(self.valacc_nwm)

            bf = pd.DataFrame({
                'shape_original': [self.shapes_grs[0]],
                'shape_P_GRS': [self.shapes_grs[len(self.shapes_grs) - 1]],
                'shape_P_RRS': [self.shapes_rrs[len(self.shapes_rrs) - 1]],
                'shape_P_NWM': [self.shapes_nwm[len(self.shapes_nwm) - 1]],
                'avg_ValAcc_GRS': [avgvalAcc_g],
                'avg_ValAcc_RRS': [avgvalAcc_r],
                'avg_ValAcc_NWM': [avgvalAcc_n]

            })
            bf.to_csv('../../outputs/prunedM/' + self.model_name + self.file_name + '_comparisonstats.csv')

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

    srecd = statRecord()
    #eva.load_grs(grs.model_name,grs.file_name,training.paras,grs.paras,training.flops,grs.flops,grs.testacchis,grs.valacchis,grs.shapeString)
    srecd.load_tq(q.file_name,grs.model,q.model_name,training.flops,training.paras,grs.flops,grs.paras,
                grs.valacchis,grs.testacchis,t.flops,q.finalParas,q.finalaccval,q.finalacctest,grs.shapeString)
    srecd.run()
