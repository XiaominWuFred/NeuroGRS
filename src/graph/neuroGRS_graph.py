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
sys.path.append("../actors")
sys.path.append("../util")
sys.path.append("../../graphGen/util")
sys.path.append("../../wrapped/welter_py/src/gems/edges")
sys.path.append("../../wrapped/welter_py/src/tools/graph")
sys.path.append("../../wrapped/welter_py/src/tools/runtime")

from welt_py_graph import Graph
from welt_py_util import welt_py_util_simple_scheduler
from welt_py_fifo_basic import welt_py_fifo_basic_new

import modelPre as prem
from datasetPre import datasetPre
from neuroGRS_training import neuroGRS_training
from neuroGRS_GRS import neuroGRS_GRS
from neuroGRS_TQ import neuroGRS_TQ
from neuroGRS_designEvaluation import neuroGRS_designEvaluation
from neuroGRS_designSelection import neuroGRS_designSelection
from neuroGRS_statRecord import neuroGRS_statRecord
from neuroGRS_switch import neuroGRS_switch
from neuroGRS_select import neuroGRS_select
from neuroGRS_fork import neuroGRS_fork



class neuroGRS_graph(Graph):
    def __init__(self):
        try:
            if int(sys.argv[1]) == 1:
                self.EUP = True
            else:
                self.EUP = False
        except:
            print("Please provide EUP as first argument")

        self.args = [each for each in sys.argv]

        super().__init__(actorCount=11, fifoCount=1,scheduler=self.scheduler)
        self.descriptor = [None] * self.actorCount

    def construct(self):
        #Preparation for dataset
        try:
            #version with experiment data set
            dataPre = datasetPre(self.args[2:])
            #version with common data set
            
        except:
            print("Please provide arguments needed by dataPre begin with the second argument")
            return None
        #Preparation for models
        modelPre = prem.modelPre(dataPre.D_Row, dataPre.D_Column)

        #actor_train
        #input FIFO
        modelnameq_train = welt_py_fifo_basic_new(capacity=8, index=0)
        #output FIFO

        modelnameqout_train = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_train = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_train = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_train = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_train = welt_py_fifo_basic_new(capacity=8, index=0)
        typeq_train = welt_py_fifo_basic_new(capacity=8, index=0)

        for each in modelPre.model_name:
            modelnameq_train.welt_py_fifo_basic_write(each)

        train = neuroGRS_training(modelnameq_train,
                                  modelnameqout_train,valaccq_train,testaccq_train,
                                  flopsq_train,parasq_train,typeq_train,
                                  dataPre.file_name,dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                        dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,dataPre.cnntrainX,dataPre.cnntrainY,
                        dataPre.cnnvalX,dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,savemodel=True)


        #actor_fork_train
        fork_inq_ary_train = [modelnameqout_train]

        modelnameqout_train_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        fork_1q_ary_train = [modelnameqout_train_1]

        modelnameqout_train_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        fork_2q_ary_train = [modelnameqout_train_2]
        fork_train = neuroGRS_fork(fork_inq_ary_train, fork_1q_ary_train, fork_2q_ary_train)


        #actor_grs
        modelqout_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        modelnameqout_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        shapesq_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        typeq_grs = welt_py_fifo_basic_new(capacity=8, index=0)

        grs = neuroGRS_GRS(modelnameqout_train_1,valaccq_train, testaccq_train,typeq_train,
                modelqout_grs,modelnameqout_grs,valaccq_grs,testaccq_grs,parasq_grs,flopsq_grs,shapesq_grs,typeq_grs,
                           dataPre.file_name, 2,
                           dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                           dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,
                           dataPre.cnntrainX,dataPre.cnntrainY,dataPre.cnnvalX,
                           dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,
                           'grs', plot=True)


        #actor_fork_grs
        fork_inq_ary_grs = [valaccq_grs,testaccq_grs,parasq_grs,flopsq_grs]

        valaccq_grs_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_grs_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_grs_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_grs_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        fork_1q_ary_grs = [valaccq_grs_1,testaccq_grs_1,parasq_grs_1,flopsq_grs_1]

        valaccq_grs_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_grs_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_grs_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_grs_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        fork_2q_ary_grs = [valaccq_grs_2,testaccq_grs_2,parasq_grs_2,flopsq_grs_2]
        fork_grs = neuroGRS_fork(fork_inq_ary_grs, fork_1q_ary_grs, fork_2q_ary_grs)

        #actor_switch
        switch_dq_ary = [modelqout_grs,modelnameqout_grs,flopsq_grs_1,parasq_grs_1,testaccq_grs_1,
                         valaccq_grs_1,typeq_grs]

        modelqout_switch_T = welt_py_fifo_basic_new(capacity=8, index=0)
        modelnameqout_switch_T = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_switch_T = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_switch_T = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_switch_T = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_switch_T = welt_py_fifo_basic_new(capacity=8, index=0)
        typeq_switch_T = welt_py_fifo_basic_new(capacity=8, index=0)
        switch_Tq_ary = [modelqout_switch_T, modelnameqout_switch_T,
                 flopsq_switch_T, parasq_switch_T, testaccq_switch_T,
                         valaccq_switch_T,typeq_switch_T]

        modelqout_switch_F = welt_py_fifo_basic_new(capacity=8, index=0)
        modelnameqout_switch_F = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_switch_F = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_switch_F = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_switch_F = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_switch_F = welt_py_fifo_basic_new(capacity=8, index=0)

        switch_Fq_ary = [modelqout_switch_F, modelnameqout_switch_F,
                 flopsq_switch_F, parasq_switch_F, testaccq_switch_F,
                         valaccq_switch_F,None]
        switch = neuroGRS_switch(self.EUP,switch_dq_ary,switch_Tq_ary,switch_Fq_ary)


        #actor_tq
        modelqout_tq = welt_py_fifo_basic_new(capacity=8, index=0)
        modelnameqout_tq = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_tq = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_tq = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_tq = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_tq = welt_py_fifo_basic_new(capacity=8, index=0)

        tq = neuroGRS_TQ(modelnameqout_switch_T, modelqout_switch_T,valaccq_switch_T, testaccq_switch_T,typeq_switch_T,
                         modelqout_tq,modelnameqout_tq,valaccq_tq,testaccq_tq,parasq_tq,flopsq_tq,
                           dataPre.file_name,
                           dataPre.mlpvalX,dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,
                           dataPre.cnnvalX,dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,
                           thrs=0.001,minsize=0.001, record=True)


        #actor_fork_tq
        fork_inq_ary_tq = [valaccq_tq,testaccq_tq,parasq_tq,flopsq_tq]

        valaccq_tq_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_tq_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_tq_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_tq_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        fork_1q_ary_tq = [valaccq_tq_1,testaccq_tq_1,parasq_tq_1,flopsq_tq_1]

        valaccq_tq_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_tq_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_tq_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_tq_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        fork_2q_ary_tq = [valaccq_tq_2,testaccq_tq_2,parasq_tq_2,flopsq_tq_2]

        fork_tq = neuroGRS_fork(fork_inq_ary_tq,fork_1q_ary_tq,fork_2q_ary_tq)

        #actor select

        modelqout_select = welt_py_fifo_basic_new(capacity=8, index=0)
        modelnameqout_select = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_select = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_select = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_select = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_select = welt_py_fifo_basic_new(capacity=8, index=0)
        select_dq_ary = [modelqout_select, modelnameqout_select,
                 flopsq_select, parasq_select, testaccq_select,
                         valaccq_select]
        select_Tq_ary = [modelqout_tq,modelnameqout_tq,
                         flopsq_tq_1,parasq_tq_1,testaccq_tq_1,
                         valaccq_tq_1]
        select_Fq_ary = [modelqout_switch_F, modelnameqout_switch_F,
                 flopsq_switch_F, parasq_switch_F, testaccq_switch_F,
                         valaccq_switch_F]
        select = neuroGRS_select(self.EUP,select_dq_ary,select_Tq_ary,select_Fq_ary)


        #actor_eva
        resultqout_eva = welt_py_fifo_basic_new(capacity=8, index=0)

        eva = neuroGRS_designEvaluation(modelqout_select, modelnameqout_select,
                 flopsq_select, parasq_select, testaccq_select,valaccq_select,
                 resultqout_eva, model_amount=4,record=True)

        #actor_sel
        resultqout_sel = welt_py_fifo_basic_new(capacity=8, index=0)
        choice = 'maxtestacc'

        sel = neuroGRS_designSelection(choice,resultqout_eva, resultqout_sel)


        #actor_srecd
        srecd = neuroGRS_statRecord(self.EUP, dataPre.file_name, modelnameqout_train_2, flopsq_train,
                 parasq_train, flopsq_grs_2, parasq_grs_2, valaccq_grs_2, testaccq_grs_2, shapesq_grs,
                 flopsq_tq_2, parasq_tq_2, valaccq_tq_2, testaccq_tq_2, model_amount=4,record=True)

        #schedule actors
        actor_train = 0
        actor_fork_train = 1
        actor_grs = 2
        actor_fork_grs = 3
        actor_switch = 4
        actor_tq = 5
        actor_fork_tq = 6
        actor_select = 7
        actor_eva = 8
        actor_sel = 9
        actor_srecd = 10

        self.welt_py_graph_set_actor(actor_train, train)
        self.descriptor[actor_train] = 'actor_train'
        self.welt_py_graph_set_actor(actor_fork_train, fork_train)
        self.descriptor[actor_fork_train] = 'actor_fork_train'
        self.welt_py_graph_set_actor(actor_grs, grs)
        self.descriptor[actor_grs] = 'actor_grs'
        self.welt_py_graph_set_actor(actor_fork_grs, fork_grs)
        self.descriptor[actor_fork_grs] = 'actor_fork_grs'
        self.welt_py_graph_set_actor(actor_switch, switch)
        self.descriptor[actor_switch] = 'actor_switch'
        self.welt_py_graph_set_actor(actor_tq, tq)
        self.descriptor[actor_tq] = 'actor_tq'
        self.welt_py_graph_set_actor(actor_fork_tq, fork_tq)
        self.descriptor[actor_fork_tq] = 'actor_fork_tq'
        self.welt_py_graph_set_actor(actor_select, select)
        self.descriptor[actor_select] = 'actor_select'
        self.welt_py_graph_set_actor(actor_eva, eva)
        self.descriptor[actor_eva] = 'actor_eva'
        self.welt_py_graph_set_actor(actor_sel, sel)
        self.descriptor[actor_sel] = 'actor_sel'
        self.welt_py_graph_set_actor(actor_srecd, srecd)
        self.descriptor[actor_srecd] = 'actor_srecd'

    def scheduler(self):
        welt_py_util_simple_scheduler(self.actors, self.actorCount, self.descriptor)


if __name__ == "__main__":
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    import os
    from common import *

    os.environ["CUDA_VISIBLE_DEVICES"] = GPUscope
    with tf.device(GPU):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        graph = neuroGRS_graph()
        graph.construct()
        graph.scheduler()

        print("ALL done")
