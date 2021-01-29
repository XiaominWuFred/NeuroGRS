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
from neuroGRS_NWM import neuroGRS_NWM
from neuroGRS_statRecord_shape import neuroGRS_statRecord_shape
from neuroGRS_fork_3 import neuroGRS_fork_3
from neuroGRS_consumer import neuroGRS_consumer
import queue


class neuroGRS_graph_expt(Graph):
    def __init__(self):
        self.actors = []
        self.args = [each for each in sys.argv]
        super().__init__(actorCount=7, fifoCount=1,scheduler=self.scheduler)
        self.descriptor = [None] * self.actorCount

    def construct(self):
        #Preparation for dataset
        try:
            dataPre = datasetPre(self.args[1:])
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


        #actor_fork_3_train
        fork_inq_ary_train = [modelnameqout_train, valaccq_train, testaccq_train,typeq_train]

        modelnameqout_train_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_train_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_train_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        typeq_train_1 = welt_py_fifo_basic_new(capacity=8, index=0)
        fork_1q_ary_train = [modelnameqout_train_1, valaccq_train_1, testaccq_train_1,typeq_train_1]

        modelnameqout_train_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_train_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_train_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        typeq_train_2 = welt_py_fifo_basic_new(capacity=8, index=0)
        fork_2q_ary_train = [modelnameqout_train_2, valaccq_train_2, testaccq_train_2,typeq_train_2]

        modelnameqout_train_3 = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_train_3 = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_train_3 = welt_py_fifo_basic_new(capacity=8, index=0)
        typeq_train_3 = welt_py_fifo_basic_new(capacity=8, index=0)
        fork_3q_ary_train = [modelnameqout_train_3, valaccq_train_3, testaccq_train_3,typeq_train_3]

        fork_3_train = neuroGRS_fork_3(fork_inq_ary_train, fork_1q_ary_train, fork_2q_ary_train, fork_3q_ary_train)


        #actor_grs
        modelqout_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        modelnameqout_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        shapesq_grs = welt_py_fifo_basic_new(capacity=8, index=0)
        typeq_grs = welt_py_fifo_basic_new(capacity=8, index=0)

        grs = neuroGRS_GRS(modelnameqout_train_1, valaccq_train_1, testaccq_train_1,typeq_train_1,
                modelqout_grs,modelnameqout_grs,valaccq_grs,testaccq_grs,parasq_grs,flopsq_grs,shapesq_grs,typeq_grs,
                           dataPre.file_name, 2,
                           dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                           dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,
                           dataPre.cnntrainX,dataPre.cnntrainY,dataPre.cnnvalX,
                           dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,
                           'grs', plot=True)

        #actor_rrs
        modelqout_rrs = welt_py_fifo_basic_new(capacity=8, index=0)
        modelnameqout_rrs = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_rrs = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_rrs = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_rrs = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_rrs = welt_py_fifo_basic_new(capacity=8, index=0)
        shapesq_rrs = welt_py_fifo_basic_new(capacity=8, index=0)
        typeq_rrs = welt_py_fifo_basic_new(capacity=8, index=0)

        rrs = neuroGRS_GRS(modelnameqout_train_2, valaccq_train_2, testaccq_train_2,typeq_train_2,
                modelqout_rrs,modelnameqout_rrs,valaccq_rrs,testaccq_rrs,parasq_rrs,flopsq_rrs,shapesq_rrs,typeq_rrs,
                           dataPre.file_name, 2,
                           dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                           dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,
                           dataPre.cnntrainX,dataPre.cnntrainY,dataPre.cnnvalX,
                           dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,
                           'rrs', plot=True)

        #actor_nwm
        modelqout_nwm = welt_py_fifo_basic_new(capacity=8, index=0)
        modelnameqout_nwm = welt_py_fifo_basic_new(capacity=8, index=0)
        valaccq_nwm = welt_py_fifo_basic_new(capacity=8, index=0)
        testaccq_nwm = welt_py_fifo_basic_new(capacity=8, index=0)
        flopsq_nwm = welt_py_fifo_basic_new(capacity=8, index=0)
        parasq_nwm = welt_py_fifo_basic_new(capacity=8, index=0)
        shapesq_nwm = welt_py_fifo_basic_new(capacity=8, index=0)
        typeq_nwm = welt_py_fifo_basic_new(capacity=8, index=0)

        nwm = neuroGRS_NWM(modelnameqout_train_3, valaccq_train_3, testaccq_train_3,typeq_train_3,
                modelqout_nwm,modelnameqout_nwm,valaccq_nwm,testaccq_nwm,parasq_nwm,flopsq_nwm,shapesq_nwm,typeq_nwm,
                           dataPre.file_name, 2,
                           dataPre.mlptrainX,dataPre.mlptrainY,dataPre.mlpvalX,
                           dataPre.mlpvalY,dataPre.mlptestX,dataPre.mlptestY,
                           dataPre.cnntrainX,dataPre.cnntrainY,dataPre.cnnvalX,
                           dataPre.cnnvalY,dataPre.cnntestX,dataPre.cnntestY,
                           'nwm', plot=True)

        #actor_srecd
        srecd = neuroGRS_statRecord_shape(dataPre.file_name,
                                          modelnameqout_nwm,
                                          shapesq_grs,shapesq_rrs,shapesq_nwm,
                                          valaccq_grs,valaccq_rrs,valaccq_nwm,
                                          record=True)

        #actor_consumer
        consumer_inq_ary = [typeq_nwm,parasq_nwm,flopsq_nwm,testaccq_nwm,modelqout_nwm,
                            typeq_rrs,parasq_rrs,flopsq_rrs,testaccq_rrs,modelnameqout_rrs,modelqout_rrs,
                            typeq_grs,parasq_grs,flopsq_grs,testaccq_grs,modelnameqout_grs,modelqout_grs,
                            flopsq_train,parasq_train]
        consumer = neuroGRS_consumer(consumer_inq_ary)

        # schedule actors
        actor_train = 0
        actor_fork_3_train = 1
        actor_grs = 2
        actor_rrs = 3
        actor_nwm = 4
        actor_srecd = 5
        actor_consumer = 6


        self.welt_py_graph_set_actor(actor_train, train)
        self.descriptor[actor_train] = 'actor_train'
        self.welt_py_graph_set_actor(actor_fork_3_train, fork_3_train)
        self.descriptor[actor_fork_3_train] = 'actor_fork_3_train'
        self.welt_py_graph_set_actor(actor_grs, grs)
        self.descriptor[actor_grs] = 'actor_grs'
        self.welt_py_graph_set_actor(actor_rrs, rrs)
        self.descriptor[actor_rrs] = 'actor_rrs'
        self.welt_py_graph_set_actor(actor_nwm, nwm)
        self.descriptor[actor_nwm] = 'actor_nwm'
        self.welt_py_graph_set_actor(actor_srecd, srecd)
        self.descriptor[actor_srecd] = 'actor_srecd'
        self.welt_py_graph_set_actor(actor_consumer, consumer)
        self.descriptor[actor_consumer] = 'actor_consumer'

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

        graph = neuroGRS_graph_expt()
        graph.construct()
        graph.scheduler()
        print("ALL done")