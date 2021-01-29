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
sys.path.append("../../wrapped/welter_py/src/gems/actors/common/")
from welt_py_actor import Actor
from statRecord import statRecord
from common import *

class neuroGRS_statRecord(Actor):
    def __init__(self,mode, filename, modelname_q, flops_train_q,
                 paras_train_q, flops_grs_q, paras_grs_q, valacc_grs_q, testacc_grs_q, shapes_grs_q,
                 flops_tq_q, paras_tq_q, valacc_tq_q, testacc_tq_q, model_amount,record=True):
        super().__init__(index=0, mode="COMMON")
        #inputs:
        self.filename = filename
        self.mode = mode

        self.modelname_q = modelname_q


        self.flops_train_q = flops_train_q
        self.paras_train_q = paras_train_q


        self.valaccAry_grs_q = valacc_grs_q
        self.testaccAry_grs_q = testacc_grs_q
        self.flops_grs_q = flops_grs_q
        self.paras_grs_q = paras_grs_q
        self.shapes_grs_q = shapes_grs_q


        self.valacc_tq_q = valacc_tq_q
        self.testacc_tq_q = testacc_tq_q
        self.flops_tq_q = flops_tq_q
        self.paras_tq_q = paras_tq_q

        #variable:
        self.statrecd = statRecord(record)
        self.model_amount = model_amount

        #output:

    def enable(self):
        if record_stat:
            if self.modelname_q.welt_py_fifo_basic_population() == 0:
                return False
            else:
                return True
        else:
            return False

    def invoke(self):
        #check
        print("statRecord actor invokes")

        if self.mode is False:
            self.statrecd.load_grs(self.filename,
                                   self.modelname_q.welt_py_fifo_basic_read_direct(),
                                   self.flops_train_q.welt_py_fifo_basic_read_direct(),
                                   self.paras_train_q.welt_py_fifo_basic_read_direct(),
                                   self.flops_grs_q.welt_py_fifo_basic_read_direct(),
                                   self.paras_grs_q.welt_py_fifo_basic_read_direct(),
                                   self.valaccAry_grs_q.welt_py_fifo_basic_read_direct(),
                                   self.testaccAry_grs_q.welt_py_fifo_basic_read_direct(),
                                   self.shapes_grs_q.welt_py_fifo_basic_read_direct())

            self.statrecd.run()


        else:
            self.statrecd.load_tq(self.filename,self.modelname_q.welt_py_fifo_basic_read_direct(),
                                  self.flops_train_q.welt_py_fifo_basic_read_direct(),
                                  self.paras_train_q.welt_py_fifo_basic_read_direct(),
                                  self.flops_grs_q.welt_py_fifo_basic_read_direct(),
                                  self.paras_grs_q.welt_py_fifo_basic_read_direct(),
                                  self.valaccAry_grs_q.welt_py_fifo_basic_read_direct(),
                                  self.testaccAry_grs_q.welt_py_fifo_basic_read_direct(),
                                  self.flops_tq_q.welt_py_fifo_basic_read_direct(),
                                  self.paras_tq_q.welt_py_fifo_basic_read_direct(),
                                  self.valacc_tq_q.welt_py_fifo_basic_read_direct(),
                                  self.testacc_tq_q.welt_py_fifo_basic_read_direct(),
                                  self.shapes_grs_q.welt_py_fifo_basic_read_direct())

            self.statrecd.run()


    def terminate(self):
        pass
