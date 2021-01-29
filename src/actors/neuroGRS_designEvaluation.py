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
from designEvaluation import designEvaluation


class neuroGRS_designEvaluation(Actor):
    def __init__(self, model_q, modelname_q,
                 flops_q, paras_q, testacc_q, valacc_q,
                 result_q, model_amount,record=True):
        super().__init__(index=0, mode="COMMON")
        #inputs:
        self.model_q = model_q
        self.modelname_q = modelname_q
        self.testacc_q = testacc_q
        self.valacc_q = valacc_q
        self.flops_q = flops_q
        self.paras_q = paras_q
        #variable:
        self.eva = designEvaluation(record)
        self.model_amount = model_amount
        #output:
        self.result_q = result_q


    def enable(self):
        if self.model_q.welt_py_fifo_basic_population() == 0:
            return False
        else:
            if self.result_q.welt_py_fifo_basic_population() == 0:
                return True
            else:
                return False



    def invoke(self):
        #check
        print("designEvaluation actor invokes")
        self.eva.load(self.model_q.welt_py_fifo_basic_read_direct(),
                      self.modelname_q.welt_py_fifo_basic_read_direct(),
                      self.flops_q.welt_py_fifo_basic_read_direct(),
                      self.paras_q.welt_py_fifo_basic_read_direct(),
                      self.testacc_q.welt_py_fifo_basic_read_direct(),
                      self.valacc_q.welt_py_fifo_basic_read_direct())

        self.eva.run()
        if len(self.eva.results) == self.model_amount:
            self.result_q.welt_py_fifo_basic_write(self.eva.results)

    def terminate(self):
        pass

