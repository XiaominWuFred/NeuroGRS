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
from training import training

class neuroGRS_training(Actor):
    def __init__(self,modelnameq,modelnameqout,valaccq,testaccq,flopsq,parasq,typeq,
                 file_name,mlptrainX,mlptrainY,mlpvalX,mlpvalY,mlptestX,mlptestY,
                 cnntrainX,cnntrainY,cnnvalX,cnnvalY,cnntestX,cnntestY,
                 savemodel):
        super().__init__(index=0, mode="COMMON")
        self.modelnameq = modelnameq

        self.modelnameqout = modelnameqout
        self.valaccq = valaccq
        self.testaccq = testaccq
        self.flopsq = flopsq
        self.parasq = parasq
        self.typeq = typeq

        self.train = training(file_name,mlptrainX,mlptrainY,mlpvalX,
                        mlpvalY,mlptestX,mlptestY,cnntrainX,cnntrainY,
                        cnnvalX,cnnvalY,cnntestX,cnntestY,savemodel)

    def enable(self):
        if self.modelnameq.welt_py_fifo_basic_population() == 0:
            return False
        else:
            if self.modelnameqout.welt_py_fifo_basic_population() == 0:
                return True
            else:
                return False

    def invoke(self):
        #check
        print("training actor invokes")

        self.train.load(self.modelnameq.welt_py_fifo_basic_read_direct())
        self.train.run()

        self.modelnameqout.welt_py_fifo_basic_write(self.train.model_name)
        self.valaccq.welt_py_fifo_basic_write(self.train.valacc)
        self.testaccq.welt_py_fifo_basic_write(self.train.testacc)
        self.parasq.welt_py_fifo_basic_write(self.train.paras)
        self.flopsq.welt_py_fifo_basic_write(self.train.flops)
        self.typeq.welt_py_fifo_basic_write(self.train.cnnormlp)


    def terminate(self):
        pass