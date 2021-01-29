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
from t import t
from q import q


class neuroGRS_TQ(Actor):
    def __init__(self, modelnameqin, modelqin, valaccqin, testaccqin, typeqin, modelqout, modelnameqout, valaccqout,
                 testaccqout, parasqout, flopsqout,
                 filename, mlpvalX, mlpvalY, mlptestX, mlptestY,
                 cnnvalX, cnnvalY, cnntestX, cnntestY,
                 thrs=0.001,minsize=0.001, record=True):
        super().__init__(index=0, mode="COMMON")
        # inputs
        self.modelnameqin = modelnameqin
        self.modelqin = modelqin
        self.valaccqin = valaccqin
        self.testaccqin = testaccqin
        self.typeqin = typeqin
        self.mlpdata = [mlpvalX, mlpvalY, mlptestX, mlptestY]
        self.cnndata = [cnnvalX, cnnvalY, cnntestX, cnntestY]
        # variable:
        self.t = t(filename,thrs,minsize,record)
        self.q = q(filename,record)
        # outputs:
        self.modelqout = modelqout
        self.modelnameqout = modelnameqout
        self.valaccqout = valaccqout
        self.testaccqout = testaccqout
        self.parasqout = parasqout
        self.flopsqout = flopsqout

    def enable(self):
        if self.modelqin.welt_py_fifo_basic_population() == 0:
            return False
        else:
            if self.modelnameqout.welt_py_fifo_basic_population() == 0:
                return True
            else:
                return False

    def invoke(self):
        #check
        print("TQ actor invokes")

        type = self.typeqin.welt_py_fifo_basic_read_direct()

        if type == 'mlp':
            self.t.load(self.modelqin.welt_py_fifo_basic_read_direct(),
                        self.modelnameqin.welt_py_fifo_basic_read_direct(),
                        self.mlpdata[0], self.mlpdata[1], self.mlpdata[2], self.mlpdata[3],
                        self.valaccqin.welt_py_fifo_basic_read_direct(), 
                        self.testaccqin.welt_py_fifo_basic_read_direct())
        else:
            self.t.load(self.modelqin.welt_py_fifo_basic_read_direct(),
                        self.modelnameqin.welt_py_fifo_basic_read_direct(),
                        self.cnndata[0], self.cnndata[1], self.cnndata[2], self.cnndata[3],
                        self.valaccqin.welt_py_fifo_basic_read_direct(), self.testaccqin.welt_py_fifo_basic_read_direct())

        self.t.run()

        if type == 'mlp':
            self.q.load(self.t.model, self.t.model_name,
                          self.mlpdata[0], self.mlpdata[1], self.mlpdata[2], self.mlpdata[3],
                          self.t.finalaccval, self.t.finalacctest)
        else:
            self.q.load(self.t.model, self.t.model_name,
                          self.cnndata[0], self.cnndata[1], self.cnndata[2], self.cnndata[3],
                          self.t.finalaccval, self.t.finalacctest)

        self.q.run()


        self.modelqout.welt_py_fifo_basic_write(self.q.model)
        self.modelnameqout.welt_py_fifo_basic_write(self.q.model_name)
        self.valaccqout.welt_py_fifo_basic_write(self.q.finalaccval)
        self.testaccqout.welt_py_fifo_basic_write(self.q.finalacctest)
        self.parasqout.welt_py_fifo_basic_write(self.q.finalParas)
        self.flopsqout.welt_py_fifo_basic_write(self.t.flops)
        print("tq done")


    def terminate(self):
        pass