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
from grs import grs


class neuroGRS_GRS(Actor):
    def __init__(self, modelnameqin,valaccqin, testaccqin,typeqin,modelqout,modelnameqout,valaccqout,
                 testaccqout,parasqout,flopsqout,shapesqout,typeqout,
                 filename,  smallestUnit, mlptrainX,mlptrainY, mlpvalX, mlpvalY, mlptestX, mlptestY,
                 cnntrainX, cnntrainY, cnnvalX, cnnvalY, cnntestX, cnntestY,
                 type, plot=True):
        super().__init__(index=0, mode="COMMON")
        # inputs
        self.modelnameqin = modelnameqin

        self.valaccqin = valaccqin
        self.testaccqin = testaccqin
        self.typeqin = typeqin
        self.mlpdata = [mlptrainX, mlptrainY, mlpvalX, mlpvalY, mlptestX, mlptestY]
        self.cnndata = [cnntrainX, cnntrainY, cnnvalX, cnnvalY, cnntestX, cnntestY]
        # variable:
        self.grs = grs(filename,smallestUnit,type,plot)
        # outputs:
        self.modelqout = modelqout
        self.modelnameqout = modelnameqout
        self.valaccqout = valaccqout
        self.testaccqout = testaccqout
        self.parasqout = parasqout
        self.flopsqout = flopsqout
        self.shapesqout = shapesqout
        self.typeqout = typeqout

    def enable(self):
        if self.modelnameqin.welt_py_fifo_basic_population() == 0:
            return False
        else:
            if self.modelqout.welt_py_fifo_basic_population() == 0:
                return True
            else:
                return False

    def invoke(self):
        #check
        print("GRS actor invokes")

        type = self.typeqin.welt_py_fifo_basic_read_direct()
        if type == 'mlp':
            self.grs.load( self.modelnameqin.welt_py_fifo_basic_read_direct(),
                     self.mlpdata[0], self.mlpdata[1], self.mlpdata[2], self.mlpdata[3], self.mlpdata[4],
                     self.mlpdata[5], self.valaccqin.welt_py_fifo_basic_read_direct(), 
                           self.testaccqin.welt_py_fifo_basic_read_direct())
        else:
            self.grs.load( self.modelnameqin.welt_py_fifo_basic_read_direct(),
                     self.cnndata[0], self.cnndata[1], self.cnndata[2], self.cnndata[3], self.cnndata[4],
                     self.cnndata[5], self.valaccqin.welt_py_fifo_basic_read_direct(), 
                           self.testaccqin.welt_py_fifo_basic_read_direct())

        self.grs.run()

        self.modelqout.welt_py_fifo_basic_write(self.grs.model)
        self.modelnameqout.welt_py_fifo_basic_write(self.grs.model_name)
        self.valaccqout.welt_py_fifo_basic_write(self.grs.valacchis)
        self.testaccqout.welt_py_fifo_basic_write(self.grs.testacchis)

        self.parasqout.welt_py_fifo_basic_write(self.grs.paras)
        self.flopsqout.welt_py_fifo_basic_write(self.grs.flops)
        self.shapesqout.welt_py_fifo_basic_write(self.grs.shapeString)
        self.typeqout.welt_py_fifo_basic_write(type)

    def terminate(self):
        pass
