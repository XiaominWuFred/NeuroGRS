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


class neuroGRS_byPass(Actor):
    def __init__(self, modelnameqin,valaccqin, testaccqin,typeqin,modelqout,modelnameqout,valaccqout,
                 testaccqout,parasqout,flopsqout,shapesqout,typeqout,):
        super().__init__(index=0, mode="COMMON")
        # inputs
        self.modelnameqin = modelnameqin

        self.valaccqin = valaccqin
        self.testaccqin = testaccqin
        self.typeqin = typeqin

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
        #fake IO
        self.modelnameqin.welt_py_fifo_basic_read_direct()
        self.valaccqin.welt_py_fifo_basic_read_direct()
        self.testaccqin.welt_py_fifo_basic_read_direct()
        self.typeqin.welt_py_fifo_basic_read_direct()

        # fake IO
        self.modelqout.welt_py_fifo_basic_write(None)
        self.modelnameqout.welt_py_fifo_basic_write('name')
        self.valaccqout.welt_py_fifo_basic_write([0.99,0.99,0.99])
        self.testaccqout.welt_py_fifo_basic_write(0.99)
        self.parasqout.welt_py_fifo_basic_write(100)
        self.flopsqout.welt_py_fifo_basic_write(150)
        self.shapesqout.welt_py_fifo_basic_write('1X1X1')
        self.typeqout.welt_py_fifo_basic_write('nocare')

    def terminate(self):
        pass