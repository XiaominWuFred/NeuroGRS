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


class neuroGRS_statRecord_shape(Actor):
    def __init__(self,filename,modelname_q,
                 shapes_grs_q,shapes_rrs_q,shapes_nwm_q,valacc_grs_q,valacc_rrs_q,valacc_nwm_q,
                 record=True):
        super().__init__(index=0, mode="COMMON")
        #inputs:
        self.filename = filename
        self.modelname_q = modelname_q
        self.shapes_grs_q = shapes_grs_q
        self.shapes_rrs_q = shapes_rrs_q
        self.shapes_nwm_q = shapes_nwm_q
        self.valacc_grs_q = valacc_grs_q
        self.valacc_rrs_q = valacc_rrs_q
        self.valacc_nwm_q = valacc_nwm_q
        #variables:
        self.statrecd = statRecord(record)
        #output:

    def enable(self):
        if self.modelname_q.welt_py_fifo_basic_population() == 0:
            return False
        else:
            return True

    def invoke(self):
        #check
        print("statRecord_shape actor invokes")

        self.statrecd.loadShapeComp(self.filename,
                                    self.modelname_q.welt_py_fifo_basic_read_direct(),
                                    self.shapes_grs_q.welt_py_fifo_basic_read_direct(),
                                    self.shapes_rrs_q.welt_py_fifo_basic_read_direct(),
                                    self.shapes_nwm_q.welt_py_fifo_basic_read_direct(),
                                    self.valacc_grs_q.welt_py_fifo_basic_read_direct(),
                                    self.valacc_rrs_q.welt_py_fifo_basic_read_direct(),
                                    self.valacc_nwm_q.welt_py_fifo_basic_read_direct())

        self.statrecd.runShapeComparison()


    def terminate(self):
        pass