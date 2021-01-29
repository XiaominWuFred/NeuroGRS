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

class neuroGRS_select(Actor):
    def __init__(self,eup,select_dq_ary,select_Tq_ary,select_Fq_ary):
        super().__init__(index=0, mode="COMMON")
        self.EUP = eup
        self.dq_ary = select_dq_ary
        self.Tq_ary = select_Tq_ary
        self.Fq_ary = select_Fq_ary

    def enable(self):
        if self.Fq_ary[0].welt_py_fifo_basic_population() == 0 and \
                self.Tq_ary[0].welt_py_fifo_basic_population() == 0:
            return False
        else:
            if self.dq_ary[0].welt_py_fifo_basic_population() == 0:
                return True
            else:
                return False


    def invoke(self):
        #check
        print("select actor invokes")

        tokenPass = []
        if self.EUP:
            for eachQ in self.Tq_ary:
                tokenPass.append(eachQ.welt_py_fifo_basic_read_direct())
        else:
            for eachQ in self.Fq_ary:
                tokenPass.append(eachQ.welt_py_fifo_basic_read_direct())

        for i in range(len(tokenPass)):
            self.dq_ary[i].welt_py_fifo_basic_write(tokenPass[i])


    def terminate(self):
        pass