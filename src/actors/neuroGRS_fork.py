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
import copy

class neuroGRS_fork(Actor):
    def __init__(self,fork_inq_ary,fork_1q_ary,fork_2q_ary):
        super().__init__(index=0, mode="COMMON")
        self.inq_ary = fork_inq_ary
        self.f_1_ary = fork_1q_ary
        self.f_2_ary = fork_2q_ary

    def enable(self):
        if self.inq_ary[0].welt_py_fifo_basic_population() == 0:
            return False
        else:
            return True

    def invoke(self):
        #check
        print("fork actor invokes")

        tokenPass = []
        for eachQ in self.inq_ary:
            tokenPass.append(eachQ.welt_py_fifo_basic_read_direct())

        for i in range(len(tokenPass)):
            self.f_1_ary[i].welt_py_fifo_basic_write(copy.deepcopy(tokenPass[i]))
            self.f_2_ary[i].welt_py_fifo_basic_write(tokenPass[i])

    def terminate(self):
        pass