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

#statRecd Control
record_stat = True

#threshold Control
weightPruneTH = 0.995
weightShareTH = 0.99
StructurePruneTH = 0.985

#Dropout Control
DORate = 0.5
dropoutDecay = 0.95

#GPU Control
GPUscope = '0,1'
GPU = '/cpu:1'

#Original shapes
MLPsingleShape = [32]#[32]
CNN2DsingleShape = [32,16,32]#[32,16,16]
MLPmultiShape = [32,16,8]#[32,16,8]
CNN2DmultiShape = [32,16,32,16,8]#[32,16,32,16,8]

#common functions
import numpy as np

def avgValAcc(ValAccHis):
    size = len(ValAccHis)
    valaccsum = np.sum(ValAccHis)
    avg = valaccsum/size
    return avg  
