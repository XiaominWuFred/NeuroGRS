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

import numpy as np

def weightCutor_convRandom(Wbf,WAft):
    index = np.random.choice(Wbf[0].shape[3])
    
    Wbf[0] = np.delete(Wbf[0], index, 3)
    Wbf[1] = np.delete(Wbf[1], index, 0)
    WAft[0] = np.delete(WAft[0],index,2)
    
    return Wbf,WAft,Wbf[0].shape[3]

def weightCutor_convWM(Wbf,WAft):
    #np.save('Wbf.npy',Wbf)
    absWbf = np.abs(Wbf)
    filterSum = np.sum(absWbf[0],(0,1,2)) #magnitude based on comming weights to a unit
    index = np.argmin(filterSum)
 
    Wbf[0] = np.delete(Wbf[0], index, 3)
    Wbf[1] = np.delete(Wbf[1], index, 0)
    WAft[0] = np.delete(WAft[0],index,2)
    
    return Wbf,WAft,Wbf[0].shape[3]


def weightCutor_CDRandom(Wbf,WAft,flattenSize):
    index = np.random.choice(Wbf[0].shape[3])
    Wbf[0] = np.delete(Wbf[0], index, 3)
    Wbf[1] = np.delete(Wbf[1], index, 0)
    start = flattenSize*(index)
    end = start+flattenSize
    WAft[0] = np.delete(WAft[0],np.arange(start,end),0)
    
    return Wbf,WAft,Wbf[0].shape[3]

def weightCutor_CDWM(Wbf,WAft,flattenSize):
    absWbf = np.abs(Wbf)
    filterSum = np.sum(absWbf[0],(0,1,2))
    index = np.argmin(filterSum)
    Wbf[0] = np.delete(Wbf[0], index, 3)
    Wbf[1] = np.delete(Wbf[1], index, 0)
    start = flattenSize*(index)
    end = start+flattenSize
    WAft[0] = np.delete(WAft[0],np.arange(start,end),0)
    
    return Wbf,WAft,Wbf[0].shape[3]


def weightCutorRandom(Wbf,WAft):
    random = True #control random or weight magnitude
    if(not random):
        wAbs = np.absolute(Wbf[0]);
        columnSum = np.sum(wAbs, axis = 0)
        index = np.argmin(columnSum)
    else:
        wAbs = np.absolute(Wbf[0]);
        columnSum = np.sum(wAbs, axis = 0)
        length = len(columnSum)
        index = np.random.choice(length,1)[0]
             
    Wbf[0] = np.delete(Wbf[0], index, 1)
    Wbf[1] = np.delete(Wbf[1], index, 0)
    WAft[0] = np.delete(WAft[0],index,0)
    
    return Wbf,WAft,Wbf[0].shape[1]

def weightCutorWM(Wbf,WAft):
    random = False #control random or weight magnitude
    if(not random):
        wAbs = np.absolute(Wbf[0]);
        columnSum = np.sum(wAbs, axis = 0)
        index = np.argmin(columnSum)
    else:
        wAbs = np.absolute(Wbf[0]);
        columnSum = np.sum(wAbs, axis = 0)
        length = len(columnSum)
        index = np.random.choice(length,1)[0]
             
    Wbf[0] = np.delete(Wbf[0], index, 1)
    Wbf[1] = np.delete(Wbf[1], index, 0)
    WAft[0] = np.delete(WAft[0],index,0)
    
    return Wbf,WAft,Wbf[0].shape[1]

def weightCutorIn(W,thrs):
    wAbs0 = np.absolute(W[0])
    wAbs1 = np.absolute(W[1])
    
    W[0][wAbs0<thrs] = 0
    W[1][wAbs1<thrs] = 0
    remainWN = np.count_nonzero(W[0])
    remainWN = remainWN + np.count_nonzero(W[1])
    totalWN = W[0].shape[0]*W[0].shape[1] + W[1].shape[0]
    deletedWN = totalWN - remainWN
    return W,deletedWN,totalWN,remainWN

def weightCutorInconv(W,thrs):
    
    wAbs0 = np.zeros_like(W[0])
    wAbs1 = np.zeros_like(W[1])
    
    for i in range(W[0].shape[0]):
        for j in range(W[0].shape[1]):
            for k in range(W[0].shape[2]):
                wAbs0[i][j][k] = np.absolute(W[0][i][j][k]);
    
    for i in range(W[0].shape[0]):
        for j in range(W[0].shape[1]):
            for k in range(W[0].shape[2]):
                W[0][i][j][k][wAbs0[i][j][k]<thrs] = 0;

    wAbs1 = np.absolute(W[1])
    W[1][wAbs1<thrs] = 0
    
    remainWN = 0
    for i in range(W[0].shape[0]):
        for j in range(W[0].shape[1]):
            for k in range(W[0].shape[2]):
                remainWN = remainWN + np.count_nonzero(W[0][i][j][k])
    
    remainWN = remainWN + np.count_nonzero(W[1])
    
    totalWN1 = W[0].shape[0]*W[0].shape[1]*W[0].shape[2]*W[0].shape[3]
    totalWN2 = W[1].shape[0]
    totalWN = totalWN1 + totalWN2
    deletedWN = totalWN - remainWN
    return W,deletedWN,totalWN,remainWN

def weightShare(W,digit):
    
    W[0] = np.round(W[0],digit)
    W[1] = np.round(W[1],digit)
    unique = np.unique(W[0])
    uni = np.unique(W[1])
    un = np.concatenate((unique,uni))
    unique = np.unique(un)
    return W,unique

def weightShareconv(W,digit):
    
    W[0] = np.round(W[0],digit)
    W[1] = np.round(W[1],digit)
    unique = np.unique(W[0])
    uni = np.unique(W[1])
    un = np.concatenate((unique,uni))
    unique = np.unique(un)
    return W,unique