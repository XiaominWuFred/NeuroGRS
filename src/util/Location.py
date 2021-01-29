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
import csv
import numpy as np


def getLocation(path,dir):
    data = [];
    with open(dir+path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #i = 0;
        for row in spamreader:
            data.append(list(map(float,row[0].split(','))));       
        data = np.array(data);
                        
    return data;

def matchX(X,location):
    
    newX = [];
    for eachX in X:
        ary = np.zeros((308,331))
        for i in range(len(location)):
            roundL = np.round(location[i])
            ary[int(roundL[0])-1][int(roundL[1])-1] = eachX[i]
        
        newX.append(ary)
    
    newX = np.array(newX)
    newX = newX.reshape((X.shape[0],308,331,1))
    return newX

'''
#test
X = np.full((2,273),255)

location = getLocation('CELLXY_1004.csv')
nX = matchX(X, location)
plt.imshow(nX[0])
plt.show()
'''