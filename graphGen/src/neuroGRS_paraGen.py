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

from keras.layers import Dense,Conv2D
from extraCnnWeights import ExtraCnnWeights


class neuroGRS_paraGen:
    def __init__(self,layers,model):
        self.layers = layers
        self.model = model
        self.folder = 'extractedParas'
        self.paraGen()

    def paraGen(self):
        layerWeights = []
        for eachLayer in self.layers:
            layerWeights.append(eachLayer.get_weights())

        exr = ExtraCnnWeights(self.folder)

        for i in range(len(self.layers)):
            if isinstance(self.layers[i], Conv2D):
                exr.extractCnnLayerW(layerWeights[i], self.layers[i].name)
            if isinstance(self.layers[i], Dense):
                exr.extractDenseLayerW(layerWeights[i], self.layers[i].name)

if __name__ == "__main__":
    pass