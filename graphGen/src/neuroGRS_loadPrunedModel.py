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

from keras.models import model_from_json


class neuroGRS_loadPrunedModel:
    def __init__(self,model_name,file_name,path):
        self.model_name = model_name
        self.file_name = file_name
        self.path = path
        self.model_P = None
        self.model_O = None
        self.loadTrainedModel()
        self.loadOriginalModel()

    def loadTrainedModel(self):
        # load json and create model
        json_file = open(self.path + self.model_name + self.file_name + "model_GRS_pruned.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(self.path + self.model_name + self.file_name + "model_GRS_weights.h5")
        self.model_P = model

    def loadOriginalModel(self):
        # load json and create model
        json_file = open(self.path + self.model_name + self.file_name + "model_original.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(self.path + self.model_name + self.file_name + "model_original_weights.h5")
        self.model_O = model

if __name__ == "__main__":
    pass