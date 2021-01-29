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
from keras import backend as K


def refreshTF(model,model_name,file_name):
    model_json = model.to_json()
    with open('../../runtime/'+model_name+file_name + "temModel.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('../../runtime/'+model_name+file_name + "tmpWeights.h5")
    #print("Saved temp model to disk \nclear section")
    K.clear_session()
    del model
    # load json and create model
    json_file = open('../../runtime/'+model_name+file_name + "temModel.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('../../runtime/'+model_name+file_name + "tmpWeights.h5")
    #print("Loaded temp model from disk")
    return model

def saveModel(model,model_name):
    model_json = model.to_json()
    with open('../../runtime/'+model_name + ".json", "w") as json_file:
        json_file.write(model_json)

def saveWeights(model):
    model.save_weights('../../runtime/TQtmpWeights.h5')

def loadWeights(model):
    model.load_weights('../../runtime/TQtmpWeights.h5')

def loadModel(model_name):
    json_file = open('../../runtime/'+model_name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model

def loadTrainedModel(model_name,file_name):
    # load json and create model
    json_file = open('../../outputs/modelinfo/' + model_name + file_name + "model_original.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('../../outputs/modelinfo/' + model_name + file_name + "model_original_weights.h5")
    #print("Loaded temp model from disk")
    return model

def loadGRSModel(model_name,file_name):
    # load json and create model
    json_file = open('../../outputs/modelinfo/' + model_name + file_name + "model_GRS_pruned.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('../../outputs/modelinfo/' + model_name + file_name + "model_GRS_weights.h5")
    #print("Loaded temp model from disk")
    return model

def loadTModel(model_name,file_name):
    # load json and create model
    json_file = open('../../outputs/modelinfo/' + model_name + file_name + "model_T_pruned.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('../../outputs/modelinfo/' + model_name + file_name + "model_T_weights.h5")
    #print("Loaded temp model from disk")
    return model
