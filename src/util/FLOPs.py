import tensorflow as tf
import keras.backend as K
from keras.models import model_from_json
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from keras import Sequential
from tensorflow import keras
from datasetPre import datasetPre


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
    return flops.total_float_ops, params.total_parameters

def ParaFlop(jsonfile,modeltype):
    K.clear_session()
    # load json and create model
    json_file = open(jsonfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    #model.load_weights(h5file)
    if modeltype == 'cnn':
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    #paras = model.count_params()

    sess = K.get_session()
    graph = sess.graph
    flops,paras = stats_graph(graph)
    K.clear_session()
    return flops,paras

def ParaFlopTest(jsonfile,h5file,modeltype,x,y):
    K.clear_session()
    # load json and create model
    json_file = open(jsonfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(h5file)
    if modeltype == 'cnn':
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    #scores =model.evaluate(x,y,verbose=0)
    #acc_val = scores[1]
    #print(acc_val)
    #paras = model.count_params()

    sess = K.get_session()
    graph = sess.graph
    flops,paras = stats_graph(graph)
    K.clear_session()
    return flops,paras

if __name__ == "__main__":
    dataPre = datasetPre(['v','04e1'])
    jsonfile = '../../../NeuroGRSoutputSave/outputs0602bw/modelinfo/mlpmulti_1004_e1_V{0}seed0model_original.json'
    h5file = '../../../NeuroGRSoutputSave/outputs0602bw/modelinfo/mlpmulti_1004_e1_V{0}seed0model_original_weights.h5'

    paras,flops = ParaFlopTest(jsonfile,h5file,'mlp',dataPre.mlptestX,dataPre.mlptestY)
    print(paras,flops)

