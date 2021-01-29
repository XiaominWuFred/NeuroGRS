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

import keras
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from keras.regularizers import l2
from common import *
from tfRefresh import *


class modelPre(object):
    def __init__(self,D_Row = None,D_Column = None):
        #For CNN model input dimension
        self.D_Row = D_Row
        self.D_Column = D_Column
        #required output array of model name
        self.model_name = []
        #required instruction on construction
        self.run()

    def run(self):
        #Model_2
        model_shape = CNN2DmultiShape
        cnnmulti = Sequential()
        cnnmulti.add(Conv2D(model_shape[0], (2, 2), activation='relu', input_shape=(self.D_Row, self.D_Column, 1), name="conv1"))
        cnnmulti.add(MaxPooling2D(pool_size=(2, 2)))
        cnnmulti.add(Conv2D(model_shape[1], (2, 2), activation='relu', name="conv2"))
        cnnmulti.add(Dropout(DORate))
        cnnmulti.add(Flatten())
        cnnmulti.add(Dense(model_shape[2], activation='linear', use_bias=True, name="dense1"))
        cnnmulti.add(Activation('relu'))
        cnnmulti.add(Dropout(DORate))
        cnnmulti.add(Dense(model_shape[3], activation='linear', use_bias=True, name='dense2'))
        cnnmulti.add(Activation('relu'))
        cnnmulti.add(Dropout(DORate))
        cnnmulti.add(Dense(model_shape[4], activation='linear', use_bias=True, name='dense3'))
        cnnmulti.add(Activation('relu'))
        cnnmulti.add(Dropout(DORate))
        cnnmulti.add(Dense(2, activation='softmax', use_bias=True, name='dense4'))
        cnnmulti.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        #Save Model_2
        #self.model_name.append('cnnmulti')
        #saveModel(cnnmulti,'cnnmulti')


        #Model_4
        mlpsingle = keras.Sequential([
            keras.layers.Dense(MLPsingleShape[0], activation=tf.nn.relu, use_bias=True, name='fst'),
            keras.layers.Dropout(0.5, name='dropout'),
            keras.layers.Dense(2, activation=tf.nn.softmax, use_bias=True, name='snd')
        ])
        mlpsingle.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        #Save_Model_4
        #self.model_name.append('mlpsingle')
        #saveModel(mlpsingle, 'mlpsingle')


        #Model_1
        cnnsingle = Sequential()
        cnnsingle.add(Conv2D(CNN2DsingleShape[0], (2, 2), activation='relu', input_shape=(self.D_Row, self.D_Column, 1), name="conv1"))
        cnnsingle.add(MaxPooling2D(pool_size=(2, 2)))
        cnnsingle.add(Conv2D(CNN2DsingleShape[1], (2, 2), activation='relu', name="conv2"))
        cnnsingle.add(Dropout(DORate))
        cnnsingle.add(Flatten())
        cnnsingle.add(Dense(CNN2DsingleShape[2], activation='linear', use_bias=True, name="dense1"))
        cnnsingle.add(Activation('relu'))
        cnnsingle.add(Dropout(DORate))
        cnnsingle.add(Dense(2, activation='softmax', use_bias=True, name='dense2'))
        cnnsingle.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        #Save Model_1
        #self.model_name.append('cnnsingle')
        #saveModel(cnnsingle,'cnnsingle')


        #Model_3
        mlpmulti = keras.Sequential([
            keras.layers.Dense(MLPmultiShape[0], activation=tf.nn.relu, use_bias=True,  name='fst'),
            keras.layers.Dense(MLPmultiShape[1], activation=tf.nn.relu, use_bias=True,  name='snd'),
            keras.layers.Dense(MLPmultiShape[2], activation=tf.nn.relu, use_bias=True,  name='trd'),
            keras.layers.Dropout(0.5, name='dropout'),
            keras.layers.Dense(2, activation=tf.nn.softmax, use_bias=True,
                               name='fth')
        ])
        mlpmulti.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        #Save Model_3
        self.model_name.append('mlpmulti')
        saveModel(mlpmulti, 'mlpmulti')




if __name__ == "__main__":
    modelPre = modelPre(17,17)
    print(modelPre.model_name)