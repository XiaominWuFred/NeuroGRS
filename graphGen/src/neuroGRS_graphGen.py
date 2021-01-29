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

from keras.layers import Dense, Conv2D, MaxPooling2D


class neuroGRS_graphGen:
    def __init__(self, modelname, rawLayers, DaCLayers, DaCName, type, layersAmount, sampleSize, sampleDim):
        self.modelname = modelname
        self.rawLayers = rawLayers  # list of raw layers including maxpool
        self.DaCLayers = DaCLayers  # list of layers conv or dense
        self.DaCName = DaCName
        self.type = type  # cnn or mlp
        self.layersAmount = layersAmount
        self.sampleSize = sampleSize
        self.sampleDim = sampleDim

        # variables:
        self.fifoCount = None
        self.actorCount = None
        self.fifos = []
        self.actors = []
        self.hyperParas = {}

    # write header
    def writeHeader(self, path):
        with open(path + "lide_c_" + self.modelname + "_graph.h", "w") as file:
            file.write("#ifndef _lide_c_" + self.modelname + "_graph_h\n"
                         "#define _lide_c_" + self.modelname + "_graph_h\n"
                       "#include <stdio.h>\n"
                       "#include <stdlib.h>\n"
                       "#include \"lide_c_basic.h\"\n"
                       "#include \"lide_c_actor.h\"\n"
                       "#include \"lide_c_fifo.h\"\n"
                       "#include \"lide_c_graph.h\"\n"
                       "#include \"lide_c_util.h\"\n"
                       "#define BUFFER_CAPACITY 16\n")

            actorCount = 0
            fifoCount = 0
            previousLayer = None
            for i in range(len(self.rawLayers)):
                if isinstance(self.rawLayers[i], Conv2D):
                    if actorCount == 0:
                        # first conv layer
                        actor = "ACTOR_READW_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_W2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_READB_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_B2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_READIN_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_IN2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_" + self.rawLayers[i].name + "2RELU "
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_RELUCNN_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1
                    else:
                        # TO EDIT:
                        actor = "ACTOR_READW_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_W2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_READB_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_B2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1
                        if isinstance(previousLayer, MaxPooling2D):

                            fifo = "FIFO_MP2" + self.rawLayers[i].name
                            self.fifos.append(fifo)
                            file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                            fifoCount = fifoCount + 1
                        else:

                            fifo = "FIFO_RELU2" + self.rawLayers[i].name
                            self.fifos.append(fifo)
                            file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                            fifoCount = fifoCount + 1

                        fifo = "FIFO_" + self.rawLayers[i].name + "2RELU "
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_RELUCNN_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                if isinstance(self.rawLayers[i], Dense):
                    if actorCount == 0:

                        actor = "ACTOR_READIN_DENSE_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_IN2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_READW_DENSE_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_W2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_READB_DENSE_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_B2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_DENSE_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_" + self.rawLayers[i].name + "2RELU "
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_RELUDEN_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1
                    elif isinstance(previousLayer, (Conv2D, MaxPooling2D)):
                        # first dense layer

                        actor = "ACTOR_READW_FLATTENDENSE_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_W2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_READB_FLATTENDENSE_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_B2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_FLATTENDENSE_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1
                        if isinstance(previousLayer, MaxPooling2D):

                            fifo = "FIFO_MP2" + self.rawLayers[i].name
                            self.fifos.append(fifo)
                            file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                            fifoCount = fifoCount + 1
                        else:

                            fifo = "FIFO_RELU2" + self.rawLayers[i].name
                            self.fifos.append(fifo)
                            file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                            fifoCount = fifoCount + 1

                        if i != len(self.rawLayers) - 1:

                            actor = "ACTOR_RELUDEN_" + self.rawLayers[i].name
                            self.actors.append(actor)
                            file.write("#define " + actor + " " + str(actorCount) + "\n")
                            actorCount = actorCount + 1

                            fifo = "FIFO_" + self.rawLayers[i].name + "2RELU "
                            self.fifos.append(fifo)
                            file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                            fifoCount = fifoCount + 1
                        else:

                            fifo = "FIFO_" + self.rawLayers[i].name + "2SOFTMAX "
                            self.fifos.append(fifo)
                            file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                            fifoCount = fifoCount + 1
                    else:

                        actor = "ACTOR_READW_DENSE_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_W2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_READB_DENSE_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_B2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1

                        actor = "ACTOR_DENSE_" + self.rawLayers[i].name
                        self.actors.append(actor)
                        file.write("#define " + actor + " " + str(actorCount) + "\n")
                        actorCount = actorCount + 1

                        fifo = "FIFO_RELU2" + self.rawLayers[i].name
                        self.fifos.append(fifo)
                        file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                        fifoCount = fifoCount + 1
                        if i != len(self.rawLayers) - 1:

                            actor = "ACTOR_RELUDEN_" + self.rawLayers[i].name
                            self.actors.append(actor)
                            file.write("#define " + actor + " " + str(actorCount) + "\n")
                            actorCount = actorCount + 1

                            fifo = "FIFO_" + self.rawLayers[i].name + "2RELU "
                            self.fifos.append(fifo)
                            file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                            fifoCount = fifoCount + 1
                        else:

                            fifo = "FIFO_" + self.rawLayers[i].name + "2SOFTMAX "
                            self.fifos.append(fifo)
                            file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                            fifoCount = fifoCount + 1

                if isinstance(self.rawLayers[i], MaxPooling2D):
                    actor = "ACTOR_MAXPOOL_" + self.rawLayers[i].name
                    self.actors.append(actor)
                    file.write("#define " + actor + " " + str(actorCount) + "\n")
                    actorCount = actorCount + 1

                    fifo = "FIFO_RELU2" + self.rawLayers[i].name
                    self.fifos.append(fifo)
                    file.write("#define " + fifo + " " + str(fifoCount) + "\n")
                    fifoCount = fifoCount + 1

                previousLayer = self.rawLayers[i]

            # write fixed actors softmax and writeout

            actor = "ACTOR_SOFTMAX"
            self.actors.append(actor)
            file.write("#define " + actor + " " + str(actorCount) + "\n")
            actorCount = actorCount + 1

            fifo = "FIFO_SM2WO"
            self.fifos.append(fifo)
            file.write("#define " + fifo + " " + str(fifoCount) + "\n")
            fifoCount = fifoCount + 1

            actor = "ACTOR_WRITEOUT"
            self.actors.append(actor)
            file.write("#define " + actor + " " + str(actorCount) + "\n")
            actorCount = actorCount + 1

            file.write("#define ACTOR_COUNT " + str(actorCount) + "\n")
            file.write("#define FIFO_COUNT " + str(fifoCount) + "\n")

            file.write("struct _lide_c_" + self.modelname + "_graph_context_struct;\n")
            file.write("typedef struct _lide_c_" + self.modelname + "_graph_context_struct "
                        "lide_c_" + self.modelname + "_graph_context_type;\n")

            file.write("lide_c_" + self.modelname + "_graph_context_type *lide_c_" + self.modelname +
                       "_graph_new(const char * InputsFile, \n")
            for eachName in self.DaCName:
                file.write("    const char* W" + eachName + ", const char* B" + eachName + ",\n")
            file.write("    const char* OutFile );\n")
            file.write(
                "void lide_c_" + self.modelname + "_graph_terminate(lide_c_" +
                self.modelname + "_graph_context_type *graph);\n")
            file.write(
                "void lide_c_" + self.modelname + "_graph_scheduler(lide_c_" +
                self.modelname + "_graph_context_type *graph);\n")
            file.write("#endif\n")
            file.close()

    # write source
    def writeSource(self, path):
        with open(path + "lide_c_" + self.modelname + "_graph.c", "w") as file:
            # write includes
            file.write("#include <stdio.h>\n")
            file.write("#include <stdlib.h>\n")
            file.write("#include \"lide_c_" + self.modelname + "_graph.h\"\n")
            if self.type == 'cnn':
                file.write("#include \"lide_c_conv2D.h\"\n")
                file.write("#include \"lide_c_conv2DHead.h\"\n")
                file.write("#include \"lide_c_flattenDense.h\"\n")
                file.write("#include \"lide_c_maxpool.h\"\n")
                file.write("#include \"lide_c_read2D.h\"\n")
                file.write("#include \"lide_c_reluCnn.h\"\n")
            file.write("#include \"lide_c_dense.h\"\n")
            file.write("#include \"lide_c_headDense.h\"\n")
            file.write("#include \"lide_c_read1D.h\"\n")
            file.write("#include \"lide_c_reluDense.h\"\n")
            file.write("#include \"lide_c_softmax.h\"\n")
            file.write("#include \"lide_c_writeOut.h\"\n")
            # write struct
            file.write("struct _lide_c_" + self.modelname + "_graph_context_struct {\n"
                                                            "#include \"lide_c_graph_context_type_common.h\"\n"
                                                            "};\n")

            # write macro
            for each in self.rawLayers:
                if isinstance(each, Conv2D):
                    file.write("#define FILTERDIM" + each.name + " " + str(each.kernel_size[0]) + "\n")
                    file.write("#define INPUTDIM" + each.name + " " + str(each.input_shape[1]) + "\n")
                    file.write("#define INPUTNUM" + each.name + " " + str(self.sampleSize) + "\n")
                    file.write("#define PICN" + each.name + " " + str(each.input_shape[3]) + "\n")
                    file.write("#define FILTERNUM" + each.name + " " + str(each.output_shape[3]) + "\n")
                    file.write("#define OUTDIM" + each.name + " " + str(each.output_shape[1]) + "\n")
                    file.write("#define OUTNUM" + each.name + " " + str(self.sampleSize * each.output_shape[3]) + "\n")
                if isinstance(each, MaxPooling2D):
                    file.write("#define MPSTEP" + each.name + " " + str(each.strides[0]) + "\n")
                    # file.write("#define MPDIM" + each.name + " " + str(each.pool_size[0]) + "\n")
                    file.write("#define OUTDIMMP" + each.name + " " + str(each.output_shape[1]) + "\n")
                    file.write("#define FILTERNUM" + each.name + " " + str(each.output_shape[3]) + "\n")
                if isinstance(each, Dense):
                    file.write("#define NODENUM" + each.name + " " + str(each.units) + "\n")
                    file.write("#define INPUTNUM" + each.name + " " + str(self.sampleSize) + "\n")
                    file.write("#define INPUTDIM" + each.name + " " + str(self.sampleDim) + "\n")

            # write func
            file.write("lide_c_" + self.modelname + "_graph_context_type *lide_c_" + self.modelname +
                       "_graph_new(const char * InputsFile, \n")
            for eachName in self.DaCName:
                file.write("    const char* W" + eachName + ", const char* B" + eachName + ",\n")
            file.write("    const char* OutFile ){\n")
            file.write("    int token_size;\n"
                       "    lide_c_" + self.modelname + "_graph_context_type * context = NULL;\n"
                        "    context = (lide_c_" + self.modelname + "_graph_context_type *)lide_c_util_malloc(sizeof(\n"
                        "        lide_c_" + self.modelname + "_graph_context_type));\n"
                         "    context->actor_count = ACTOR_COUNT;\n"
                         "    context->fifo_count = FIFO_COUNT;\n"
                         "    context->actors = (lide_c_actor_context_type **)lide_c_util_malloc(\n"
                         "        context->actor_count * sizeof(lide_c_actor_context_type *));\n"
                         "    context->fifos = (lide_c_fifo_pointer *)lide_c_util_malloc(\n"
                         "        context->fifo_count * sizeof(lide_c_fifo_pointer));\n"
                         "    context->descriptors = (char **)lide_c_util_malloc(context->actor_count * \n"
                         "        sizeof(char*));\n")
            # write FIFO define
            file.write("    token_size = sizeof(float*);\n")
            for each in self.fifos:
                file.write("    context->fifos[" + each + "] = "
                      "(lide_c_fifo_pointer)lide_c_fifo_new(BUFFER_CAPACITY, token_size);\n")

            # write actor define
            actorCount = 0
            previousLayer = None
            for i in range(len(self.rawLayers)):
                if isinstance(self.rawLayers[i], Conv2D):
                    if actorCount == 0:
                        # first conv layer
                        file.write("    context->actors[" + "ACTOR_READW_" + self.rawLayers[i].name + "] = "
                                  "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                  "context->fifos[" + "FIFO_W2" +
                                   self.rawLayers[i].name + "]," +
                                   "W" + self.rawLayers[i].name + ","
                                  "FILTERNUM" + self.rawLayers[i].name + "*PICN" +
                                   self.rawLayers[i].name +
                                   "*FILTERDIM" + self.rawLayers[i].name +
                                   "*FILTERDIM" + self.rawLayers[i].name + "));\n")
                        file.write("    context->actors[ACTOR_READB_" + self.rawLayers[i].name + "] = "
                                     "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                     "context->fifos[FIFO_B2" +
                                   self.rawLayers[i].name + "], B" +
                                   self.rawLayers[i].name + ",FILTERNUM" + self.rawLayers[i].name + "));\n")
                        file.write("    context->actors[ACTOR_READIN_" + self.rawLayers[i].name + "] = "
                                    "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                    "context->fifos[FIFO_IN2" +
                                   self.rawLayers[i].name + "],InputsFile,"
                                                            "INPUTNUM" + self.rawLayers[i].name + "*PICN" +
                                   self.rawLayers[i].name +
                                   "*INPUTDIM" + self.rawLayers[i].name + "*INPUTDIM" + self.rawLayers[
                                       i].name + "));\n")
                        file.write("    context->actors[ACTOR_" + self.rawLayers[i].name + "] = "
                                   "(lide_c_actor_context_type*)(lide_c_conv2DHead_new("
                                   "context->fifos[FIFO_IN2" +
                                   self.rawLayers[i].name + "], "
                                                            "context->fifos[FIFO_W2" + self.rawLayers[i].name + "],"
                                                            "context->fifos[FIFO_B2" +
                                   self.rawLayers[i].name + "],"
                                                            "context->fifos[FIFO_" + self.rawLayers[i].name + "2RELU],"
                                                              "INPUTDIM" +
                                   self.rawLayers[i].name + ", INPUTNUM" + self.rawLayers[i].name +
                                   ", FILTERDIM" + self.rawLayers[i].name + ", "
                                                                            "FILTERNUM" + self.rawLayers[
                                       i].name + ", OUTDIM" + self.rawLayers[i].name +
                                   ", OUTNUM" + self.rawLayers[i].name + ", PICN" + self.rawLayers[i].name + "));\n")
                        file.write("    context->actors[ACTOR_RELUCNN_" + self.rawLayers[i].name + "] = "
                                   "(lide_c_actor_context_type*)(lide_c_reluCnn_new("
                                   "context->fifos[FIFO_" +
                                   self.rawLayers[i].name + "2RELU],"
                                                            "context->fifos[FIFO_RELU2" + self.rawLayers[
                                       i + 1].name + "],"
                                                     "INPUTNUM" + self.rawLayers[i].name +
                                   ",OUTDIM" + self.rawLayers[i].name +
                                   ",FILTERNUM" + self.rawLayers[i].name + "));\n")
                        actorCount = actorCount + 5

                    else:
                        # non first conv layer
                        file.write("    context->actors[" + "ACTOR_READW_" + self.rawLayers[i].name + "] = "
                                  "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                  "context->fifos[" + "FIFO_W2" +
                                   self.rawLayers[i].name + "]," +
                                   "W" + self.rawLayers[i].name + ","
                                                                  "FILTERNUM" + self.rawLayers[i].name + "*PICN" +
                                   self.rawLayers[i].name +
                                   "*FILTERDIM" + self.rawLayers[i].name +
                                   "*FILTERDIM" + self.rawLayers[i].name + "));\n")
                        file.write("    context->actors[ACTOR_READB_" + self.rawLayers[i].name + "] = "
                                     "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                     "context->fifos[FIFO_B2" +
                                   self.rawLayers[i].name + "], B" +
                                   self.rawLayers[i].name + ",FILTERNUM" + self.rawLayers[i].name + "));\n")
                        file.write("    context->actors[ACTOR_RELUCNN_" + self.rawLayers[i].name + "] = "
                                   "(lide_c_actor_context_type*)(lide_c_reluCnn_new("
                                   "context->fifos[FIFO_" +
                                   self.rawLayers[i].name + "2RELU],"
                                                            "context->fifos[FIFO_RELU2" + self.rawLayers[
                                       i + 1].name + "],"
                                                     "INPUTNUM" + self.rawLayers[i].name +
                                   ",OUTDIM" + self.rawLayers[i].name +
                                   ",FILTERNUM" + self.rawLayers[i].name + "));\n")
                        actorCount = actorCount + 3
                        if isinstance(previousLayer, MaxPooling2D):
                            # layer follows maxpool
                            file.write("    context->actors[ACTOR_" + self.rawLayers[i].name + "] = "
                                       "(lide_c_actor_context_type*)(lide_c_conv2d_new("
                                       "context->fifos[FIFO_MP2" +
                                       self.rawLayers[i].name + "], "
                                        "context->fifos[FIFO_W2" + self.rawLayers[i].name + "],"
                                        "context->fifos[FIFO_B2" +
                                       self.rawLayers[i].name + "],"
                                                                "context->fifos[FIFO_" + self.rawLayers[
                                           i].name + "2RELU],"
                                                     "INPUTDIM" + self.rawLayers[i].name + ", INPUTNUM" +
                                       self.rawLayers[i].name +
                                       ", FILTERDIM" + self.rawLayers[i].name + ", "
                                                                                "FILTERNUM" + self.rawLayers[
                                           i].name + ", OUTDIM" + self.rawLayers[i].name +
                                       ", OUTNUM" + self.rawLayers[i].name + ", PICN" + self.rawLayers[
                                           i].name + "));\n")
                            actorCount = actorCount + 1
                        else:
                            # layer not follow maxpool
                            file.write("    context->actors[ACTOR_" + self.rawLayers[i].name + "] = "
                                        "(lide_c_actor_context_type*)(lide_c_conv2d_new("
                                       "context->fifos[FIFO_RELU2" +
                                       self.rawLayers[i].name + "], "
                                                                "context->fifos[FIFO_W2" + self.rawLayers[i].name + "],"
                                                                "context->fifos[FIFO_B2" +
                                       self.rawLayers[i].name + "],"
                                                                "context->fifos[FIFO_" + self.rawLayers[
                                           i].name + "2RELU],"
                                                     "INPUTDIM" + self.rawLayers[i].name + ", INPUTNUM" +
                                       self.rawLayers[i].name +
                                       ", FILTERDIM" + self.rawLayers[i].name + ", "
                                                                                "FILTERNUM" + self.rawLayers[
                                           i].name + ", OUTDIM" + self.rawLayers[i].name +
                                       ", OUTNUM" + self.rawLayers[i].name + ", PICN" + self.rawLayers[
                                           i].name + "));\n")
                            actorCount = actorCount + 1

                if isinstance(self.rawLayers[i], Dense):
                    if actorCount == 0:
                        # head dense layer
                        file.write("    context->actors[ACTOR_READIN_DENSE_" + self.rawLayers[i].name + "] = "
                                        "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                        "context->fifos[FIFO_IN2" +
                                   self.rawLayers[i].name + "], "
                                                            "InputsFile,"
                                                            "INPUTNUM" + self.rawLayers[i].name +
                                   "*INPUTDIM" + self.rawLayers[i].name + "));\n")
                        file.write("    context->actors[ACTOR_READW_DENSE_" + self.rawLayers[i].name + "] = "
                                       "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                        "context->fifos[FIFO_W2" +
                                   self.rawLayers[i].name + "], "
                                                            "W" + self.rawLayers[i].name + ","
                                                                                           "INPUTDIM" + self.rawLayers[
                                       i].name + "*"
                                                 "NODENUM" + self.rawLayers[i].name + "));\n")
                        file.write("    context->actors[ACTOR_READB_DENSE_" + self.rawLayers[i].name + "] = "
                                       "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                       "context->fifos[FIFO_B2" +
                                   self.rawLayers[i].name + "],"
                                                            "B" + self.rawLayers[i].name + ","
                                                                                           "NODENUM" + self.rawLayers[
                                       i].name + "));\n")
                        file.write("    context->actors[ACTOR_DENSE_" + self.rawLayers[i].name + "] = "
                                        "(lide_c_actor_context_type*)(lide_c_headDense_new("
                                        "context->fifos[FIFO_IN2" +
                                   self.rawLayers[i].name + "],"
                                                            "context->fifos[FIFO_W2" + self.rawLayers[i].name + "],"
                                                            "context->fifos[FIFO_B2" +
                                   self.rawLayers[i].name + "],"
                                                            "context->fifos[FIFO_" + self.rawLayers[i].name + "2RELU],"
                                                            "INPUTNUM" +
                                   self.rawLayers[i].name + ",INPUTDIM" + self.rawLayers[i].name + ", "
                                                               "NODENUM" +
                                   self.rawLayers[i].name + "));\n")
                        file.write("    context->actors[ACTOR_RELUDEN_" + self.rawLayers[i].name + "] = "
                                   "(lide_c_actor_context_type*)(lide_c_reluDense_new("
                                   "context->fifos[FIFO_" +
                                   self.rawLayers[i].name + "2RELU],"
                                                            "context->fifos[FIFO_RELU2" + self.rawLayers[
                                       i + 1].name + "],"
                                                     "INPUTNUM" + self.rawLayers[i].name + ",NODENUM" + self.rawLayers[
                                       i].name + "));\n")
                        actorCount = actorCount + 5

                    elif isinstance(previousLayer, (Conv2D, MaxPooling2D)):
                        # first dense layer after conv
                        file.write("    context->actors[ACTOR_READW_FLATTENDENSE_" + self.rawLayers[i].name + "] = "
                                      "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                      "context->fifos[FIFO_W2" +
                                   self.rawLayers[i].name + "], "
                                                            "W" + self.rawLayers[i].name + ","
                                                                                           "FILTERNUM" + self.rawLayers[
                                       i - 1].name + "*OUTDIM" + self.rawLayers[i - 1].name +
                                   "*OUTDIM" + self.rawLayers[i - 1].name +
                                   "*NODENUM" + self.rawLayers[i].name + "));\n")
                        file.write("    context->actors[ACTOR_READB_FLATTENDENSE_" + self.rawLayers[i].name + "] = "
                                  "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                  "context->fifos[FIFO_B2" +
                                   self.rawLayers[i].name + "], "
                                                            "B" + self.rawLayers[i].name + ",NODENUM" + self.rawLayers[
                                       i].name + "));\n")
                        actorCount = actorCount + 2
                        if isinstance(previousLayer, MaxPooling2D):
                            # dense after maxpool
                            if i != len(self.rawLayers) - 1:
                                # not last dense
                                file.write("    context->actors[ACTOR_FLATTENDENSE_" + self.rawLayers[i].name + "] = "
                                            "(lide_c_actor_context_type*)(lide_c_flattenDense_new("
                                            "context->fifos[FIFO_MP2" +
                                           self.rawLayers[i].name + "],"
                                                                    "context->fifos[FIFO_W2" + self.rawLayers[
                                               i].name + "],"
                                                         "context->fifos[FIFO_B2" + self.rawLayers[i].name + "],"
                                                         "context->fifos[FIFO_" +
                                           self.rawLayers[i].name + "2RELU],"
                                                                    "INPUTNUM" + self.rawLayers[i].name + ","
                                                                                                          "FILTERNUM" +
                                           self.rawLayers[i - 1].name +
                                           "*OUTDIM" + self.rawLayers[i - 1].name +
                                           "*OUTDIM" + self.rawLayers[i - 1].name + ","
                                                                                    "NODENUM" + self.rawLayers[i].name +
                                           ",OUTDIM" + self.rawLayers[i - 1].name + ", "
                                                                                    "FILTERNUM" + self.rawLayers[
                                               i - 1].name + "));\n")
                                file.write("    context->actors[ACTOR_RELUDEN_" + self.rawLayers[i].name + "] = "
                                           "(lide_c_actor_context_type*)(lide_c_reluDense_new("
                                           "context->fifos[FIFO_" +
                                           self.rawLayers[i].name + "2RELU],"
                                                                    "context->fifos[FIFO_RELU2" + self.rawLayers[
                                               i + 1].name + "],"
                                                             "INPUTNUM" + self.rawLayers[i].name +
                                           ",NODENUM" + self.rawLayers[i].name + "));\n")
                                actorCount = actorCount + 2
                            else:
                                # last dense
                                file.write("    context->actors[ACTOR_FLATTENDENSE_" + self.rawLayers[i].name + "] = "
                                            "(lide_c_actor_context_type*)(lide_c_flattenDense_new("
                                            "context->fifos[FIFO_MP2" +
                                           self.rawLayers[i].name + "],"
                                                                    "context->fifos[FIFO_W2" + self.rawLayers[
                                               i].name + "],"
                                                         "context->fifos[FIFO_B2" + self.rawLayers[i].name + "],"
                                                         "context->fifos[FIFO_" +
                                           self.rawLayers[i].name + "2SOFTMAX],"
                                                                    "INPUTNUM" + self.rawLayers[i].name + ","
                                                                                                          "FILTERNUM" +
                                           self.rawLayers[i - 1].name +
                                           "*OUTDIM" + self.rawLayers[i - 1].name +
                                           "*OUTDIM" + self.rawLayers[i - 1].name + ","
                                                                                    "NODENUM" + self.rawLayers[i].name +
                                           ",OUTDIM" + self.rawLayers[i - 1].name + ", "
                                                                                    "FILTERNUM" + self.rawLayers[
                                               i - 1].name + "));\n")
                                actorCount = actorCount + 1
                        else:
                            # dense after conv
                            if i != len(self.rawLayers) - 1:
                                # not last dense
                                file.write("    context->actors[ACTOR_FLATTENDENSE_" + self.rawLayers[i].name + "] = "
                                            "(lide_c_actor_context_type*)(lide_c_flattenDense_new("
                                            "context->fifos[FIFO_RELU2" +
                                           self.rawLayers[i].name + "],"
                                                                    "context->fifos[FIFO_W2" + self.rawLayers[
                                               i].name + "],"
                                                         "context->fifos[FIFO_B2" + self.rawLayers[i].name + "],"
                                             "context->fifos[FIFO_" +
                                           self.rawLayers[i].name + "2RELU],"
                                                                    "INPUTNUM" + self.rawLayers[i].name + ","
                                                                                                          "FILTERNUM" +
                                           self.rawLayers[i - 1].name +
                                           "*OUTDIM" + self.rawLayers[i - 1].name +
                                           "*OUTDIM" + self.rawLayers[i - 1].name + ","
                                                                                    "NODENUM" + self.rawLayers[i].name +
                                           ",OUTDIM" + self.rawLayers[i - 1].name + ", "
                                                                                    "FILTERNUM" + self.rawLayers[
                                               i - 1].name + "));\n")
                                file.write("    context->actors[ACTOR_RELUDEN_" + self.rawLayers[i].name + "] = "
                                           "(lide_c_actor_context_type*)(lide_c_reluDense_new("
                                           "context->fifos[FIFO_" +
                                           self.rawLayers[i].name + "2RELU],"
                                                                    "context->fifos[FIFO_RELU2" + self.rawLayers[
                                               i + 1].name + "],"
                                                             "INPUTNUM" + self.rawLayers[i].name +
                                           ",NODENUM" + self.rawLayers[i].name + "));\n")
                                actorCount = actorCount + 2
                            else:
                                # last dense
                                file.write("    context->actors[ACTOR_FLATTENDENSE_" + self.rawLayers[i].name + "] = "
                                            "(lide_c_actor_context_type*)(lide_c_flattenDense_new("
                                            "context->fifos[FIFO_RELU2" +
                                           self.rawLayers[i].name + "],"
                                                                    "context->fifos[FIFO_W2" + self.rawLayers[
                                               i].name + "],"
                                                         "context->fifos[FIFO_B2" + self.rawLayers[i].name + "],"
                                                         "context->fifos[FIFO_" +
                                           self.rawLayers[i].name + "2SOFTMAX],"
                                                                    "INPUTNUM" + self.rawLayers[i].name + ","
                                                                                                          "FILTERNUM" +
                                           self.rawLayers[i - 1].name +
                                           "*OUTDIM" + self.rawLayers[i - 1].name +
                                           "*OUTDIM" + self.rawLayers[i - 1].name + ","
                                                                                    "NODENUM" + self.rawLayers[i].name +
                                           ",OUTDIM" + self.rawLayers[i - 1].name + ", "
                                                                                    "FILTERNUM" + self.rawLayers[
                                               i - 1].name + "));\n")
                                actorCount = actorCount + 1


                    else:
                        # other hidden dense layer
                        if i != len(self.rawLayers) - 1:
                            # not last dense
                            file.write("    context->actors[ACTOR_READW_DENSE_" + self.rawLayers[i].name + "] = "
                                       "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                       "context->fifos[FIFO_W2" +
                                       self.rawLayers[i].name + "], "
                                                                "W" + self.rawLayers[i].name + ","
                                                                "NODENUM" +
                                       self.rawLayers[i-1].name + "*"
                                                                "NODENUM" + self.rawLayers[i].name + "));\n")
                            file.write("    context->actors[ACTOR_READB_DENSE_" + self.rawLayers[i].name + "] = "
                                       "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                       "context->fifos[FIFO_B2" +
                                       self.rawLayers[i].name + "],"
                                                                "B" + self.rawLayers[i].name + ","
                                                                                               "NODENUM" +
                                       self.rawLayers[i].name + "));\n")
                            file.write("    context->actors[ACTOR_DENSE_" + self.rawLayers[i].name + "] = "
                                     "(lide_c_actor_context_type*)(lide_c_dense_new("
                                     "context->fifos[FIFO_RELU2" +
                                       self.rawLayers[i].name + "],"
                                                                "context->fifos[FIFO_W2" + self.rawLayers[i].name + "],"
                                                                "context->fifos[FIFO_B2" +
                                       self.rawLayers[i].name + "],"
                                                                "context->fifos[FIFO_" + self.rawLayers[
                                           i].name + "2RELU],"
                                                     "INPUTNUM" + self.rawLayers[i].name +
                                       ",NODENUM" + self.rawLayers[i-1].name + ", "
                                                     "NODENUM" + self.rawLayers[i].name + "));\n")
                            file.write("    context->actors[ACTOR_RELUDEN_" + self.rawLayers[i].name + "] = "
                                       "(lide_c_actor_context_type*)(lide_c_reluDense_new("
                                       "context->fifos[FIFO_" +
                                       self.rawLayers[i].name + "2RELU],"
                                                                "context->fifos[FIFO_RELU2" + self.rawLayers[
                                           i + 1].name + "],"
                                                         "INPUTNUM" + self.rawLayers[i].name + ",NODENUM" +
                                       self.rawLayers[i].name + "));\n")
                            actorCount = actorCount + 4

                        else:
                            # last dense
                            file.write("    context->actors[ACTOR_READW_DENSE_" + self.rawLayers[i].name + "] = "
                                       "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                       "context->fifos[FIFO_W2" +
                                       self.rawLayers[i].name + "], "
                                                                "W" + self.rawLayers[i].name + ","
                                                                                               "NODENUM" +
                                       self.rawLayers[i-1].name + "*"
                                                                "NODENUM" + self.rawLayers[i].name + "));\n")
                            file.write("    context->actors[ACTOR_READB_DENSE_" + self.rawLayers[i].name + "] = "
                                       "(lide_c_actor_context_type*)(lide_c_read1D_new("
                                       "context->fifos[FIFO_B2" +
                                       self.rawLayers[i].name + "],"
                                                                "B" + self.rawLayers[i].name + ","
                                                                                               "NODENUM" +
                                       self.rawLayers[i].name + "));\n")
                            file.write("    context->actors[ACTOR_DENSE_" + self.rawLayers[i].name + "] = "
                                     "(lide_c_actor_context_type*)(lide_c_dense_new("
                                     "context->fifos[FIFO_RELU2" +
                                       self.rawLayers[i].name + "],"
                                                                "context->fifos[FIFO_W2" + self.rawLayers[i].name + "],"
                                                                "context->fifos[FIFO_B2" +
                                       self.rawLayers[i].name + "],"
                                                                "context->fifos[FIFO_" + self.rawLayers[
                                           i].name + "2SOFTMAX],"
                                                     "INPUTNUM" + self.rawLayers[i].name + ",NODENUM" + self.rawLayers[
                                           i-1].name + ", "
                                                     "NODENUM" + self.rawLayers[i].name + "));\n")
                            actorCount = actorCount + 3

                if isinstance(self.rawLayers[i], MaxPooling2D):
                    # maxpool
                    file.write("    context->actors[ACTOR_MAXPOOL_" + self.rawLayers[i].name + "] = "
                               "(lide_c_actor_context_type*)(lide_c_maxpool_new("
                               "context->fifos[FIFO_RELU2" +
                               self.rawLayers[i].name + "],"
                                                        "context->fifos[FIFO_MP2" + self.rawLayers[i + 1].name + "],"
                                                        "MPSTEP" +
                               self.rawLayers[i].name + ", OUTNUM" + self.rawLayers[i - 1].name +
                               ",OUTDIM" + self.rawLayers[i - 1].name + "));\n")
                    actorCount = actorCount + 1

                previousLayer = self.rawLayers[i]

            # write softmax actor
            file.write("    context->actors[ACTOR_SOFTMAX] = "
                       "(lide_c_actor_context_type*)(lide_c_softmax_new("
                       "context->fifos[FIFO_" + self.rawLayers[len(self.rawLayers) - 1].name + "2SOFTMAX],"
                       "context->fifos[FIFO_SM2WO],"
                       "INPUTNUM" +
                       self.rawLayers[len(self.rawLayers) - 1].name +
                       ",NODENUM" + self.rawLayers[len(self.rawLayers) - 1].name + "));\n")
            actorCount = actorCount + 1

            # write writeout actor
            file.write("    context->actors[ACTOR_WRITEOUT] = "
                       "(lide_c_actor_context_type *)(lide_c_writeout_new("
                       "OutFile,context->fifos[FIFO_SM2WO],"
                       "INPUTNUM" + self.rawLayers[len(self.rawLayers) - 1].name +
                       ",NODENUM" + self.rawLayers[len(self.rawLayers) - 1].name + "));\n")
            actorCount = actorCount + 1

            print("written actors: " + str(actorCount))

            # write fixed contents
            file.write("    context->scheduler = (lide_c_graph_scheduler_ftype)\n")
            file.write("        lide_c_" + self.modelname + "_graph_scheduler;\n")
            file.write("    return context;\n")
            file.write("}\n")
            file.write(
                "void lide_c_" + self.modelname + "_graph_scheduler(lide_c_" +
                self.modelname + "_graph_context_type *context){\n")
            file.write(
                "    lide_c_util_simple_scheduler(context->actors, context->actor_count, context->descriptors);\n")
            file.write("    return;\n")
            file.write("}\n")
            file.close()

    def writeMakeGraph(self, path):

        with open(path + "Makefile", "w") as file:
            file.write("" + self.modelname + ": lide_c_" + self.modelname + "_graph.c\n"
                        "	g++ -g -o lide_c_" + self.modelname + "_graph.o  -c lide_c_" +
                       self.modelname + "_graph.c \\\n"
                        "	-I. -I../actor -I../includes -I../../cnnnsight \\\n\n")
            file.write("libr: lide_c_" + self.modelname + "_graph.o\n"
                        "	ar rcs libgraph.a lide_c_" + self.modelname + "_graph.o\n\n")
            file.write("all: " + self.modelname + " libr\n\n"
                        ".PHONY: clean\n\n"
                        "clean:\n"
	                    "	rm lide_c_" + self.modelname + "_graph.o libgraph.a\n")

            file.close()

    # write driver
    def writeDriver(self, path, backPath):
        with open(path + "driver.c", "w") as file:
            file.write("#include <stdio.h>\n")
            file.write("#include <stdlib.h>\n")
            file.write("#include <unistd.h>\n")
            file.write("#include <time.h>\n")
            file.write("#include \"lide_c_util.h\"\n")
            file.write("#include \"lide_c_" + self.modelname + "_graph.h\"\n")
            file.write("int main(int argc, char **argv) {\n")
            if self.type == 'cnn':
                file.write("    const char* InputsFile = \"" + backPath + "testInputXCNN.csv\";\n")
            else:
                file.write("    const char* InputsFile = \"" + backPath + "testInputXDEN.csv\";\n")
            file.write("    const char* OutFile = \"../result.csv\";\n")
            for eachName in self.DaCName:
                file.write(
                    "    const char* W" + eachName + "=\"" + backPath + "" + eachName + "Weight.csv\";\n")
                file.write(
                    "    const char* B" + eachName + " = \"" + backPath + "" + eachName + "Bias.csv\";\n")

            file.write("    lide_c_graph_context_type *graph = NULL;\n"
                       "    graph = (lide_c_graph_context_type*)lide_c_" + self.modelname + "_graph_new(InputsFile,\n")
            for eachName in self.DaCName:
                file.write("    W" + eachName + ", B" + eachName + ",\n")
            file.write("    OutFile);\n")
            #file.write("	double time_spent = 0.0;\n    clock_t begin = clock();\n")
            file.write("    graph->scheduler(graph);\n")
            #file.write("	clock_t end = clock();\n    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;\n    printf(\"Graph takes %f ms\\n\", time_spent*1000);\n")
            file.write("    return 0;\n"
                       "}\n")

            file.close()

if __name__ == "__main__":
    modelname = 'test'
    rawLayers = []
    DaCLayers = []
    type = 'cnn'
    layersAmount = 5
    gen = neuroGRS_graphGen(modelname, rawLayers, DaCLayers, type, layersAmount)
    gen.writeHeader()
