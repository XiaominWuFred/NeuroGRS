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


class designSelection(object):
    def __init__(self):
        #inputs
        self.results = None
        #variable
        #outputs
        self.model = None

    def load(self,results):
        self.results = results

    def pick(self,choice):
        models = []
        metrics = []
        if choice == 'minpara':
            for each in self.results:
                models.append(self.results[each][0])
                metrics.append(self.results[each][1])
            idx = np.argmin(metrics)
            self.model = models[idx]

        if choice == 'minflops':
            for each in self.results:
                models.append(self.results[each][0])
                metrics.append(self.results[each][2])
            idx = np.argmin(metrics)
            self.model = models[idx]

        if choice == 'maxtestacc':
            for each in self.results:
                models.append(self.results[each][0])
                metrics.append(self.results[each][3])
            idx = np.argmax(metrics)
            self.model = models[idx]

        if choice == 'maxvalacc':
            for each in self.results:
                models.append(self.results[each][0])
                metrics.append(self.results[each][4])
            idx = np.argmax(metrics)
            self.model = models[idx]

        return self.model

