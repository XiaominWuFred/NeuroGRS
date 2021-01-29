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

'''
class metrics
author: Xiaomin Wu

calculate useful metrics from predicted results with given labels
 
'''
from sklearn.metrics import roc_auc_score
from sklearn.metrics.scorer import roc_auc_scorer
import numpy as np;
from builtins import int

class metrics(object):
    PPLP = 0; #predict positive label positive
    PPLN = 0; #predict positive label negative
    PNLP = 0; #predict negative label positive
    PNLN = 0; #predict negative label negative 
    sum = 0;
    
    def __init__(self):
        self.PNLN = 0;
        self.PNLP = 0;
        self.PPLN = 0;
        self.PPLP = 0;
        self.sum = 0;
    
    def CreatConfusionMatrix(self,Y_pre,Y_label):
        
        self.PNLN = 0;
        self.PNLP = 0;
        self.PPLN = 0;
        self.PPLP = 0;
        self.sum = 0;
        
        for i in range(len(Y_pre)):
            if(Y_label[i] == 1):
                if(Y_pre[i] != 1):
                    self.PNLP = self.PNLP + 1;
                else:
                    self.PPLP = self.PPLP + 1;
            else:
                if(Y_pre[i] != 1):
                    self.PNLN = self.PNLN + 1;
                else:
                    self.PPLN = self.PPLN + 1;
    
        self.sum = self.PPLN+self.PPLP+self.PNLN+self.PNLP;
        
    def TruePositives(self):
        return self.PPLP;
    
    def TrueNegatives(self):
        return self.PNLN;
    
    def FalsePositives(self): #type-1 error
        return self.PPLN;
    
    def FalseNegatives(self): #type-2 error
        return self.PNLP;
    
    def Accuracy(self):
        acc = (self.PPLP + self.PNLN)/self.sum;
        return acc;
    
    def Precision(self):
        return self.PPLP / (self.PPLP+self.PPLN);
    
    def PositiveRecall(self): #sensitivity
        return self.PPLP / (self.PPLP+self.PNLP);
    
    def NegativeRecall(self): #Specificity
        return self.PNLN/(self.PNLN+self.PPLN);
    
    def Fscore(self):
        return ((self.PPLP / (self.PPLP+self.PNLP)) + (self.PNLN/(self.PNLN+self.PPLN)))/2;
    
    # Y_pre_score is in probability; if using keras, obtain by model.predict_proba(x_test)
    def AUC2(self,Y_pre_score,Y_label): 
        Y_label = Y_label.astype(int);
        Y_pre_score = Y_pre_score[np.arange(Y_label.shape[0]),Y_label];
        return roc_auc_score(Y_label, Y_pre_score);
    
    def AUC1(self,Y_pre_score,Y_label): 
        Y_label = Y_label.astype(int);
        #Y_pre_score = Y_pre_score[np.arange(Y_label.shape[0]),Y_label];
        return roc_auc_score(Y_label, Y_pre_score);
    
    def printMetrics2(self,Y_pre,Y_label,Y_pre_score):
        self.CreatConfusionMatrix(Y_pre,Y_label);
        acc = self.Accuracy();
        precision = self.Precision();
        sensitivity = self.PositiveRecall();
        specificity = self.NegativeRecall();
        fscore = self.Fscore();
        roc_auc_score = self.AUC2(Y_pre_score, Y_label);
        print("acc:",acc,"\n",
              "precision:",precision,"\n",
              "sensitivity:",sensitivity,"\n",
              "specificity:",specificity,"\n",
              "fscore:",fscore,"\n",
              "roc_auc_score:",roc_auc_score,"\n");
        return acc, precision, sensitivity, specificity, fscore, roc_auc_score;
    
    def printMetrics1(self,Y_pre,Y_label,Y_pre_score):
        self.CreatConfusionMatrix(Y_pre,Y_label);
        acc = self.Accuracy();
        precision = self.Precision();
        sensitivity = self.PositiveRecall();
        specificity = self.NegativeRecall();
        fscore = self.Fscore();
        roc_auc_score = self.AUC1(Y_pre_score, Y_label);
        print("acc:",acc,"\n",
              "precision:",precision,"\n",
              "sensitivity:",sensitivity,"\n",
              "specificity:",specificity,"\n",
              "fscore:",fscore,"\n",
              "roc_auc_score:",roc_auc_score,"\n");
        return acc, precision, sensitivity, specificity, fscore, roc_auc_score;
        