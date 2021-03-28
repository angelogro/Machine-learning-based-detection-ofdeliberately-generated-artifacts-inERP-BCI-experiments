#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:00:30 2020

@author: angelo
"""
import global_vars
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.multiclass import OneVsOneClassifier
import os
from wyrm import io

import bci_minitoolbox as bci
from bci_dataPrep import *
from bci_plotFuncs import *
from bci_classify import *
from bci_processing import *

import pandas as pd
import seaborn as sns
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
from wyrm.processing import lda_train

#%%
"""
Time intervals of epo data from which features are extracted
"""
# 3.2.4 Feaure Extraction - Features for Linear Discriminant Analysis
ivals = [[160, 190], [200,230],[240,270],[280,290],[300,310],[320,330],[340,350],
        [360,370],[380, 390], [400,430],[440,470],[480, 520]]

#%%
# 3.2.1 Data preprocessing
commonAverageChannelNums=list(map(lambda x:cindex[x],commonAverageChannels))
car_o=np.mean(epo_o[:,commonAverageChannelNums,:],axis=1)[:,np.newaxis,:]
epo_o-=car_o
car_a=np.mean(epo_a[:,commonAverageChannelNums,:],axis=1)[:,np.newaxis,:]
epo_a-=car_a
#%%
epo_o_target = epo_o[:,:,mrk_class_o==0]

epo = np.concatenate([epo_o_target,epo_a],axis=2)
mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a],axis=0)

#%%
# LDA Classifier: Artifact vs Target
# 3.3.2 Classification Results - Shrinkage LDA
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
for i,(k,v) in enumerate(cindexes.items()):
    if k in [1,5,9,31]:
        meanconf,stdconf = classifyTargetVsArtifact(clf,mrk_class_a,epo_o_target[:,v,:],epo_a[:,v,:],ivals,epo_t_o)
        print(f"{k} channels: Mean {meanconf} Std {stdconf}")
        xaxis = [artifactDict[x] for x in np.unique(mrk_class_a)]
        
        plt.figure(figsize=(10,6))
        df = pd.DataFrame(zip(xaxis*2,["TP"]*6+["TN"]*6,np.concatenate((meanconf[:,0,0],meanconf[:,1,1])),
                              np.concatenate((stdconf[:,0,0],stdconf[:,1,1]))),columns=["Artifact Type", "Type","Average Rate","std"])
        grouped_barplot(df, "Artifact Type", "Type", "Average Rate", "std",colors=plotcolours,
                        title="LDA with shrinkage - Classification Accuracy")  
        

#%%
# Multiclass
# 3.3.2 Classification Results - Shrinkage LDA
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
for i,(k,v) in enumerate(cindexes.items()):
    if k in [1,5,9,31]:
        meanconf,stdconf = classifyTargetVsArtifacts(clf,mrk_class,epo[:,v,:],ivals,epo_t_o)
        print(f"{k} channels: Mean {meanconf} Std {stdconf}")
        disp=ConfusionMatrixDisplay(meanconf, list(map(lambda x:abbrDict[x],np.unique(mrk_class))))
        plt.figure(figsize=(10,10))
        disp.plot(values_format='.2f',cmap='Blues')
                
#%%
# No Shrinkage
# LDA Classifier: Artifact vs Target 
# 3.3.2 Classification Results - Linear Discriminant Analysis
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage=1e-7)
for i,(k,v) in enumerate(cindexes.items()):
    if k in [1,5,9,31]:
        meanconf,stdconf = classifyTargetVsArtifact(clf,mrk_class_a,epo_o_target[:,v,:],epo_a[:,v,:],ivals,epo_t_o)
        xaxis = [artifactDict[x] for x in np.unique(mrk_class_a)]
        plt.figure(figsize=(10,6))
        print(f"{k} channels: Mean {meanconf} Std {stdconf}")
        df = pd.DataFrame(zip(xaxis*2,["TP"]*6+["TN"]*6,np.concatenate((meanconf[:,0,0],meanconf[:,1,1])),
                              np.concatenate((stdconf[:,0,0],stdconf[:,1,1]))),columns=["Artifact Type", "Type","Average Rate","std"])
        grouped_barplot(df, "Artifact Type", "Type", "Average Rate", "std",colors=plotcolours,
                        title="LDA - Classification Accuracy") 
#%%
# No Shrinkage
# Multiclass
# 3.3.2 Classification Results - Linear Discriminant Analysis
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage=1e-7)
meanconf,stdconf = classifyTargetVsArtifacts(clf,mrk_class,epo,ivals,epo_t_o)

disp=ConfusionMatrixDisplay(meanconf, list(map(lambda x:abbrDict[x],np.unique(mrk_class))))
plt.figure(figsize=(10,10))
disp.plot(values_format='.2f',cmap='Blues')

