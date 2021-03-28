#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:29:03 2021

@author: angelo
"""
import global_vars

import bci_minitoolbox as bci
from bci_dataPrep import epo_o,epo_t_o,mrk_class_o,epo_a,epo_t_a,mrk_class_a,clab,mnt
from bci_plotFuncs import *


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
# Extracting only target ERPs
epo_o_target = epo_o[:,:,mrk_class_o==0]
epo = np.concatenate([epo_o_target,epo_a],axis=2)
mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a],axis=0)


#%%
# Generation of scalp map plots of artifacts
arti_ivals =[[-100,0],[0,100],[100,200],[200,300],[300,400],[400,500],[500,600],[600,700]]
#%%
# 3.2.3 Artifacts - Figure 3.7
plotScalpmapsArtifact(epo_a, epo_t_a, clab, mrk_class_a, arti_ivals, mnt)
# 3.2.3 Artifacts - Figure 3.6
plotMeanArtifactSignals(epo_a, epo_t_a, clab, ['Fz','Cz','Pz'], mrk_class_a)
#%%
# Generation of Mean epochs of target/non-target
# 3.2.2 Oddball Task - Figure 3.4
plotMeanOddballSignals(epo_o,epo_t_o,clab,['Fz','Cz','Pz'],mrk_class_o)


#%%
# 3.2.2 Oddball Task - Figure 3.5
plotScalpmapsArtifact(epo_o, epo_t_o, clab, mrk_class_o, arti_ivals, mnt)

#%%
# Not used
# Plot data of two channels
# Extract relevant data (T = target, NT = NonTarget)
chans=['Cz','Pz']
time=300
plt.figure(figsize=(10,10))
for mrk_c in np.unique(mrk_class):
    first=epo[np.where(epo_t_o==time),clab==chans[0],mrk_class==mrk_c]
    second=epo[np.where(epo_t_o==time),clab==chans[1],mrk_class==mrk_c]
    plt.scatter(first,second,s=2,label=artifactDict[mrk_c])
plt.legend()
plt.title(f"Channels {chans[0]} and {chans[1]} @ timepoint {time} ms")
plt.xlabel(chans[0]); plt.ylabel(chans[1]);
plt.xlim((-10,10))  