#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:56:06 2020

@author: angelo
"""
from pathlib import Path
import sys
import os
evaluationFolder=Path(__file__).parents[1]
projectFolder = Path(__file__).parents[2]
measurementFolder = os.path.join(projectFolder,'data','experiment_data','eeg_data')
simulationInputFolder = os.path.join(projectFolder,'data','simulation_data','input','signal_data')
simulationOutputFolder = os.path.join(projectFolder,'data','simulation_data','output')
sys.path.insert(1,os.path.join(evaluationFolder,'lib'))

import numpy as np

#%% Preprocessing
highpass = 0.1
lowpass = 0
ival = [-100,1000]
ref_ival=[-100,0]
downsample_fac=10

#%%
commonAverageChannels = np.array(['F3', 'F4', 'P3', 'P4', 'F7',
       'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Pz', 'FC1', 'FC2',
       'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz'])
#%%

artifactDict={0:'Target',
              1:'Press feet',
              2:'Lift tongue',
              3:'Forced breath',
              4:'Clench teeth',
              5:'Move eyes',
              6:'Push breath',
              7:'Wrinkle nose',
              8:'Swallow',
              9:'Relax'}

abbrDict={0:'T',
              1:'PF',
              2:'LT',
              3:'Forced breath',
              4:'CT',
              5:'Move eyes',
              6:'PB',
              7:'WN',
              8:'SW',
              9:'Relax'}

plotcolours=['powderblue','lightskyblue']

clab = np.array(['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3',
    'P4', 'O1', 'O2', 'F7','F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2',
    'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz'])

cindex={j:i for i,j in enumerate(clab)}

clabs={31:clab,
       27:np.array(['F3', 'F4', 'C3', 'C4', 'P3',
    'P4', 'O1', 'O2', 'F7','F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2',
    'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'POz']),
       23:np.array(['F3', 'F4', 'C3', 'C4','O1', 'O2', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2',
    'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'POz']),
       19:np.array(['F3', 'F4', 'C3', 'C4','O1', 'O2', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
    'FC5', 'FC6', 'CP5', 'CP6', 'POz']),
       15:np.array(['F3', 'F4', 'C3', 'C4','T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
     'CP5', 'CP6', 'POz']),
       12:np.array(['F3', 'F4', 'C3', 'C4','T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz']),
       9:np.array(['C3', 'C4','T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']),
       7:np.array(['C3', 'C4','T7', 'T8', 'Cz', 'Fz','Pz']),
       5:np.array(['C3', 'C4','Cz','Fz', 'Pz']),
       3:np.array(['Cz','Fz', 'Pz']),
       1:np.array(['Cz'])#,
       #'frontal':np.array(['F3','F4','F7','F8','FC5','FC6']),
       #'1-frontal':np.array(['F3'])
    }
cindexes={k:[cindex[c] for c in v] for k,v in clabs.items()}

ivals_dic={12: [[160, 190], [200,230],[240,270],[280,290],[300,310],[320,330],[340,350],
        [360,370],[380, 390], [400,430],[440,470],[480, 520]],
           11: [[160, 230],[240,270],[280,290],[300,310],[320,330],[340,350],
        [360,370],[380, 390], [400,430],[440,470],[480, 520]],
           10: [[160, 230],[240,290],[300,310],[320,330],[340,350],
        [360,370],[380, 390], [400,430],[440,470],[480, 520]],
           9: [[160, 230],[240,290],[300,330],[340,350],
        [360,370],[380, 390], [400,430],[440,470],[480, 520]],
           8: [[160, 230],[240,290],[300,330],[340,350],
        [360,390], [400,430],[440,470],[480, 520]],
           7: [[240,290],[300,330],[340,350],
        [360,390], [400,430],[440,470],[480, 520]],
           6: [[300,330],[340,350],
        [360,390], [400,430],[440,470],[480, 520]],
           5: [[300,330],[340,350],
        [360,390], [400,430],[440,470]],
           4: [[300,350],[360,390], [400,430],[440,470]],
           3: [[300,350],[360,390], [400,430]],
           2: [[300,350],[360,390]],
           1: [[360,390]]
           }

samples_dic=[1000,500,250,150,100,75,50,30]