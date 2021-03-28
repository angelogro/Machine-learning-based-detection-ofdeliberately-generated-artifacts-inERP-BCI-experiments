#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:22:59 2021

@author: angelo
"""

from global_vars import simulationInputFolder
from bci_processing import saveMeanArtifactSignals
from bci_dataPrep import epo_a,epo_t_a,mrk_class_a

# 4.1.4 Artifact Modelling

# Generation of Mean epochs of artifacts for later use as simulation input
saveMeanArtifactSignals(epo_a,epo_t_a,mrk_class_a,n_batches=8,path=simulationInputFolder)

#plotMeanArtifactSignals(epo_a,epo_t_a,clab,['Cz','Pz'],mrk_class_a)
