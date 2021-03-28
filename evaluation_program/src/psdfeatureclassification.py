#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:11:55 2021

@author: angelo
"""


#!/usr/bin/env python3
import global_vars
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate
from sklearn.metrics import confusion_matrix,make_scorer
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.multiclass import OneVsOneClassifier
from covar import cov_shrink_ss
import os

from wyrm import io

import bci_minitoolbox as bci
from bci_dataPrep import epo_o,mrk_class_o,mrk_class_a,epo_a,epo_t_a,epo_t_o,clab,mnt
from bci_plotFuncs import *
from bci_classify import *
from bci_processing import *

import pandas as pd
import seaborn as sns
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter

                                                                     
#%%

# Extraction of target stimulus epochs
epo_o_target = epo_o[:,:,mrk_class_o==0]

# Concatenating target stimuli and artifact epochs into one matrix
epo = np.concatenate([epo_o_target,epo_a],axis=2)
mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a],axis=0)

#%%

# 3.2.4 Feature Extraction - PSD features

frequency_bands=[[0,3],[10,20],[21,25],[26,30],[31,35],[36,40],[41,50]]  
f, psd = signal.welch(epo, fs=100,nperseg=1024,axis=0,scaling='spectrum')   
psd = np.log(psd)

#%%
# Features of artifacts
mean_features_a=[]
for artifact in np.unique(mrk_class_a):
    mean_feature_a=[]
    psd_=psd[:,:,mrk_class==artifact]
    for band in frequency_bands: 
        idx_range = np.where((f>=band[0])&(f<=band[1]))[0]
        mean_feature_a.append(np.mean(psd_[idx_range,:,:],axis=0))
    mean_features_a.append(np.array(mean_feature_a))

# Features of target
psd_=psd[:,:,mrk_class==0]
mean_feature_o=[]
for band in frequency_bands: 
    idx_range = np.where((f>=band[0])&(f<=band[1]))[0]
    mean_feature_o.append(np.mean(psd_[idx_range,:,:],axis=0))    
mean_feature_o=np.array(mean_feature_o)        

#%%

# Create scoring functions to get results from individual folds in the
# upcoming cross-validation


def scoring_hit(y_true,y_pred):
    result_set = set(y_true)
    try:
        lab1=result_set.pop()
    except KeyError:
        return 0
    return sum((y_true==lab1)*(y_pred==lab1))/sum(y_true==lab1)

def scoring_correctrejection(y_true,y_pred):
    result_set = set(y_true)
    lab1=result_set.pop()
    try:
        lab2=result_set.pop()
    except KeyError:
        return 0
    
    return sum((y_true==lab2)*(y_pred==lab2))/sum(y_true==lab2)

def target(y,y_):
    return len(y[y==set(y).pop()])

def artifact(y,y_):
    return len(y)-len(y[y==set(y).pop()])


    
#%%
scoring_shrink={'target':target,'artifact':artifact,'hit':scoring_hit,'rej':scoring_correctrejection}
#%% Target vs Artifact classification
# 3.3.2 Classification Results - Power Spectral Density Features
folds=20
sample_vect =  np.arange(10,mean_feature_o.shape[2],20)
clf = train_LDAshrink

# Shuffle the data
mean_feature_o=np.swapaxes(mean_feature_o,0,2)
np.random.shuffle(mean_feature_o)
mean_feature_o=np.swapaxes(mean_feature_o,0,2)
meanconf,stdconf=[],[]

for (k,v) in cindexes.items():
    
    if k in [1,5,9,31]:   

        max_target_avg_scores=[]
        max_artifact_avg_scores=[]
        max_artifact_var_scores=[]
        max_target_var_scores=[]
        gammas=[]
        for i,artifact_class in enumerate(np.unique(mrk_class_a)):
            artifact_avg_scores=[]
            artifact_var_scores=[]
            target_avg_scores=[]
            target_var_scores=[]
            gammas=[]
            for numTSamples in sample_vect:
                mrk_class_ = np.concatenate([np.zeros(numTSamples),np.ones(mrk_class_a[mrk_class_a==artifact_class].shape[0])],axis=0)
                meanfeatures=np.concatenate((mean_feature_o[:,:,:numTSamples],mean_features_a[i]),axis=2)
                meanfeatures=meanfeatures[:,v,:].reshape(mrk_class_.shape[0],-1)

                scores=crossvalidationDetailedLoss(clf,meanfeatures.T,mrk_class_,scoring=scoring_shrink,verbose=False,folds=folds)
                
    
                target_avg_score = sum(scores['test_target']*scores['test_hit'])/sum(scores['test_target'])
                target_var_score = np.var(scores['test_hit'])
                artifact_avg_score = sum(scores['test_artifact']*scores['test_rej'])/sum(scores['test_artifact'])
                artifact_var_score = np.var(scores['test_rej'])

                target_var_scores.append(target_var_score)
                target_avg_scores.append(target_avg_score)
                artifact_avg_scores.append(artifact_avg_score)
                artifact_var_scores.append(artifact_var_score)
                gammas.append(np.mean(scores['gamma']))
                
            # Uncomment and comment the next line to obtain the best score
            # which is obtained when not using all target samples
            #maxscoreindex=np.argmax(np.array(target_avg_scores)+np.array(artifact_avg_scores))
            maxscoreindex=len(sample_vect)-1
            max_target_avg_scores.append(target_avg_scores[maxscoreindex])
            max_artifact_avg_scores.append(artifact_avg_scores[maxscoreindex])
            max_artifact_var_scores.append(artifact_var_scores[maxscoreindex])
            max_target_var_scores.append(target_var_scores[maxscoreindex])
            
            plt.figure()
            ax = plt.subplot(111)
            plt.semilogy()
            plt.plot(sample_vect,artifact_avg_scores,label='Artifact accuracy')
            plt.plot(sample_vect,target_avg_scores,label='Target accuracy')
            plt.plot(sample_vect,gammas,label='Shrinkage')
            
            ax.set_title(f"PSD features - {k} channels - {artifactDict[artifact_class]}")
            plt.xlabel("# samples")
            plt.ylabel("class. accuracy / shrinkage")
            plt.grid(which='major',axis='both')
            plt.grid(which='minor',axis='both')
            plt.legend()
            plt.show()
            
         
        xaxis = [abbrDict[x] for x in np.unique(mrk_class_a)]
        df = pd.DataFrame(zip(xaxis*2,["TP"]*6+["TN"]*6,max_target_avg_scores+max_artifact_avg_scores,
                              max_target_var_scores+max_artifact_var_scores),columns=["Artifact Type", "Type","Average Rate","std"])
        grouped_barplot(df, "Artifact Type", "Type", "Average Rate", "std",colors=plotcolours,
                        title=f"LDA shrinkage with PSD features - Classification Accuracy ")
        print(df)
            
#%%
score_hit = make_scorer(scoring_hit)
score_rej = make_scorer(scoring_correctrejection)
score_target = make_scorer(target)
score_artifact =make_scorer(artifact)
scoring={'target':score_target,'artifact':score_artifact,'hit':score_hit,'rej':score_rej}


#%% Target vs Artifact classification on each single electrode
# Using sklearn function
cv=20
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
validation_errors=np.zeros((31,len(np.unique(mrk_class_a))))
for v in range(31):
    validation_error=[]
    artifact_avg_scores=[]
    artifact_var_scores=[]
    target_avg_scores=[]
    target_var_scores=[]
    for i,artifact_class in enumerate(np.unique(mrk_class_a)):
    
        mrk_class_ = np.concatenate([np.zeros(mean_feature_o.shape[2]),np.ones(mrk_class_a[mrk_class_a==artifact_class].shape[0])],axis=0)
        meanfeatures=np.concatenate((mean_feature_o,mean_features_a[i]),axis=2)
        meanfeatures=meanfeatures[:,v,:].reshape(mrk_class_.shape[0],-1)
        scores = cross_validate(clf,meanfeatures , mrk_class_, cv=cv,return_estimator=True,scoring=scoring)
        
        target_avg_score = sum(scores['test_target']*scores['test_hit'])/sum(scores['test_target'])
        target_var_score = np.var(scores['test_hit'])
        artifact_avg_score = sum(scores['test_artifact']*scores['test_rej'])/sum(scores['test_artifact'])
        artifact_var_score = np.var(scores['test_rej'])

        validation_errors[v,i]=1-np.mean((artifact_avg_score,target_avg_score))
#%%
# Plot the validation error for each single channel on a scalp map for all artifact types
for i,art in enumerate(np.unique(mrk_class_a)):
    plt.figure()
    plt.title("Channelwise validation error - "+artifactDict[art])
    bci.scalpmap(mnt,validation_errors[:,i],clim=[0,np.max(validation_errors)],cb_label='[a.u.]',cmap='GnBu')       
#%%
# Shrinkage investigation            
            
# Shrinkage vs target samples
gammas=[]
for numTSamples in sample_vect:
    mean_features=mean_feature_o[:,:,:numTSamples]
    mean_features = mean_features.reshape(-1,numTSamples)
    mean_features=mean_features.T.copy(order='C')
    c,g=cov_shrink_ss(mean_features)
    gammas.append(g)
#%%    
plt.figure()
plt.plot(sample_vect,gammas)
plt.semilogy()
plt.title("Shrinkage of covariance matrix (target samples)")
plt.xlabel("# samples")
plt.ylabel("shrinkage")
plt.grid(which='major',axis='both')
plt.grid(which='minor',axis='both')

#%%
# Shrinkage vs artifact sample
clf = train_LDAshrink

for i,artifact_class in enumerate([1]):#enumerate(np.unique(mrk_class_a)):
    gammas=[]
    sample_vect = range(10,sum(mrk_class_o==0),20)
    for numTSamples in sample_vect:
        mrk_class_ = np.concatenate([np.zeros(numTSamples),np.ones(sum(mrk_class_a==artifact_class))],axis=0)
        meanfeatures=np.concatenate((mean_feature_o[:,:,:numTSamples],mean_features_a[i][:,:,:]),axis=2)
        meanfeatures=meanfeatures.reshape(numTSamples+sum(mrk_class_a==artifact_class),-1)
        scores=crossvalidationDetailedLoss(clf,meanfeatures.T,mrk_class_,scoring=scoring_shrink,verbose=False)
                
        gammas.append(np.mean(scores['gamma']))
    plt.figure()
    plt.title(artifactDict[artifact_class])
    plt.plot(sample_vect,gammas)
    plt.semilogy()
    plt.xlabel("# samples")
    plt.ylabel("shrinkage")
    plt.grid(which='major',axis='both')
    plt.grid(which='minor',axis='both')
