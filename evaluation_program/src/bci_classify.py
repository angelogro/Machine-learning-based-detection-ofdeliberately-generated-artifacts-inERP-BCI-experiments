#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:20:01 2020

@author: angelo
"""

from scipy import signal
import numpy as np
import os
from global_vars import artifactDict
from sklearn.metrics import confusion_matrix
import bci_minitoolbox as bci
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate
from sklearn.metrics import make_scorer
from bci_classifiers import crossvalidation,train_LDAshrink,train_LDA,loss_weighted_error,crossvalidationDetailedLoss

def convertEpoToFeature(epo,epo_t,mrk_class,ivals):
    '''
    Description:
        Converts data to features of mean amplitude in specified intervals.
    Usage:
        feat, mrk_class = convertEpoToFeature(epo,epo_t,mrk_class,ivals)
    Parameters:
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ivals: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
    Returns:
        meanAmplFeature: a 2D array (features x epochs)
        mrk_class: a 1D array that assigns markers to classes (0, 1)
    '''
    assert epo.shape[2]==len(mrk_class)
    assert epo.shape[0]==len(epo_t)
    
    sampleIndices = [np.where((epo_t>=ivals[i][0])&(epo_t<=ivals[i][1])) for i in range (len(ivals))]
    meanAmpl=np.zeros((len(ivals),epo.shape[1],epo.shape[2]))

    for i in range(len(ivals)):
       meanAmpl[i]=np.mean(epo[sampleIndices[i],:,:].squeeze(),axis=0)
       
    meanAmplFeature = meanAmpl.reshape((len(ivals)*epo.shape[1]),-1)
    return meanAmplFeature,mrk_class
    

def calcAndValLDATemp(epo,epo_t,mrk_class,ival):
    '''
    Description:
        Crossvalidates a LDA and Shrinked-LDA classifier on temporal features. 
    Usage:
        feat, mrk_class = calcAndValLDATemp(epo,epo_t,mrk_class,ivals)
    Parameters:
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ival: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
    Returns:
        None
    '''
    #Convert time intervals to list of sample point number within an epoch recording
    sampleIndices = [np.where((epo_t>=ival[i][0])&(epo_t<=ival[i][1])) for i in range (len(ival))]
    
    meanAmpl=np.zeros((len(ival),epo.shape[1],epo.shape[2]))
    for i in range(len(ival)):
       meanAmpl[i]=np.mean(epo[sampleIndices[i],:,:].squeeze(),axis=0)
       print('***Interval '+str(ival[i])+' ms :')
       print('  Using common LDA   : ' +str(crossvalidation(train_LDA,meanAmpl[i],mrk_class,10,False) [0] ))
       print('  Using shrinked LDA : ' +str(crossvalidation(train_LDAshrink,meanAmpl[i],mrk_class,10,False)[0]))

def calcAndValLDATempSpat(epo, epo_t, mrk_class, ivals,returnParams=False,n_folds=9):  
    '''
    Description:
        Crossvalidates LDA and Shrinked-LDA classifier on spatio-temporal
        features returning the calculated Shrinked-LDA parameters
    Usage:
        paramsLDA = calcAndValLDATempSpat(epo,epo_t,mrk_class,ivals,returnParams)
    Parameters:
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ivals: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
        returnParams: boolean indicating if the parameters shall be returned
    Returns:
        paramsLDA: {optional} 1D array with tuples (weigth vector, bias) of all
        calculated classifiers
    '''
    meanAmplFeature,mrk_class = convertEpoToFeature(epo,epo_t,mrk_class,ivals)
    print('**** All features in one vector: ****')
    print('  Using common LDA   : {:5.1f}'.format(crossvalidation(train_LDA,meanAmplFeature,mrk_class,n_folds,False,False)[0]  ))
    errTeFP,errTeFN,paramsLDA = crossvalidationDetailedLoss(train_LDAshrink,meanAmplFeature,mrk_class,n_folds,False,returnParams)
    print('  Using shrinked LDA : FP: {:5.1f} , FN: {:5.1f}'.format(errTeFP,errTeFN))
    if returnParams:
        return paramsLDA
 
def calcAndValLDACSP(epo, epo_t, mrk_class, ivals,returnParams=False,verbose=False):  
    '''
    Description:
        Crossvalidates LDA and Shrinked-LDA classifier on spatio-temporal
        features returning the calculated Shrinked-LDA parameters
    Usage:
        paramsLDA = calcAndValLDATempSpat(epo,epo_t,mrk_class,ivals,returnParams)
    Parameters:
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ivals: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
        returnParams: boolean indicating if the parameters shall be returned
    Returns:
        paramsLDA: {optional} 1D array with tuples (weigth vector, bias) of all
        calculated classifiers
    '''
    meanAmplFeature,mrk_class = convertCSPToFeature(epo,epo_t,mrk_class,ivals)

    print('**** All features in one vector: ****')
    print('  Using common LDA   : {:5.1f}'.format(crossvalidationDetailedLoss(train_LDA,meanAmplFeature,mrk_class,3,verbose=verbose,returnParams=False)[0]  ))
    errTeFP,errTeFN,paramsLDA = crossvalidationDetailedLoss(train_LDAshrink,meanAmplFeature,mrk_class,3,verbose=verbose,returnParams=True)
    print('  Using shrinked LDA : FP: {:5.1f} , FN: {:5.1f}'.format(errTeFP,errTeFN))
    if returnParams:
        return paramsLDA
    

def convertCSPToFeature(epo,epo_t,mrk_class,ivals):
    '''
    Description:
        Converts data to features of mean amplitude in specified intervals.
    Usage:
        feat, mrk_class = convertEpoToFeature(epo,epo_t,mrk_class,ivals)
    Parameters:
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ivals: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
    Returns:
        meanAmplFeature: a 2D array (features x epochs)
        mrk_class: a 1D array that assigns markers to classes (0, 1)
    '''
    assert epo.shape[2]==len(mrk_class)
    assert epo.shape[0]==len(epo_t)
    
    sampleIndices = [np.where((epo_t>=ivals[i][0])&(epo_t<=ivals[i][1])) for i in range (len(ivals))]
    logVarAmpl=np.zeros((len(ivals),epo.shape[1],epo.shape[2]))
    for i in range(len(ivals)):
       logVarAmpl[i]=np.log(np.var(epo[sampleIndices[i],:,:].squeeze(),axis=0))
       
    meanAmplFeature = logVarAmpl.reshape((len(ivals)*epo.shape[1]),-1)
    return meanAmplFeature,mrk_class



def validateLDA(params,epo,epo_t,mrk_class,ivals):         
    '''
    Description:
        Validates artifact data with LDA parameters.
    Usage:
        meanError = validateLDA(params,epo,epo_t,mrk_class,ival)
    Parameters:
        params: 1D array with tuples (weigth vector, bias) of all calculated classifiers
        epo: 3D array of segmented signals (samples x channels x epochs)
        epo_t: 1D array of time points of epochs relative to marker (in ms)
        mrk_class: 1D array with class information for each marker
        ivals: 1D array with lists containing start and endtime of each interval
                e.g. [[0,100],[150,180],[330,400]]
        returnParams: boolean indicating if the parameters shall be returned
    Returns:
        meanError: averaged classification error over all set of parameters (in %)
    '''
    X,y = convertEpoToFeature(epo,epo_t,mrk_class,ivals)
    errTe = []
    for w,b in params:
        out = w.T.dot(X) - b
        errTe.append(np.sum(out>0)/y.shape*100)
    return np.mean(errTe)

def balanceClasses(epo,mrk_class,_type='upsample'):
    sample_0,sample_1 = sum(mrk_class==0),sum(mrk_class==1)
    if sample_0 > sample_1:
        mrk_class_ = np.concatenate((mrk_class,np.ones(sample_0-sample_1)))
        sample_indices = np.argwhere(mrk_class==1).squeeze()
        np.random.shuffle(sample_indices)
    else:
        mrk_class_ = np.concatenate((mrk_class,np.zeros(sample_1-sample_0)))
        sample_indices = np.argwhere(mrk_class==0).squeeze()
        np.random.shuffle(sample_indices)
    while epo.shape[2]<mrk_class_.shape[0]:
        epo = np.concatenate((epo,epo[:,:,sample_indices[:(mrk_class_.shape[0]-epo.shape[2])]]),axis=2)
    return epo,mrk_class_

def scoring_hit(y_true,y_pred):
    result_set = set(y_true)
    lab1=result_set.pop()
    lab2=result_set.pop()
    return sum((y_true==lab1)*(y_pred==lab1))/sum(y_true==lab1)
def scoring_correctrejection(y_true,y_pred):
    result_set = set(y_true)
    lab1=result_set.pop()
    lab2=result_set.pop()
    return sum((y_true==lab2)*(y_pred==lab2))/sum(y_true==lab2)
score_hit = make_scorer(scoring_hit)
score_rej = make_scorer(scoring_correctrejection)
scoring={'hit':score_hit,'rej':score_rej}

def classifyTargetVsArtifact(clf,mrk_class_a,epo_o_target,epo_a,ivals,epo_t_o,cv=20,verbose=True):
    meanconf,stdconf=[],[]
    for artifact_class in np.unique(mrk_class_a):
        confusion_matrices=[]
        mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a[mrk_class_a==artifact_class]],axis=0)
        epo = np.concatenate([epo_o_target,epo_a[:,:,mrk_class_a==artifact_class]],axis=2)
        meanAmplFeature,mrk_class = convertEpoToFeature(epo,epo_t_o,mrk_class,ivals)
        scores = cross_validate(clf, meanAmplFeature.T, mrk_class, cv=cv,return_estimator=True,scoring=scoring)
        if verbose:
            print(f"Artifact {artifactDict[artifact_class]}: \n hits {scores['test_hit']} \n rejs {scores['test_rej']}")
        
        for estimator in scores['estimator']:
            conf_mat = confusion_matrix(mrk_class,estimator.predict(meanAmplFeature.T),normalize='true')
            confusion_matrices.append(conf_mat)
        confusion_matrices = np.array(confusion_matrices)   
        meanconf.append(np.mean(confusion_matrices,axis=0))
        stdconf.append(np.std(confusion_matrices,axis=0))
        
    return np.array(meanconf),np.array(stdconf)

def classifyTargetVsArtifacts(clf,mrk_class,epo,ivals,epo_t_o,cv=20):
    confusion_matrices=[]

    meanAmplFeature,mrk_class = convertEpoToFeature(epo,epo_t_o,mrk_class,ivals)
    scores = cross_validate(clf, meanAmplFeature.T, mrk_class, cv=cv,return_estimator=True)
    for estimator in scores['estimator']:
        estimator.predict(meanAmplFeature.T)
        conf_mat = confusion_matrix(mrk_class,estimator.predict(meanAmplFeature.T),normalize='true')
        confusion_matrices.append(conf_mat)
    confusion_matrices = np.array(confusion_matrices)   
    meanconf=np.mean(confusion_matrices,axis=0)
    stdconf=np.std(confusion_matrices,axis=0)
    return meanconf,stdconf