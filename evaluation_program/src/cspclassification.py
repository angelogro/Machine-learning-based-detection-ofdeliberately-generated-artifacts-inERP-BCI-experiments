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
from bci_dataPrep import epo_o,mrk_class_o,mrk_class_a,epo_a,epo_t_a,epo_t_o,clab,mnt,prepareData,data_list,data_list_art
from bci_plotFuncs import *
from bci_classify import *
from bci_processing import *

import pandas as pd
import seaborn as sns
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter


#%%
# For plotting PSD related data, no prior downsampling is required

epo_o,epo_t_o,mrk_class_o,clab,mnt = prepareData(data_list,downsample_factor=1,highPassCutOff=highpass,lowPassCutOff=lowpass,ival=ival,
                                                                     ref_ival=ref_ival,reject_voltage=0,nRemovePCA = 0,
                                                                     performpca=False)  
epo_a,epo_t_a,mrk_class_a,clab,mnt = prepareData(data_list_art,downsample_factor=1,highPassCutOff=highpass,lowPassCutOff=lowpass,ival=ival,
                                                                     ref_ival=ref_ival,reject_voltage=0,nRemovePCA = 0,performpca=False
                                                                     )

#%%

# 3.2.4 Time intervals used for feature extraction
ivals = [[160, 190], [200,230],[240,270],[280,290],[300,310],[320,330],[340,350],
        [360,370],[380, 390], [400,430],[440,470],[480, 520]]

#%%

# 3.2.1 Substraction of common average reference voltages
commonAverageChannelNums=list(map(lambda x:cindex[x],commonAverageChannels))
car_o=np.mean(epo_o[:,commonAverageChannelNums,:],axis=1)[:,np.newaxis,:]
epo_o-=car_o
car_a=np.mean(epo_a[:,commonAverageChannelNums,:],axis=1)[:,np.newaxis,:]
epo_a-=car_a
#%%

# Extraction of target stimulus epochs
epo_o_target = epo_o[:,:,mrk_class_o==0]

# Concatenating target stimuli and artifact epochs into one matrix
epo = np.concatenate([epo_o_target,epo_a],axis=2)
mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a],axis=0)


#%%
# Prepare CSP data
# 3.2.4 Feature Extraction - CSP features
# Figure 3.8

for art in np.unique(mrk_class_a):
    plotPSDfromEpoMulticlass(epo,mrk_class,clab,'Cz',[0,art],'artifact',1000,frange=[0,50])
    
#%%
# 3.2.4 Feature Extraction - CSP features
# Figure 3.8
    
plotPSDR2fromEpoMulticlass(epo,mrk_class,clab,clab,1000,frange=[0,50])
#%%
# 3.2.4 Feature Extraction - CSP features

frequency_bands=[[[15,20],[20,35],[35,50]],
                 [[20,30],[30,40],[40,50]],
                 [[10,20],[30,50],[22,28]],
                 [[15,20],[25,30],[40,50]],
                 [[15,25],[30,35],[35,50]],
                 [[10,20],[20,30],[30,40],[40,50]]]    


#%%
# 3.2.4 Feature Extraction - CSP features

epo_fr={}
for bands,artifact in zip(frequency_bands,np.unique(mrk_class_a)):
    epo_fr[artifact]=[]

    for band in bands:
        epo_o_b,epo_t_o,mrk_class_o,clab,mnt = prepareData(data_list,downsample_factor=1,highPassCutOff=highpass,lowPassCutOff=lowpass,ival=ival,
                                                                             ref_ival=ref_ival,reject_voltage=1000,nRemovePCA = 0,bandpass=[1,40],
                                                                             performpca=False) 
        epo_a_b,epo_t_a,mrk_class_a,clab,mnt = prepareData(data_list_art,downsample_factor=1,highPassCutOff=highpass,lowPassCutOff=lowpass,ival=ival,
                                                                         ref_ival=ref_ival,reject_voltage=1000,nRemovePCA = 0,bandpass=[1,40],performpca=False,
                                                                         )
        epo_o_target_b = epo_o_b[:,:,mrk_class_o==0]

        epo_fr[artifact].append(np.concatenate([epo_o_target_b,epo_a_b],axis=2))
    mrk_class = np.concatenate([np.zeros(epo_o_target_b.shape[2]),mrk_class_a],axis=0)
#%%
# 3.3.2 Classification Results - Common Spatial Patterns
    
numCSP=4
mrk_class = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a],axis=0)
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage=1e-8)
nFolds=10
for i,(k,v) in enumerate(cindexes.items()):
    
    if k in [5,9,31]:     
        meanconf,stdconf=[],[]
        for idx,artifact in enumerate(np.unique(mrk_class_a)):
            fold_scores_art=[]
            fold_scores_target=[]
            mrk_class_ = np.concatenate([np.zeros(epo_o_target.shape[2]),mrk_class_a[mrk_class_a==artifact]],axis=0)
            epo_x=epo[:,:,((mrk_class==0) | (mrk_class==artifact))]
            epo_x=epo_x[:,v,:]
            for ff in range(nFolds):
                
                meanAmplFeatures = []
                meanAmplFeatures_te = []
                
                nDim, _,nSamples = epo_x.shape
                inter = np.round(np.linspace(0, nSamples, num=nFolds + 1))
                perm = np.random.permutation(nSamples)
                idxTe = perm[int(inter[ff]):int(inter[ff + 1]) + 1]
                idxTr = np.setdiff1d(range(nSamples), idxTe)
                
                for epo_ in epo_fr[artifact]:
                    epo_=epo_[:,:,((mrk_class==0) | (mrk_class==artifact))]
                    epo_=epo_[:,v,:]
                    epo_fold,mrk_class_fold = epo_[:,:,idxTr],mrk_class_[idxTr]
                    epo_fold_te,mrk_class_fold_te = epo_[:,:,idxTe],mrk_class_[idxTe]
                    W,_,d = calculate_csp(epo_fold,mrk_class_fold )
    
                    # Calculate maximum difference in eigenvalues of the two classes
                    ddif = np.abs(d-(1-d))
                    
                    # Get the indexes of the 4 maximum eigenvalues   
                    eigidx = np.argsort(ddif)[-numCSP:]
                    
                    # Spatial filtering of signals
                    CSPs = apply_spatial_filter(epo_fold,W[:,eigidx])
                    AllCSP = apply_spatial_filter(epo_fold,W)
                    A = np.dot(np.dot(np.cov(epo_fold.reshape(epo_fold.shape[1],-1)),W),np.linalg.inv(np.cov(AllCSP.reshape(epo_fold.shape[1],-1))))

                    epoCSP = np.abs(signal.hilbert(CSPs,axis = 0))
                    
                    aveCSPsT = np.mean(epoCSP[:,:,mrk_class_fold==0],axis=2)
                    aveCSPsArtifact = np.mean(epoCSP[:,:,mrk_class_fold==artifact],axis=2) 
    
                    meanAmplFeature,mrk_class_fold = convertCSPToFeature(CSPs,epo_t_o,mrk_class_fold,ivals)              
                                    
                    CSPs_te = apply_spatial_filter(epo_fold_te,W[:,eigidx])
                    epoCSP_te = np.abs(signal.hilbert(CSPs_te,axis = 0))
    
                    meanAmplFeature_te,mrk_class_fold_te = convertCSPToFeature(CSPs_te,epo_t_o,mrk_class_fold_te,ivals)
                    meanAmplFeatures_te.append(meanAmplFeature_te)
                    meanAmplFeatures.append(meanAmplFeature)

                meanAmplFeatures = np.array(meanAmplFeatures)
                meanAmplFeatures=meanAmplFeatures.reshape(-1,meanAmplFeatures.shape[2])
                meanAmplFeatures_te = np.array(meanAmplFeatures_te)
                meanAmplFeatures_te=meanAmplFeatures_te.reshape(-1,meanAmplFeatures_te.shape[2])
                    
                    
                clf.fit(meanAmplFeatures.T,mrk_class_fold)
                conf_mat=confusion_matrix(mrk_class_fold_te,clf.predict(meanAmplFeatures_te.T),normalize='true')
                try:
                    fold_scores_art.append((conf_mat[1,1],sum(mrk_class_fold_te==artifact)))
                except IndexError:
                    fold_scores_art.append((0,sum(mrk_class_fold_te==artifact)))
                try:
                    fold_scores_target.append((conf_mat[0,0],sum(mrk_class_fold_te==0)))
                except IndexError:
                    fold_scores_target.append((0,sum(mrk_class_fold_te==0)))
            art_scores=np.array(fold_scores_art)
            art_score=sum(art_scores[:,0]*art_scores[:,1])/sum(art_scores[:,1])
            tar_scores=np.array(fold_scores_target)
            tar_score=sum(tar_scores[:,0]*tar_scores[:,1])/sum(tar_scores[:,1])
            meanconf.append([tar_score,art_score])
            stdconf.append([np.var(tar_scores[:,0]),np.var(art_scores[:,0])])

        
        meanconf,stdconf=np.array(meanconf),np.array(stdconf)
        xaxis = [abbrDict[x] for x in np.unique(mrk_class_a)]
        df = pd.DataFrame(zip(xaxis*2,["TP"]*6+["TN"]*6,np.concatenate((meanconf[:,0],meanconf[:,1]),axis=0),
                              np.concatenate((stdconf[:,0],stdconf[:,1]),axis=0)),columns=["Artifact Type", "Type","Average Rate","std"])
        grouped_barplot(df, "Artifact Type", "Type", "Average Rate", "std",colors=plotcolours,
                        title=f"LDA shrinkage with CSP features, {k} channels ")
        print(df)