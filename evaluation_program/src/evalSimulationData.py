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
highpass = 0.1
lowpass = 0
bandpass = [20,50]
ival = [-100,1000]
ref_ival=[-100,0]

#%%
ivals = [[160, 190], [200,230],[240,270],[280,290],[300,310],[320,330],[340,350],
        [360,370],[380, 390], [400,430],[440,470],[480, 520]]
#%%
n_samples=1000
n_samples_a=125
epo_t= np.linspace(0, 1000,101)
datadict={}
#%% DATA PREPARATION

for n_source in ['5','10','20']:
    for art in ['1','2','4','6','7','8']:
        for slicenum in range(8):
            dat=np.genfromtxt(os.path.join(simulationOutputFolder,art+'_'+str(slicenum),n_source+'.csv'),delimiter=',')
            datadict[(art+'_'+str(slicenum),n_source)]=np.reshape(dat,(31,111,125),order='F').swapaxes(0,1)[10:,:,:]
#%%
for n_source in ['5','10','20']:        
    for stddev in [0,0.2,0.4,0.6,0.8,1]:
        dat=np.genfromtxt(os.path.join(simulationOutputFolder,'stddev'+str(stddev),n_source+'.csv'),delimiter=',')
        datadict[('t'+str(stddev),n_source)]=np.reshape(dat,(31,111,1000),order='F').swapaxes(0,1)[:101,:,:]
#%%

mrk_class_a=[]        
for art in ['1','2','4','6','7','8']:
    mrk_class_a.append(np.ones(8*n_samples_a)[:n_samples]*int(art))
   
mrk_class_a=np.array(mrk_class_a).ravel()

for n_source in ['5','10','20']:
    for art in ['1','2','4','6','7','8']:
        datadict[(art,n_source)]=np.concatenate((datadict[(art+'_0',n_source)],datadict[(art+'_1',n_source)],
                                             datadict[(art+'_2',n_source)],datadict[(art+'_3',n_source)],
                                             datadict[(art+'_4',n_source)],datadict[(art+'_5',n_source)],
                                             datadict[(art+'_6',n_source)],datadict[(art+'_7',n_source)]),axis=2) 
    locals()['art'+n_source]=np.concatenate((datadict[('1',n_source)],datadict[('2',n_source)],
                                             datadict[('4',n_source)],datadict[('6',n_source)],
                                             datadict[('7',n_source)],datadict[('8',n_source)]),axis=2)
mrk_class_o=np.zeros(n_samples)
    

#%%
mrk_class=np.concatenate((mrk_class_o,mrk_class_a))

#%%
std_devs=[0,0.2,0.4,0.6,0.8,1]
for stddev in std_devs: 
    epo5=np.concatenate((datadict[('t'+str(stddev),'5')],art5),axis=2)
    epo10=np.concatenate((datadict[('t'+str(stddev),'10')],art10),axis=2)
    epo20=np.concatenate((datadict[('t'+str(stddev),'20')],art20),axis=2)

#%%
# Matrix containing the classfication results

accMatrix=np.zeros((len(cindexes),len(std_devs)))
#%%

# LDA Classifier: Artifact vs Target
# 4.3.1 Classification Tasks
# 4.3.2 Classification Results - LDA/ Shrinkage LDA
# remove shrinkage argument to run LDA without shrinkage
clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto' )

for n_source in ['5','10','20']:
    print(f"Number of sources: {n_source}")
    for j,stddev in enumerate(std_devs):
        print(f"Standard deviation: {stddev}")
        for i,(k,v) in enumerate(cindexes.items()):
                print(f"Number of channels: {k}")
                meanconf,stdconf = classifyTargetVsArtifact(clf,mrk_class_a,datadict[('t'+str(stddev),n_source)][:,v,:],
                                                            locals()['art'+n_source][:,v,:],ivals,epo_t,cv=20,verbose=False)
                accMatrix[i,j]=np.mean(np.concatenate((meanconf[:,0,0],meanconf[:,1,1])))
                print(accMatrix[i,j])
    locals()['accMatrix_shrink'+n_source]=accMatrix.copy()

#%% This code might throw an error and should be executed separately.
keys=locals().keys()
for var in keys:
    if var.startswith('accMatrix_'):
        plot_matrix(locals()[var],var, cindexes, std_devs)

 
#%%
# Plot the data

xaxis=[abbrDict[x] for x in np.unique(mrk_class_a)]
plt.figure(figsize=(10,6))
df = pd.DataFrame(zip(xaxis*2,["TP"]*6+["TN"]*6,np.concatenate((meanconf[:,0,0],meanconf[:,1,1])),
                      np.concatenate((stdconf[:,0,0],stdconf[:,1,1]))),columns=["Artifact Type", "Type","Average Rate","std"])
grouped_barplot(df, "Artifact Type", "Type", "Average Rate", "std",colors=plotcolours,
                title="LDA with shrinkage - Classification Accuracy")  

#%%
# Matrix containing the classfication results
accMatrix=np.zeros((3,len(cindexes),len(std_devs)))
#%%
# LDA Multiclass Classifier
# 4.3.1 Classification Tasks
# 4.3.2 Classification Results - LDA/ Shrinkage LDA
# remove shrinkage argument to run LDA without shrinkage
for l,n_source in enumerate(['5','10','20']):
    print(f"Number of sources: {n_source}")
    for j,stddev in enumerate(std_devs):
        epo_=np.concatenate((datadict[('t'+str(stddev),n_source)],locals()['art'+n_source]),axis=2)
        print(f"Standard deviation: {stddev}")
        for i,(k,v) in enumerate(cindexes.items()):
            print(f"Number of channels: {k}")
            clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
            meanconf,stdconf = classifyTargetVsArtifacts(clf,mrk_class,epo_[:,v,:],ivals,epo_t,cv=20)
            
            disp=ConfusionMatrixDisplay(meanconf, list(map(lambda x:abbrDict[x],np.unique(mrk_class))))
            plt.figure(figsize=(10,10))
            plt.title(n_source+'_'+str(stddev)+str(k))
            disp.plot(values_format='.2f',cmap='Blues',title='Sources: '+n_source+' Stddev: '+str(stddev)+' Channels: '+str(k))
            accMatrix[l,i,j]=np.mean(np.diag(meanconf))
#%%
for l,n_source in enumerate(['5','10','20']):
    plot_matrix(accMatrix[l],'', cindexes, std_devs)
#%%
# As a control the scalpmaps from the simulated data are shown
    
arti_ivals =[[0,100],[100,200],[200,300],[300,400],[400,500],[500,600],[600,700]]
plotScalpmapsArtifact(epo20, epo_t, clab, mrk_class, arti_ivals, mnt)