import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.special import expit
from covar import cov_shrink_ss
def train_LDA(X, y):
    '''
    Synopsis:
        w, b= train_LDA(X, y)
    Arguments:
        X: data matrix (features X samples)
        y: labels with values 0 and 1 (1 x samples)
    Output:
        w: LDA weight vector
        b: bias term
    '''
    mu1 = np.mean(X[:, y == 0], axis=1)
    mu2 = np.mean(X[:, y == 1], axis=1)

    ## Three ways to get an estimate of the covariance
    # -- 1. Simply average class-covariance matrices
    #C1 = np.cov(X[:, y==0])
    #C2 = np.cov(X[:, y==1])
    #C = (C1 + C2) / 2
    # -- 2. Weighted average of class-covariance matrices
    #C1 = np.cov(X[:, y==0])
    #C2 = np.cov(X[:, y==1])
    #N1= np.sum(y==0)           # this would be the weighted average  
    #N2= np.sum(y==1)
    #C= (N1-1)/(N1+N2-1)*C1 + (N2-1)/(N1+N2-1)*C2
    # -- 3. Center features classwise to estimate covariance on all samples at once
    Xpool = np.concatenate((X[:, y==0]-mu1[:,np.newaxis], X[:, y==1]-mu2[:,np.newaxis]), axis=1)
    C = np.cov(Xpool)

    w = np.linalg.pinv(C).dot(mu2-mu1)
    b = w.T.dot((mu1 + mu2) / 2)
    return w, b,{}

def train_LDAshrink(X, y):
    '''
    Synopsis:
        w, b= train_LDAshrink(X, y)
    Arguments:
        X: data matrix (features X samples)
        y: labels with values 0 and 1 (1 x samples)
    Output:
        w: LDA weight vector
        b: bias term
    '''
    meanT = np.mean(X[:,y==1],axis=1)
    meanNT = np.mean(X[:,y==0],axis=1)
    
    meanedT = X[:,y==1]-meanT[:,np.newaxis]
    meanedNT = X[:,y==0]-meanNT[:,np.newaxis]

    #Adding the data up again
    recData = np.hstack((meanedT,meanedNT))
    recData=recData.T.copy(order='C')
    sigma,gamma=cov_shrink_ss(recData)
    #sigma,gamma=cov_shrink(recData)

    returndic={}
    returndic['gamma']=gamma
    # Pseudoinverse also possible np.linalg.pinv
    w = np.dot(np.linalg.pinv(sigma),(meanT-meanNT))

    b = np.dot(w.T,(meanNT+meanT)/2)

    return w,b,returndic

    

def cov_shrink(X):
    '''
    Estimate covariance of given data using shrinkage estimator.
    
    Synopsis:
        C= cov_shrink(X)
    Argument:
        X: data matrix (features x samples)
    Output:
        C: estimated covariance matrix
    '''
    
    # Create Zk
    u = np.mean(X,axis=1)
    Zk = np.zeros((X.shape[0],X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
            Zk[:,:,i] = np.outer((X[:,i]-u),(X[:,i]-u).T)
#    Numerator in gamma* formula, taking variance of all ith and jth elements
    varSumZk = np.sum(np.apply_along_axis(lambda m: np.var(m),axis=2,arr=Zk))
#    Sigma hat
    C = np.cov(X)
    v = np.trace(C)/C.shape[0]
#    Denominator in gamma* formula
    denom = np.sum((C-v*np.identity(C.shape[0]))**2)
#    Formula see script
    
    gamma = 1.0*X.shape[1]/((X.shape[1]-1)**2)*varSumZk/denom
    C = (1-gamma)*C+gamma*v*np.identity(C.shape[0])
    return C,gamma

def crossvalidation(classifier_fcn, X, y, folds=10, verbose=False,returnParams=False):
    '''
    Synopsis:
        loss_te, loss_tr= crossvalidation(classifier_fcn, X, y, folds=10, verbose=False)
    Arguments:
        classifier_fcn: handle to function that trains classifier as output w, b
        X:              data matrix (features X samples)
        y:              labels with values 0 and 1 (1 x samples)
        folds:         number of folds
        verbose:        print validation results or not
    Output:
        loss_te: value of loss function averaged across test data
        loss_tr: value of loss function averaged across training data
    '''
    nDim, nSamples = X.shape
    inter = np.round(np.linspace(0, nSamples, num=folds + 1)).astype(int)
    perm = np.random.permutation(nSamples)
    errTr = np.zeros([folds, 1])
    errTe = np.zeros([folds, 1])
    returnParams_ = []

    for ff in range(folds):
        idxTe = perm[inter[ff]:inter[ff + 1] + 1]
        idxTr = np.setdiff1d(range(nSamples), idxTe)
        w, b = classifier_fcn(X[:, idxTr], y[idxTr])
        out = w.T.dot(X) - b
        errTe[ff] = loss_weighted_error(out[idxTe], y[idxTe])
        errTr[ff] = loss_weighted_error(out[idxTr], y[idxTr])
        returnParams_.append((w,b))

    if verbose:
        print('{:5.1f} +/-{:4.1f}  (training:{:5.1f} +/-{:4.1f})  [using {}]'.format(errTe.mean(), errTe.std(),
                                                                                     errTr.mean(), errTr.std(), 
                                                                                     classifier_fcn.__name__))
    if returnParams:
        return np.mean(errTe), np.mean(errTr), returnParams_
    return np.mean(errTe), np.mean(errTr)

def crossvalidationDetailedLoss(classifier_fcn, X, y, folds=10,scoring={}, verbose=False,returnParams=False):
    '''
    Synopsis:
        loss_te, loss_tr= crossvalidation(classifier_fcn, X, y, folds=10, verbose=False)
    Arguments:
        classifier_fcn: handle to function that trains classifier as output w, b
        X:              data matrix (features X samples)
        y:              labels with values 0 and 1 (1 x samples)
        folds:         number of folds
        verbose:        print validation results or not
    Output:
        loss_te: value of loss function averaged across test data
        loss_tr: value of loss function averaged across training data
    '''
    nDim, nSamples = X.shape
    inter = np.round(np.linspace(0, nSamples, num=folds + 1)).astype(int)
    perm = np.random.permutation(nSamples)
    errTrFP = np.zeros([folds, 1])
    errTeFP = np.zeros([folds, 1])
    errTrFN = np.zeros([folds, 1])
    errTeFN = np.zeros([folds, 1])
    returnParams_ = []
    scores={}
    for name,fun in scoring.items():
        scores['test_'+name]=np.zeros(folds)
    for ff in range(folds):
        idxTe = perm[inter[ff]:inter[ff + 1] + 1]
        idxTr = np.setdiff1d(range(nSamples), idxTe)
        w, b ,returndic = classifier_fcn(X[:, idxTr], y[idxTr])
        out = w.T.dot(X) - b
        out[out<0]=0
        out[out>0]=1
        returnParams_.append((w,b))
        for name,fun in scoring.items():
            scores['test_'+name][ff]=fun(out[idxTe],y[idxTe])
        for key,val in returndic.items():
            try:
                scores[key].extend([val])
            except:
                scores[key]=[val]

    if verbose:
        for name,fun in scoring.items():
            print(f'{name}: {fun(out[idxTe],y[idxTe])}')
    return scores


def loss_weighted_error(out, y, perclass = False):
    '''
    Synopsis:
        loss= loss_weighted_error( out, y )
    Arguments:
        out:  output of the classifier
        y:    true class labels
    Output:
        loss: weighted error
    '''
    falsePositive = out[y == 0] >= 0
    mFalsePositive = 0 if len(falsePositive)==0 else np.mean(falsePositive)
    falseNegative = out[y == 1] < 0
    mFalseNegative = 0 if len(falseNegative)==0 else np.mean(falseNegative)
    
    loss = 50 * (mFalseNegative+mFalsePositive)
    if perclass:
        return 100*mFalsePositive,100*mFalseNegative
    return loss