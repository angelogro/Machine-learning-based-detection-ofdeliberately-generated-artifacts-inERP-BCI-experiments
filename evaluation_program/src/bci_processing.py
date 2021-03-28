#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:55:22 2020

@author: angelo
"""
import numpy as np
import scipy as sp
from scipy import signal
from scipy.signal import savgol_filter
from global_vars import *


def calculate_csp(epo,mrk_class, classes=None):
    """Calculate the Common Spatial Pattern (CSP) for two classes.

    This method calculates the CSP and the corresponding filters. Use
    the columns of the patterns and filters.

    Examples
    --------
    Calculate the CSP for the first two classes::

    >>> w, a, d = calculate_csp(epo)
    >>> # Apply the first two and the last two columns of the sorted
    >>> # filter to the data
    >>> filtered = apply_spatial_filter(epo, w[:, [0, 1, -2, -1]])
    >>> # You'll probably want to get the log-variance along the time
    >>> # axis, this should result in four numbers (one for each
    >>> # channel)
    >>> filtered = np.log(np.var(filtered, 0))

    Select two classes manually::

    >>> w, a, d = calculate_csp(epo, [2, 5])

    Parameters
    ----------
    epo : epoched Data object
        this method relies on the ``epo`` to have three dimensions in
        the following order: class, time, channel
    classes : list of two ints, optional
        If ``None`` the first two different class indices found in
        ``epo.axes[0]`` are chosen automatically otherwise the class
        indices can be manually chosen by setting ``classes``

    Returns
    -------
    v : 2d array
        the sorted spacial filters
    a : 2d array
        the sorted spacial patterns. Column i of a represents the
        pattern of the filter in column i of v.
    d : 1d array
        the variances of the components

    Raises
    ------
    AssertionError :
        If:

          * ``classes`` is not ``None`` and has less than two elements
          * ``classes`` is not ``None`` and the first two elements are
            not found in the ``epo``
          * ``classes`` is ``None`` but there are less than two
            different classes in the ``epo``

    See Also
    --------
    :func:`apply_spatial_filter`, :func:`apply_csp`, :func:`calculate_spoc`
References
    ----------
    http://en.wikipedia.org/wiki/Common_spatial_pattern

    """
    n_channels = epo.shape[1]
    if classes is None:
        # automagically find the first two different classidx
        # we don't use uniq, since it sorts the classidx first
        # first check if we have a least two diffeent idxs:
        assert len(np.unique(mrk_class)) >= 2
        cidx1 = mrk_class[0]
        cidx2 = mrk_class[mrk_class != cidx1][0]
    else:
        assert (len(classes) >= 2 and
            classes[0] in mrk_class and
            classes[1] in mrk_class)
        cidx1 = classes[0]
        cidx2 = classes[1]

    epoc1 = epo[:,:,mrk_class==cidx1]
    epoc2 = epo[:,:,mrk_class==cidx2]

    # we need a matrix of the form (observations, channels) so we stack trials
    # and time per channel together
    x1 = epoc1.reshape(-1, epoc1.shape[1])
    x2 = epoc2.reshape(-1, epoc2.shape[1])
    # compute covariance matrices of the two classes
    c1 = np.cov(x1.transpose())
    c2 = np.cov(x2.transpose())
    # solution of csp objective via generalized eigenvalue problem
    # in matlab the signature is v, d = eig(a, b)
    d, v = sp.linalg.eig(c1, c1+c2)
    d = d.real
    # make sure the eigenvalues and -vectors are correctly sorted
    indx = np.argsort(d)
    # reverse
    indx = indx[::-1]
    d = d.take(indx)
    v = v.take(indx, axis=1)
    a = sp.linalg.inv(v).transpose()
    return v, a, d

def apply_spatial_filter(epo, w):
    """Apply spatial filter to ``Data`` object.

    This method applies the spatial filter ``w`` to a continuous or
    epoched ``Data`` object.

    Depending on the filter ``w``, since the original channel names may
    become meaningless. For that you can either set a ``prefix`` (e.g.
    ``CSP``) and the resulting channels will be renamed to prefix +
    channel number (e.g. ``CSP0``, ``CSP1``, etc.).

    Alternatively you can set a suffix e.g. ``Laplace`` and the
    resulting channels will be renamed to original channel name + suffix
    (e.g. ``Cz Laplace``, etc.)

    If neither pre- or postfix are set, the channel names will be kept.

    Parameters
    ----------
    dat : Data
        Data object
    w : 2d array
        Spatial filter matrix
    prefix : str, optional
        the channel prefix
    postfix : str, optional
        the channel postfix
    chanaxis : int, optional
        the index of the channel axis

    Returns
    -------
    dat : Data
        The resulting spatial-filtered data

    Examples
    --------

    >>> w, _, _ = calculate_csp(epo)
    >>> epo_filtered = apply_spatial_filter(epo, w, prefix='CSP ')
    >>> epo_filtered.names[-1] = 'CSP Channel'

    Raises
    ------
    ValueError : If prefix and postfix are not None
    TypeError : If prefix or postfix are not None and not str

    See Also
    --------
    :func:`calculate_csp`, :func:`calculate_cca`, :func:`apply_spatial_filter`

    """
    """
    if prefix is not None and postfix is not None:
        raise ValueError('Please chose either pre- or postfix, not both.')
    dat = dat.copy()
    dat = swapaxes(dat, -1, chanaxis)
    shape_orig = dat.data.shape
    # the target shape will change in the channel-dimension we set that
    # to -1 as in 'automagic'
    shape_target = list(shape_orig)
    shape_target[chanaxis] = -1
    data = dat.data.reshape(-1, dat.data.shape[-1])
    data = np.dot(data, w)
    dat.data = data.reshape(shape_target)
    dat = swapaxes(dat, -1, chanaxis)
    if prefix is not None:
        dat.axes[chanaxis] = [prefix+str(i) for i, _ in enumerate(dat.axes[chanaxis])]
    if postfix is not None:
        dat.axes[chanaxis] = [chan+postfix for chan in dat.axes[chanaxis]]
    return dat
    """
    epo = np.swapaxes(epo,1,2)
    epo_filt = np.dot(epo,w)
    epo_filt = np.swapaxes(epo_filt,1,2)
    return epo_filt

def calcPSDfromEpoMultichannel(epo,mrk_class,clab,sel_channels,sel_mrk_class,fs,frange=None):
    epo = epo[:,:,mrk_class==sel_mrk_class]
    epo = epo.swapaxes(0,1)
    epo = epo.swapaxes(2,1)
    epo = epo.reshape(len(clab),-1)
    if frange == None:
        frange = [0,fs/2]
    
    lstPSD = []
    for channel in sel_channels:
        f, psd = signal.welch(epo[np.where(clab==channel)[0][0],:], fs=fs,nperseg=4096)
        idx_range = np.where((f>=frange[0])&(f<=frange[1]))
        lstPSD.append(psd[idx_range])

    return f[idx_range],lstPSD

def saveMeanArtifactSignals(epo_a,epo_t,mrk_class,path,n_batches=0):

    for j,artifact_class in zip(range(len(np.unique(mrk_class))),np.unique(mrk_class)):
        if n_batches==0:
            
            meanArtifact=np.mean(epo_a[:,:,mrk_class==artifact_class],axis=2)
            smooth_meanArtifact=savgol_filter(meanArtifact,11,3,axis=0)
            np.savetxt(str(artifact_class),smooth_meanArtifact)
        else:
            
            epo = epo_a[:,:,mrk_class==artifact_class]
            batchsize=int(epo.shape[2]/n_batches)
            oldindex=0
            for i in range(epo.shape[2]):  
                meanArtifact=np.mean(epo[:,:,oldindex:oldindex+batchsize],axis=2)
                smooth_meanArtifact=savgol_filter(meanArtifact,11,3,axis=0)
                np.savetxt(os.path.join(path,str(artifact_class)+'_'+str(i)),smooth_meanArtifact)
                oldindex+=batchsize
                
                if oldindex+batchsize>epo.shape[2]:
                    meanArtifact=np.mean(epo[:,:,oldindex:],axis=2)
                    smooth_meanArtifact=savgol_filter(meanArtifact,11,3,axis=0)
                    np.savetxt(os.path.join(path,str(artifact_class)+'_'+str(i)),smooth_meanArtifact)
                    break
              
                    
                    
                    
