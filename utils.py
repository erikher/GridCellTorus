from datetime import datetime 
import functools
import glob
from gudhi.clustering.tomato import Tomato
import h5py
import matplotlib
from matplotlib import animation, cm, transforms, pyplot as plt, gridspec as grd
from matplotlib.collections import PathCollection
import numba
import numpy as np
from ripser import Rips, ripser
from scipy import stats, signal, optimize
from scipy.optimize import minimize
import scipy.io as sio
from scipy.ndimage import gaussian_filter,  gaussian_filter1d, rotate, binary_dilation, binary_closing
from scipy.stats import binned_statistic_2d, pearsonr, multivariate_normal
from scipy.special import factorial
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.stats
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score
import sys
import time
import os

_LOG_2PI = np.log(2 * np.pi)



colors_mods = {}
colors_mods['R1'] = [[0.7,0.3,0.3]]
colors_mods['R2'] = [[0.7,0.7,0]]
colors_mods['R3'] = [[0.7,0.,0.7]]
colors_mods['Q1'] = [[0.3,0.7,0.3]]
colors_mods['Q2'] = [[0.,0.7,0.7]]
colors_mods['S1'] = [[0.3,0.3,0.7]]


colors_envs = {}
colors_envs['R1OFday1'] = [[0.7,0.3,0.3]]
colors_envs['R2OFday1'] = [[0.7,0.7,0]]
colors_envs['R3OFday1'] = [[0.7,0.,0.7]]
colors_envs['R1OFday2'] = [[0.7,0.3,0.3]]
colors_envs['R2OFday2'] = [[0.7,0.7,0]]
colors_envs['R3OFday2'] = [[0.7,0.,0.7]]
colors_envs['Q1OF'] = [[0.3,0.7,0.3]]
colors_envs['Q2OF'] = [[0.,0.7,0.7]]
colors_envs['S1OF'] = [[0.3,0.3,0.7]]
colors_envs['R1WWday1'] = [[0.7,0.3,0.3]]
colors_envs['R2WWday1'] = [[0.7,0.7,0]]
colors_envs['R3WWday1'] = [[0.7,0.,0.7]]
colors_envs['Q1WW'] = [[0.3,0.7,0.3]]
colors_envs['Q2WW'] = [[0.,0.7,0.7]]
colors_envs['S1WW'] = [[0.3,0.3,0.7]]
colors_envs['R1REMday2'] = [[0.7,0.3,0.3]]
colors_envs['R2REMday2'] = [[0.7,0.7,0]]
colors_envs['R3REMday2'] = [[0.7,0.,0.7]]
colors_envs['Q1REM'] = [[0.3,0.7,0.3]]
colors_envs['Q2REM'] = [[0.,0.7,0.7]]
colors_envs['R1SWSday2'] = [[0.7,0.3,0.3]]
colors_envs['R2SWSday2'] = [[0.7,0.7,0]]
colors_envs['R3SWSday2'] = [[0.7,0.,0.7]]
colors_envs['Q1SWS'] = [[0.3,0.7,0.3]]
colors_envs['Q2SWS'] = [[0.,0.7,0.7]]

Information_rate_OF_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]
Information_rate_C_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]
Information_rate_REM_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM']
Information_rate_SWS_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM']
Explained_variance_OF_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]
Explained_variance_C_lbls =['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]
Explained_variance_REM_lbls =['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM']
Explained_variance_SWS_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM','Q1','Q1_SEM','Q2','Q2_SEM']
Covariance_explained_OF_lbls = ['R1','R2','R3', 'Q1','Q2','S1']
Covariance_explained_C_lbls = ['R1','R2','R3', 'Q1','Q2','S1']
Covariance_explained_REM_lbls = ['R1','R2','R3', 'Q1','Q2','S1']
Covariance_explained_SWS_lbls = ['R1','R2','R3', 'Q1','Q2','S1']

Peak_distance_OFvsC_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]
Peak_distance_OFvsREM_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]
Peak_distance_OFvsSWS_lbls = ['R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]

Pearson_correlation_OFvsC_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]
Pearson_correlation_OFvsREM_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM']
Pearson_correlation_OFvsSWS_lbls = ['R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM']

colors_mods = {}
colors_mods['R1'] = [[0.7,0.3,0.3]]
colors_mods['R2'] = [[0.7,0.7,0]]
colors_mods['R3'] = [[0.7,0.,0.7]]
colors_mods['Q1'] = [[0.3,0.7,0.3]]
colors_mods['Q2'] = [[0.,0.7,0.7]]
colors_mods['S1'] = [[0.3,0.3,0.7]]

ylims = {}
ylims['corr'] = [-0.05, 1]
ylims['dist'] = [0, 2.6/(2*np.pi)*360]
ylims['info'] = [0,110]
ylims['ED_OF'] = [0,0.2]
ylims['ED_WW'] = [0,0.2]

ytics = {}
ytics['corr'] = [0.0, 0.25, 0.5, 0.75, 1]
ytics['dist'] = [0, 50, 100, 150]
ytics['ED_OF'] = [0, 0.05,0.1, 0.15, 0.2]
ytics['ED_WW'] = [0, 0.05,0.1, 0.15, 0.2]
ytics['info'] = [0.0, 25, 50, 75,100]

Peak_distance_OFvsC_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]
Peak_distance_OFvsREM_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM']
Peak_distance_OFvsSWS_lbls = ['R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM' ]

Pearson_correlation_OFvsC_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]
Pearson_correlation_OFvsREM_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM']
Pearson_correlation_OFvsSWS_lbls = ['R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM']



def plot_stats(stats_tor, stats_space = [], lbls = [], statname='', sess_name=''):
    if statname == 'dist':
        stats_tor = np.array(stats_tor)/(2*np.pi)*360
        stats_space = np.array(stats_space)/(2*np.pi)*360
    num_mods = len(stats_tor)
    bSems = False
    if lbls[1][-3:] == 'SEM':
        bSems = True
        num_mods = int(num_mods/2)
    fig = plt.figure(figsize=(2,6))
    ax = fig.add_subplot(111)
    xs = [0.01, 0.09]
    if len(stats_space):
        if len(stats_space):
            for i in range(num_mods):
                if bSems:
                    i*=2

                ax.scatter(xs, [stats_tor[i],stats_space[i]], s = 250, marker='.', #lw = 3, 
                    color = colors_mods[lbls[i]], zorder=-1)
                ax.plot(xs, [stats_tor[i],stats_space[i]], lw = 4, c = colors_mods[lbls[i]][0], alpha = 0.5, zorder = -2)

        if bSems:
            for i in range(num_mods):
                i*=2
                ax.errorbar(xs, [stats_tor[i],stats_space[i]], lw = 4, yerr=[stats_tor[i+1],stats_space[i+1]], 
                    c = colors_mods[lbls[i]][0], 
                    fmt='none', alpha = 1, zorder=-2)

    ax.set_xlim([0,0.1])
    ax.set_ylim(ylims[statname])
    
    ax.set_xticks(xs)
    ax.set_xticklabels(np.zeros(len(xs),dtype=str))
    ax.xaxis.set_tick_params(width=1, length =7)
    ax.set_yticks(ytics[statname])
    ax.set_yticklabels(np.zeros(len(ytics[statname]),dtype=str))
    ax.yaxis.set_tick_params(width=1, length =7)

    
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/(abs(y1-y0))*3.5)
    
    plt.gca().axes.spines['top'].set_visible(False)
    plt.gca().axes.spines['right'].set_visible(False)


## Note, the following funtion is imported from the UMAP library
@numba.njit(parallel=True, fastmath=True) 
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_samples * n_neighbors), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
                #val = ((knn_dists[i, j] - rhos[i]) / (sigmas[i]))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals

## Note, the following funtion is imported from the UMAP library
@numba.njit(
    fastmath=True
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=0.0, bandwidth=1.0):
    target = np.log2(k) * bandwidth
#    target = np.log(k) * bandwidth
#    target = k
    
    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = np.inf
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > 1e-5:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
#                    psum += d / mid
 
                else:
                    psum += 1.0
#                    psum += 0

            if np.fabs(psum - target) < 1e-5:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        result[i] = mid
        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < 1e-3 * mean_ith_distances:
                result[i] = 1e-3 * mean_ith_distances
        else:
            if result[i] < 1e-3 * mean_distances:
                result[i] = 1e-3 * mean_distances

    return result, rho



def pca(data, dim=2):
    if dim < 2:
        return data, [0]
    m, n = data.shape
    # mean center the data
    # data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eig(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dim]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    tot = np.sum(evals)
    var_exp = [(i / tot) * 100 for i in sorted(evals[:dim], reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    components = np.dot(evecs.T, data.T).T
    return components, var_exp, evals[:dim]


def smooth_image(img, sigma):
    filterSize = max(np.shape(img))
    grid = np.arange(-filterSize+1, filterSize, 1)
#    covariance = np.square([sigma, sigma])
    xx,yy = np.meshgrid(grid, grid)

    pos = np.dstack((xx, yy))

    var = multivariate_normal(mean=[0,0], cov=[[sigma**2,0],[0,sigma**2]])
    k = var.pdf(pos)
    k = k/np.sum(k)

    nans = np.isnan(img)
    imgA = img.copy()
    imgA[nans] = 0
    imgA = scipy.signal.convolve2d(imgA, k, mode='valid')
#    imgA = gaussian_filter(imgA, sigma = sigma, mode = mode)
    imgD = img.copy()
    imgD[nans] = 0    
    imgD[~nans] = 1
    radius = 1
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    dk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=bool)
    imgE = np.zeros((filterSize+2,filterSize+2))
    imgE[1:-1,1:-1] = imgD
    imgE= binary_closing(imgE,iterations =1, structure =dk)
    imgD = imgE[1:-1,1:-1]

    imgB = np.divide(scipy.signal.convolve2d(imgD, k, mode='valid'), scipy.signal.convolve2d(np.ones(np.shape(imgD)), k, mode='valid'))
    imgC = np.divide(imgA,imgB)
    imgC[imgD==0] = -np.inf
    return imgC

def smooth_tuning_map(mtot, numangsint, sig, bClose = True):
    numangsint_1 = numangsint-1
    mid = int((numangsint_1)/2)
    indstemp1 = np.zeros((numangsint_1,numangsint_1), dtype=int)
    indstemp1[indstemp1==0] = np.arange((numangsint_1)**2)
    indstemp1temp = indstemp1.copy()
    mid = int((numangsint_1)/2)
    mtemp1_3 = mtot.copy()
    for i in range(numangsint_1):
        mtemp1_3[i,:] = np.roll(mtemp1_3[i,:],int(i/2))
    mtot_out = np.zeros_like(mtot)
    mtemp1_4 = np.concatenate((mtemp1_3, mtemp1_3, mtemp1_3),1)
    mtemp1_5 = np.zeros_like(mtemp1_4)
    mtemp1_5[:, :mid] = mtemp1_4[:, (numangsint_1)*3-mid:]  
    mtemp1_5[:, mid:] = mtemp1_4[:,:(numangsint_1)*3-mid]      
    if bClose:
        mtemp1_6 = smooth_image(np.concatenate((mtemp1_5,mtemp1_4,mtemp1_5)) ,sigma = sig)
    else:
        mtemp1_6 = gaussian_filter(np.concatenate((mtemp1_5,mtemp1_4,mtemp1_5)) ,sigma = sig)
    for i in range(numangsint_1):
        mtot_out[i, :] = mtemp1_6[(numangsint_1)+i, 
                                          (numangsint_1) + (int(i/2) +1):(numangsint_1)*2 + (int(i/2) +1)] 
    return mtot_out

def sample_denoising(data,  k = 10, num_sample = 500, omega = 0.2, metric = 'euclidean'):    
    n = data.shape[0]
    leftinds = np.arange(n)
    F_D = np.zeros(n)
    if metric in ("cosine", "correlation", "dice", "jaccard"):
        angular = True
    else:
        angular = False
    
    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    sigmas, rhos = smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    X = result.toarray()
    F = np.sum(X,1)
    Fs = np.zeros(num_sample)
    Fs[0] = np.max(F)
    i = np.argmax(F)
    inds_all = np.arange(n)
    inds_left = inds_all>-1
    inds_left[i] = False
    inds = np.zeros(num_sample, dtype = int)
    inds[0] = i
    for j in np.arange(1,num_sample):
        F -= omega*X[i,:]
        Fmax = np.argmax(F[inds_left])
        Fs[j] = F[Fmax]
        i = inds_all[inds_left][Fmax]
        
        inds_left[i] = False   
        inds[j] = i
    d = np.zeros((num_sample, num_sample))
    
    for j,i in enumerate(inds):
        d[j,:] = X[i, inds]
    return inds, d, Fs


def preprocess_dataX2(Xin, num_bins):
    Xin = np.transpose(Xin)
    num_dim, num_times = np.shape(Xin)

    tmp = np.linspace(-0.001, 1.001, num_bins+1)

    dig = (np.digitize(np.array(Xin[0,:]), tmp)-1)*num_bins
    dig += (np.digitize(np.array(Xin[1,:]), tmp)-1)

    X = np.zeros((num_times, num_bins**2))
    X[range(num_times), dig] = 1
    return np.transpose(X)

def normscale(X):
    return (X- np.min(X))/(np.max(X)-np.min(X))

def information_score(mtemp, circ, mu):
    numangsint = mtemp.shape[0]+1
    circ = np.ravel_multi_index(circ-1, np.shape(mtemp))
    mtemp = mtemp.flatten() 
    p = np.bincount(circ, minlength = (numangsint-1)**2)/len(circ)
    logtemp = np.log2(mtemp/mu)
    mtemp = np.multiply(np.multiply(mtemp,p), logtemp)
    return np.sum(mtemp[~np.isnan(mtemp)])


def get_ang_hist(c11all, c12all,xx, yy):
    def circmean(x):
        return np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x)))
    numangsint = 101
    binsx = np.linspace(np.min(xx)+ (np.max(xx) - np.min(xx))*0.0025,np.max(xx)- (np.max(xx) - np.min(xx))*0.0025, numangsint)
    binsy = np.linspace(np.min(yy)+ (np.max(yy) - np.min(yy))*0.0025,np.max(yy)- (np.max(yy) - np.min(yy))*0.0025, numangsint)

    nnans = ~np.isnan(c11all)

    mtot, x_edge, y_edge, circ = binned_statistic_2d(xx[nnans],yy[nnans], c11all[nnans], 
        statistic=circmean, bins=(binsx,binsy), range=None, expand_binnumbers=True)
    nans = np.isnan(mtot)
    sintot = np.sin(mtot)
    costot = np.cos(mtot)
    sintot[nans] = 0
    costot[nans] = 0
    sig = 1
    sintot = gaussian_filter(sintot,sig)
    costot = gaussian_filter(costot,sig)

    mtot = np.arctan2(sintot, costot)
    mtot[nans] = np.nan


    nnans = ~np.isnan(c12all)
    mtot1, x_edge, y_edge, circ = binned_statistic_2d(xx[nnans],yy[nnans], c12all[nnans], 
        statistic=circmean, bins=(binsx,binsy), range=None, expand_binnumbers=True)
    nans = np.isnan(mtot1)
    sintot = np.sin(mtot1)
    costot = np.cos(mtot1)
    sintot[nans] = 0    
    costot[nans] = 0
    sintot = gaussian_filter(sintot,sig)
    costot = gaussian_filter(costot,sig)
    mtot1 = np.arctan2(sintot, costot)    
    mtot1[nans] = np.nan
    return mtot, mtot1, x_edge, y_edge

def fit_sine_wave(mtot):
    def circ_dist(xtmp,ytmp):
        return np.abs(np.arctan2(np.sin(xtmp-ytmp), np.cos(xtmp -ytmp)))
    nb = 25
    nnans = ~np.isnan(mtot)
    numangsint = 151
    x,y = np.meshgrid(np.linspace(0,3*np.pi,numangsint-1), np.linspace(0,3*np.pi,numangsint-1))

    def cos_wave(p):
        x1 =rotate(x, p[0]*360/(2*np.pi), reshape= False)
        return np.mean(np.square(np.cos(p[2]*x1[nb:-nb,nb:-nb][nnans]+p[1])-np.cos(mtot[nnans])))#*1000
    t = time.time()
    inum = 10
    jnum = 10
    knum = 10
    i_space = np.linspace(0,np.pi,inum)
    j_space = np.linspace(0,2*np.pi,jnum)
    k_space = np.linspace(1.,6,knum)
    a = np.zeros((inum,jnum,knum))
    for i,i1 in enumerate(i_space):
        for j,j1 in enumerate(j_space):
            for k,k1 in enumerate(k_space):            
                a[i,j,k] = cos_wave([i1,j1,k1])
    t1 = time.time()
 
    p_ind = np.unravel_index(np.argmin(a), a.shape)    
    p = [i_space[p_ind[0]],j_space[p_ind[1]],k_space[p_ind[2]]]
    res = minimize(cos_wave, p, method='SLSQP', 
        options={'disp': False})
    p = res['x']
    fun = res['fun']
    t2 = time.time()
    return p, fun
    
def plot_stripes(xx,yy, p, mtot, name):
    nb = 25
    numangsint = 151
    x,y = np.meshgrid(np.linspace(0,3*np.pi,numangsint-1), np.linspace(0,3*np.pi,numangsint-1))
    x =rotate(x, p[0]*360/(2*np.pi), reshape= False)
    fig, ax = plt.subplots(1,1)
    ax.imshow(np.cos(p[2]*x[nb:-nb,nb:-nb]+p[1]), origin = 'lower', extent = [xx[0],xx[-1],yy[0],yy[-1]])
    ax.set_aspect('equal','OF')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    fig.savefig('Figs/stripes/' + name + '_cosstripes', bOF_inches='tight', pad_inches=0.02)
        
    fig, ax = plt.subplots(1,1)
    ax.imshow(np.cos(mtot), origin = 'lower', extent = [xx[0],xx[-1],yy[0],yy[-1]])
    ax.set_aspect('equal','OF')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    fig.savefig('Figs/stripes/' + name + '_OFstripes.png', bOF_inches='tight', pad_inches=0.02)
    fig.savefig('Figs/stripes/' + name + '_OFstripes.pdf', bOF_inches='tight', pad_inches=0.02)
    return

def rot_para(params1,params2):   
    if np.abs(np.cos(params1[0])) < np.abs(np.cos(params2[0])):
        y = params1.copy()
        x = params2.copy()
    else:
        x = params1.copy()
        y = params2.copy()

    alpha = (y[0]-x[0])
    if (alpha < 0) & (np.abs(alpha) > np.pi/2):
        x[0] += np.pi 
        x[0] = x[0]%(2*np.pi)
    elif (alpha < 0) & (np.abs(alpha) < np.pi/2):
        x[0] += np.pi*4/3  
        x[0] = x[0]%(2*np.pi)
    elif np.abs(alpha) > np.pi/2:
        y[0] -= np.pi/3
        y[0] = y[0]%(2*np.pi)
    if y[0]>np.pi/2:
        y[0] -= np.pi/6
        x[0] -= np.pi/6
        x[0] = x[0]%(2*np.pi)
        y[0] = y[0]%(2*np.pi)
    if x[0]>np.pi/2:
        y[0] += np.pi/6
        x[0] += np.pi/6
        x[0] = x[0]%(2*np.pi)
        y[0] = y[0]%(2*np.pi)
    return x,y


def plot_para(ax, xedge, yedge, x, y, c1):    
    
    xmin = xedge.min()
    xedge -= xmin
    xmax = xedge.max()
    ymin = yedge.min()
    yedge -= ymin
    ymax = yedge.max()
    plt.plot([0,0],[0,ymax], '--', c = 'k')
    plt.plot([xmax,xmax],[0,ymax],  '--', c = 'k')
    plt.plot([0,xmax],[0,0],  '--', c = 'k')
    plt.plot([0,xmax],[ymax,ymax],  '--', c = 'k')
    a = 1/x[2]*np.cos(x[0])*xmax
    b = 1/x[2]*np.sin(x[0])*xmax
    c = 1/y[2]*np.cos(y[0])*ymax
    d = 1/y[2]*np.sin(y[0])*ymax

    ax.plot([0,a], [0,b], c = c1, lw = 4)
    ax.plot([0,c], [0,d], c = c1, lw = 4)
    ax.plot([a, a + c], [b,b +d], c = c1, lw = 4)
    ax.plot([c,a+c], [d,d+b] ,c = c1, lw = 4)
    return ax

def normit(xxx):
    xx = xxx-np.min(xxx)
    xx = xx/np.max(xx)
    return(xx)
    
def unique(list1):
    return(list(set(list1)))

def get_isi_acorr(spk, maxt = 0.2, res = 1e-3, thresh = 0.02, bLog = False, bOne = False):
    if bLog:
        num_bins = 100
        bin_times = np.ones(num_bins+1)*10
        bin_times = np.power(bin_times, np.linspace(np.log10(0.005), log10(maxt), num_bins+1))
        bin_times = np.unique(np.concatenate((-bin_times, bin_times)))
        num_bins = len(bin_times)
    elif bOne:
        num_bins = int(maxt/res)+1
        bin_times = np.linspace(0,maxt, num_bins)
    else:
        num_bins = int(2*maxt/res)+1
        bin_times = np.linspace(-maxt,maxt, num_bins)
    num_neurons = len(spk)
    acorr = np.zeros((num_neurons, len(bin_times)-1), dtype = int)    
#    print(bin_times)
    
    maxt-=1e-5
    mint = -maxt
    if bOne:
        mint = -1e-5
    for i, n in enumerate(spk):
        spike_times = spk[n]
        for ss in spike_times:
            stemp = spike_times[(spike_times<ss+maxt) & (spike_times>ss+mint)]
            dd = stemp-ss
            
            acorr[i,:] += np.bincount(np.digitize(dd, bin_times)-1, minlength=num_bins)[:-1]
    return acorr

def get_coords(cocycle, threshold, num_sampled, dists, coeff):
    zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
    cocycle[zint, 2] = cocycle[zint, 2] - coeff
    d = np.zeros((num_sampled, num_sampled))
    d[np.tril_indices(num_sampled)] = np.NaN
    d[cocycle[:, 1], cocycle[:, 0]] = cocycle[:, 2]
    d[dists > threshold] = np.NaN
    d[dists == 0] = np.NaN
    edges = np.where(~np.isnan(d))
    verts = np.array(np.unique(edges))
    num_edges = np.shape(edges)[1]
    num_verts = np.size(verts)
    values = d[edges]
    A = np.zeros((num_edges, num_verts), dtype=int)
    v1 = np.zeros((num_edges, 2), dtype=int)
    v2 = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        v1[i, :] = [i, np.where(verts == edges[0][i])[0]]
        v2[i, :] = [i, np.where(verts == edges[1][i])[0]]

    A[v1[:, 0], v1[:, 1]] = -1
    A[v2[:, 0], v2[:, 1]] = 1
  
    L = np.ones((num_edges,))
    Aw = A * np.sqrt(L[:, np.newaxis])
    Bw = values * np.sqrt(L)
    f = lsmr(Aw, Bw)[0]%1
    return f, verts


times_all = {}

times_all['rat_q_OF'] = ((27826, 31223),)
times_all['rat_q_WW'] = ((18977, 25355),)

times_all['rat_q_REM'] = ((10424.38,10565.21),
    (11514.83,11716.55),
    (12780.35,13035.9),
    (14301.05,14475.470000000001),
    (15088.98,15310.66),
    (17739.35,17797.21))


times_all['rat_q_sleep'] = ((9576, 18812),)

times_all['rat_q_SWS'] = ((9658.5,10419.7),
    (10570.84,11498.85),
    (11724.619999999999,12769.93),
    (13284.11,13800.96),
    (13813.43,14065.220000000001),
    (14088.54,14233.84),
    (14634.36,15082.369999999999),
    (15694.560000000001,16153.71),
    (16176.34,16244.810000000001),
    (16361.59,16657.57),
    (16774.48,16805.260000000002),
    (16819.59,16882.91),
    (16930.81,17005.510000000002),
    (17333.04,17486.34),
    (17510.46,17719.79),
    (17830.92,18347.36))

times_all['rat_r_WWday1'] = ((16925, 20704),)

times_all['rat_r_OFday1'] = ((7457, 16045),)
times_all['rat_r_OFday2'] = ((10617, 13004),)

times_all['rat_r_sleepday2'] = ((396, 9941),)

times_all['rat_r_REMday2'] = ((1001.72,1053.43),
    (1438.64,1514.23),
    (1856.97,1997.05),
    (2565.89,2667.44),
    (2721.26,2750.21),
    (3329.44,3434.35),
    (3448.83,3479.61),
    (4465.09,4611.65),
    (5441.24,5478.77),
    (5562.09,5677.59),
    (7221.41,7276.55),
    (7476.81,7545.26),
    (7568.93,7697.37),
    (8481.630000000001,8621.3),
    (8640.029999999999,8798.99),
    (9391.08,9429.26))    
    
times_all['rat_r_SWSday2'] = ((524.52,771.8299999999999),
    (784.51,859.74),
    (910.58,990.36),
    (1056.25,1200.8600000000001),
    (1217.7,1435.26),
    (1559.2,1650.32),
    (1707.31,1843.54),
    (2000.81,2067.13),
    (2099.98,2524.64),
    (2671.25,2696.87),
    (2753.85,3265.73),
    (3537.05,3994.15),
    (4010.0,4290.1900000000005),
    (4349.68,4460.13),
    (4646.79,5001.85),
    (5036.41,5299.54),
    (5318.34,5425.57),
    (5497.6,5556.06),
    (5683.95,6430.18),
    (6442.24,6591.57),
    (6610.89,6644.04),
    (6969.860000000001,7204.610000000001),
    (7309.91,7464.59),
    (7727.3,8249.349999999999),
    (8297.99,8425.86),
    (8449.71,8475.99),
    (9004.54,9319.189999999999),
    (9716.05,9941.0))



times_all['rat_s_OF'] = ((9939, 12363),)
times_all['rat_s_sleep'] = ((14942, 23133),)
times_all['rat_s_WW'] = ((13670, 14847),)
times_all['rat_s_WW2'] = ((23186, 24936),)


times_all['rat_s_REM'] = ((16756.48,16858.35),
    (18284.510000000002,18384.48),
    (19126.9,19151.98),
    (19164.2,19259.86),
    (20137.48,20188.68),
    (20975.93,21060.02),
    (21460.02,21590.43))

times_all['rat_s_SWS'] = ((15135.43,15164.98),
    (15375.96,15566.33),
    (15988.16,16276.45),
    (16362.16,16690.4),
    (16865.190000000002,16995.75),
    (17075.36,17226.59),
    (17256.239999999998,17426.66),
    (17640.2,18269.73),
    (18389.13,18576.68),
    (18646.48,19097.03),
    (19263.37,20132.2),
    (20193.7,20903.59),
    (20939.7,20968.79),
    (21068.0,21100.31),
    (21114.27,21451.13),
    (21594.84,21664.78),
    (22126.18,22416.17),
    (22505.37,22719.33),
    (22980.85,23133.0))

def get_spikes(rat_name, mod_name, day_name = '', sess_name = '', bType = None, bSmooth = True, 
               bBinned = True, bSpeed = True, bStart = False, smoothing_width = -1, folder = ''):
    if np.str.upper(rat_name) == 'R':
        f = np.load(folder + 'rat_r_' + day_name + '_grid_modules_1_2_3.npz', allow_pickle = True)
    elif np.str.upper(rat_name) == 'Q':
        f = np.load(folder + 'rat_q_grid_modules_1_2.npz', allow_pickle = True)
    elif np.str.upper(rat_name) == 'S':
        f = np.load(folder + 'rat_s_grid_modules_1.npz', allow_pickle = True)
    else:
        print('Correct rat name was not given')
        return
    spikes_all = f['spikes_mod' + str(mod_name)][()]
    t = f['t']
    if bSpeed:
        x = f['x']
        y = f['y']
    f.close()
    cell_inds = np.arange(len(spikes_all))
    if bType:    
        f = np.load('is_conjunctive_all.npz', allow_pickle = True)
        if len(day_name)>0:
            is_conj = f['is_conj_' + np.str.upper(rat_name) + mod_name + '_' + day_name]
        else:
            is_conj = f['is_conj_' + np.str.upper(rat_name) + mod_name + day_name]
        f.close()
        if bType == 'conj':
            cell_inds = cell_inds[is_conj]
        else:
            cell_inds = cell_inds[~is_conj]
    
    if sess_name.find('_')>-1: 
        f = np.load( folder + 'Results/grid_cell_classes_indices.npz', allow_pickle = True)
        ind = f[np.str.upper(rat_name) + mod_name + '_ind']
        f.close()
        if sess_name[sess_name.find('_')+1:] == 'bursty':
            cell_inds = cell_inds[ind==0]
        elif sess_name[sess_name.find('_')+1:] == 'theta':
            cell_inds = cell_inds[ind==1]
        elif sess_name[sess_name.find('_')+1:] == 'nonbursty':
            cell_inds = cell_inds[ind==2]
    if np.str.upper(sess_name[:3]) in ('SWS', 'REM'):
        times = times_all['rat_' + np.str.lower(rat_name) + '_sleep' + day_name]
    elif np.str.upper(sess_name[:3]) == 'WW2':
        times = times_all['rat_' + np.str.lower(rat_name) + '_' + sess_name + day_name]        
    else:
        times = times_all['rat_' + np.str.lower(rat_name) + '_' + sess_name[:2] + day_name]        
    min_time0, max_time0 = times[0]
    spikes = {}
    for i,m in enumerate(cell_inds):
        s = spikes_all[m]
        spikes[i] = np.array(s[(s>= min_time0) & (s< max_time0)])
        
    if not bBinned:
        if (rat_name == 'R') & (day_name == 'day1'):
            if sess_name[:2] =='OF':
                non_valid_times = [14778, 14890]
            elif sess_name[:2] == 'WW':
                non_valid_times = [18026, 18183]
            for i,m in enumerate(cell_inds):
                spikes[i] = spikes[i][(spikes[i]<= non_valid_times[0]) | (spikes[i]>= non_valid_times[1])]

        if (rat_name =='S') & (sess_name == 'WW'):
            times2 = times_all['rat_' + np.str.lower(rat_name) + '_' + sess_name + '2' + day_name]        
            min_time1, max_time1 = times2[0]
            for i,m in enumerate(cell_inds):
                s = spikes_all[m]
                spikes[i] = np.concatenate((spikes[i], np.array(s[(s>= min_time1) & (s< max_time1)])))
        if sess_name[:2] in ('OF', 'WW'):
            if bSpeed:
                times1 = np.where((t>=min_time0) & (t<max_time0))
                t = t[times1]
                x = x[times1]
                y = y[times1]
                xxs = gaussian_filter1d(x-np.min(x), sigma = 100)
                yys = gaussian_filter1d(y-np.min(y), sigma = 100)    
                dx = (xxs[1:] - xxs[:-1])*100
                dy = (yys[1:] - yys[:-1])*100
                speed = np.divide(np.sqrt(dx**2+ dy**2), t[1:] - t[:-1])
                speed = np.concatenate(([speed[0]],speed))
                ssp  = np.where(speed>2.5)[0]
                sw = np.where((ssp[1:] - ssp[:-1])>1)[0]
                for i in range(len(spikes)):
                    min_time1 = t[ssp[0]-1]
                    max_time1 = t[ssp[sw[0]]+1]
                    sall = []
                    s = spikes[i].copy()
                    for ss in range(len(sw)):
                        stemp = s[(s>= min_time1) & (s< max_time1)]
                        sall.extend(stemp)
                        min_time1 = t[ssp[sw[ss-1]]-1]
                        max_time1 = t[ssp[sw[ss]]+1]
                    spikes[i] = np.array(sall)
        else:
            start = np.array(times_all['rat_' + np.str.lower(rat_name) + '_' + sess_name[:3] + day_name])[:,0]
            start = start[start<max_time0]

            end = np.array(times_all['rat_' + np.str.lower(rat_name) + '_' + sess_name[:3] + day_name])[:,1]
            end = end[end<=max_time0]

            for i in range(len(spikes)):
                sall = []
                s = spikes[i].copy()
                for r in range(len(start)):
                    min_time1 = start[r]
                    max_time1 = end[r]
                    stemp = s[(s>= min_time1) & (s< max_time1)]
                    sall.extend(stemp)
                    min_time1 = t[ssp[sw[ss-1]]-1]
                    max_time1 = t[ssp[sw[ss]]+1]
                spikes[i] = np.array(sall)
        if bStart:
            for i in range(len(spikes)):
                spikes[i] -= min_time0
        return spikes
    else:
        res = 100000
        dt = 1000
        min_time = min_time0*res
        max_time = max_time0*res
        if bSmooth:
            sigma = 5000
            if np.str.upper(sess_name[:3]) == 'SWS':
                sigma = 2500
            if smoothing_width > -1:
                sigma = smoothing_width
            thresh = sigma*5
            num_thresh = int(thresh/dt)
            num2_thresh = int(2*num_thresh)
            sig2 = 1/(2*(sigma/res)**2)
            ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
            kerwhere = np.arange(-num_thresh,num_thresh)*dt
        if sess_name[:2] in ('OF', 'WW'):
            if bSmooth:
                tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
                spikes_bin = np.zeros((len(tt)+num2_thresh, len(spikes)))    
                for n in spikes:
                    spike_times = np.array(spikes[n]*res-min_time, dtype = int)
                    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
                    spikes_mod = dt-spike_times%dt
                    spike_times= np.array(spike_times/dt, int)
                    for m, j in enumerate(spike_times):
                        spikes_bin[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
                spikes_bin = spikes_bin[num_thresh-1:-(num_thresh+1),:]
                spikes_bin *= 1/np.sqrt(2*np.pi*(sigma/res)**2)
            else:
                tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
                spikes_bin = np.zeros((len(tt), len(spikes)), dtype = int)    
                for n in spikes:
                    spike_times = np.array(spikes[n]*res-min_time, dtype = int)
                    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
                    spikes_mod = dt-spike_times%dt
                    spike_times= np.array(spike_times/dt, int)
                    for m, j in enumerate(spike_times):
                        spikes_bin[j, n] += 1
        else:
            start = np.array(times_all['rat_' + np.str.lower(rat_name) + '_' + sess_name[:3] + day_name])[:,0]*res
            start = start[start<max_time]

            end = np.array(times_all['rat_' + np.str.lower(rat_name) + '_' + sess_name[:3] + day_name])[:,1]*res
            end = end[end<=max_time]
            
            spikes_bin = np.zeros((1,len(spikes)))
            if bSmooth:
                for r in range(len(start)):
                    min_time = start[r]
                    max_time = end[r]
                    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
                    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes)))    
                    for n in spikes:
                        spike_times = np.array(spikes[n]*res-min_time, dtype = int)
                        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
                        spikes_mod = dt-spike_times%dt
                        spike_times= np.array(spike_times/dt, int)
                        for m, j in enumerate(spike_times):
                            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
                    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
                    spikes_bin = np.concatenate((spikes_bin, spikes_temp),0)
                spikes_bin = spikes_bin[1:,:]
                spikes_bin *= 1/np.sqrt(2*np.pi*(sigma/res)**2)
            else:
                for r in range(len(start)):
                    min_time = start[r]
                    max_time = end[r]
                    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
                    spikes_temp = np.zeros((len(tt), len(spikes)), dtype = int)    
                    for n in spikes:
                        spike_times = np.array(spikes[n]*res-min_time, dtype = int)
                        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
                        spikes_mod = dt-spike_times%dt
                        spike_times= np.array(spike_times/dt, int)
                        for m, j in enumerate(spike_times):
                            spikes_temp[j, n] += 1
                    spikes_bin = np.concatenate((spikes_bin, spikes_temp),0)
                spikes_bin = spikes_bin[1:,:]   

        if (rat_name == 'R') & (day_name == 'day1'):
            if sess_name =='OF':
                valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                                              np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)
            elif sess_name == 'WW':
                valid_times = np.concatenate((np.arange(0, (18026-16925)*res/dt),
                                              np.arange((18183-16925)*res/dt, (20704-16925)*res/dt))).astype(int)
        else:
            valid_times = np.arange(len(spikes_bin[:,0]))
        spikes_bin = spikes_bin[valid_times,:]
        if (rat_name =='S') & (sess_name == 'WW'):
            spikes_bin_1 = get_spikes(rat_name, mod_name, day_name, 'WW2', bType, bSmooth,
                                                   bBinned, False, bStart, smoothing_width)
            
            spikes_bin = np.concatenate((spikes_bin, spikes_bin_1),0)

            
        if bSpeed: 
            xx, yy, aa, tt, speed = load_pos(rat_name, sess_name[:2], day_name, bSpeed = True, folder = folder)
            spikes_bin = spikes_bin[speed>2.5,:]            
            xx = xx[speed>2.5]
            yy = yy[speed>2.5]
            aa = aa[speed>2.5]
            tt = tt[speed>2.5]
            return spikes_bin, xx, yy, aa, tt
        return spikes_bin



def load_pos(rat_name, sess_name, day_name = '', bSpeed = False, folder = ''):    
    if np.str.upper(rat_name) == 'R':
        f = np.load(folder + 'rat_r_' + day_name + '_grid_modules_1_2_3.npz', allow_pickle = True)
    elif np.str.upper(rat_name) == 'Q':
        f = np.load(folder + 'rat_q_grid_modules_1_2.npz', allow_pickle = True)
    elif np.str.upper(rat_name) == 'S':
        f = np.load(folder + 'rat_s_grid_modules_1.npz', allow_pickle = True)
    else:
        print('Correct rat name was not given')
        return
    t = f['t']
    x = f['x']
    y = f['y']
    azimuth = f['azimuth']
    f.close()

    if np.str.upper(sess_name) in ('SWS', 'REM'):
        times = times_all['rat_' + np.str.lower(rat_name) + '_sleep' + day_name]
    else:
        times = times_all['rat_' + np.str.lower(rat_name) + '_' + sess_name + day_name]        

    min_time0, max_time0 = times[0]
    times = np.where((t>=min_time0) & (t<max_time0))
    x = x[times]
    y = y[times]
    t = t[times]
    azimuth = azimuth[times]

    res = 100000
    dt = 1000

    min_time = min_time0*res
    max_time = max_time0*res

    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res

    idt =  np.concatenate(([0], np.digitize(t[1:-1], tt[:])-1, [len(tt)+1]))
    idtt = np.digitize(np.arange(len(tt)), idt)-1

    idx = np.concatenate((np.unique(idtt), [np.max(idtt)+1]))
    divisor = np.bincount(idtt)
    steps = (1.0/divisor[divisor>0]) 
    N = np.max(divisor)
    ranges = np.multiply(np.arange(N)[np.newaxis,:], steps[:, np.newaxis])
    ranges[ranges>=1] = np.nan

    rangesx =x[idx[:-1], np.newaxis] + np.multiply(ranges, (x[idx[1:]] - x[idx[:-1]])[:, np.newaxis])
    xx = rangesx[~np.isnan(ranges)] 

    rangesy =y[idx[:-1], np.newaxis] + np.multiply(ranges, (y[idx[1:]] - y[idx[:-1]])[:, np.newaxis])
    yy = rangesy[~np.isnan(ranges)] 

    rangesa =azimuth[idx[:-1], np.newaxis] + np.multiply(ranges, (azimuth[idx[1:]] - azimuth[idx[:-1]])[:, np.newaxis])
    aa = rangesa[~np.isnan(ranges)] 
    if (np.str.upper(rat_name) == 'R') & (day_name == 'day1'):
        if sess_name =='OF':
            valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                                          np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)
        elif sess_name == 'WW':
            valid_times = np.concatenate((np.arange(0, (18026-16925)*res/dt),
                                          np.arange((18183-16925)*res/dt, (20704-16925)*res/dt))).astype(int)
    else:
        valid_times = np.arange(len(xx))
        
    xx = xx[valid_times]
    yy = yy[valid_times]
    aa = aa[valid_times]
    tt = tt[valid_times]
        
    if bSpeed:
        xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
        yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
        dx = (xxs[1:] - xxs[:-1])*100
        dy = (yys[1:] - yys[:-1])*100
        speed = np.sqrt(dx**2+ dy**2)/0.01
        speed = np.concatenate(([speed[0]],speed))
        if (np.str.upper(rat_name) =='S') & (sess_name == 'WW'):
            xx1, yy1, aa1, tt1, speed1 = load_pos(rat_name, 'WW2', day_name, True, folder = folder)
            xx = np.concatenate((xx,xx1))
            yy = np.concatenate((yy,yy1))
            aa = np.concatenate((aa,aa1))
            tt = np.concatenate((tt,tt1))
            speed = np.concatenate((speed,speed1))
        return xx, yy, aa, tt, speed
    if (np.str.upper(rat_name) =='S') & (sess_name == 'WW'):
        xx1, yy1, aa1, tt1 = load_pos(rat_name, 'WW2', folder = folder)
        xx = np.concatenate((xx,xx1))
        yy = np.concatenate((yy,yy1))
        aa = np.concatenate((aa,aa1))
        tt = np.concatenate((tt,tt1))
    return xx,yy,aa,tt



def plot_barcode(persistence, file_name = ''):

    diagrams_roll = {}
    filenames=glob.glob('Results/Roll/' + file_name + '_H2_roll_*')
    for i, fname in enumerate(filenames): 
        f = np.load(fname, allow_pickle = True)
        diagrams_roll[i] = list(f['diagrams'])
        f.close() 

    cs = np.repeat([[0,0.55,0.2]],3).reshape(3,3).T
    alpha=1
    inf_delta=0.1
    legend=True
    colormap=cs
    maxdim = len(persistence)-1
    dims =np.arange(maxdim+1)
    num_rolls = len(diagrams_roll)

    if num_rolls>0:
        diagrams_all = np.copy(diagrams_roll[0])
        for i in np.arange(1,num_rolls):
            for d in dims:
                diagrams_all[d] = np.concatenate((diagrams_all[d], diagrams_roll[i][d]),0)
        infs = np.isinf(diagrams_all[0])
        diagrams_all[0][infs] = 0
        diagrams_all[0][infs] = np.max(diagrams_all[0])
        infs = np.isinf(diagrams_all[0])
        diagrams_all[0][infs] = 0
        diagrams_all[0][infs] = np.max(diagrams_all[0])


    min_birth, max_death = 0,0            
    for dim in dims:
        persistence_dim = persistence[dim][~np.isinf(persistence[dim][:,1]),:]
        min_birth = min(min_birth, np.min(persistence_dim))
        max_death = max(max_death, np.max(persistence_dim))
    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta
    axis_start = min_birth - delta            
    plotind = (dims[-1]+1)*100 + 10 +1
    fig = plt.figure()
    gs = grd.GridSpec(len(dims),1)

    indsall =  0
    labels = ["$H_0$", "$H_1$", "$H_2$"]
    for dit, dim in enumerate(dims):
        axes = plt.subplot(gs[dim])
        axes.axis('off')
        d = np.copy(persistence[dim])
        d[np.isinf(d[:,1]),1] = infinity
        dlife = (d[:,1] - d[:,0])
        dinds = np.argsort(dlife)[-30:]
        dl1,dl2 = dlife[dinds[-2:]]
        if dim>0:
            dinds = dinds[np.flip(np.argsort(d[dinds,0]))]
        axes.barh(
            0.5+np.arange(len(dinds)),
            dlife[dinds],
            height=0.8,
            left=d[dinds,0],
            alpha=alpha,
            color=colormap[dim],
            linewidth=0,
        )
        indsall = len(dinds)
        if num_rolls>0:
            bins = 50
            cs = np.flip([[0.4,0.4,0.4], [0.6,0.6,0.6], [0.8, 0.8,0.8]])
            cs = np.repeat([[1,0.55,0.1]],3).reshape(3,3).T
            cc = 0
            lives1_all = diagrams_all[dim][:,1] - diagrams_all[dim][:,0]
            x1 = np.linspace(diagrams_all[dim][:,0].min()-1e-5, diagrams_all[dim][:,0].max()+1e-5, bins-2)
            
            dx1 = (x1[1] - x1[0])
            x1 = np.concatenate(([x1[0]-dx1], x1, [x1[-1]+dx1]))
            dx = x1[:-1] + dx1/2
            ytemp = np.zeros((bins-1))
            binned_birth = np.digitize(diagrams_all[dim][:,0], x1)-1
            x1  = d[dinds,0]
            ytemp =x1 + np.max(lives1_all)
            axes.fill_betweenx(0.5+np.arange(len(dinds)), x1, ytemp, color = cs[(dim)], zorder = -2, alpha = 0.3)

        axes.plot([0,0], [0, indsall], c = 'k', linestyle = '-', lw = 1)
        axes.plot([0,indsall],[0,0], c = 'k', linestyle = '-', lw = 1)
        axes.set_xlim([0, infinity])




def toroidal_alignment(rat_name, mod_name, sess_name_0, sess_name_1, day_name, bPlot = False, folder = ''):
    num_shuffle = 100
    numangsint = 51
    numangsint_1 = numangsint-1
    bins = np.linspace(0,2*np.pi, numangsint)
    bins_torus = np.linspace(0,2*np.pi, numangsint)
    sig1 = 15
    file_name_1 =  rat_name + '_' + mod_name + '_' + sess_name_0
    file_name_2 =  rat_name + '_' + mod_name + '_' + sess_name_1
    file_name =   rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1
    if len(day_name)>0:
        file_name_1 += '_' + day_name  
        file_name_2 += '_' + day_name  
        file_name += '_' + day_name  
    ############################## Load 1 ############################
    f = np.load(folder + 'Results/' + file_name_1  + '_decoding.npz',
        allow_pickle = True)
    call = f['coords']
    callbox_1 = f['coordsbox']
    times_box_1 = f['times_box']
    c11all_1 = call[:,0]
    c12all_1 = call[:,1]
    times_1 = f['times']
    f.close()    
    ############################## Load 2 ############################
    f = np.load(folder + 'Results/' + file_name_2 + '_decoding.npz',
        allow_pickle = True)
    call = f['coords']
    callbox_2 = f['coordsbox']
    times_box_2 = f['times_box']
    c11all_2 = call[:,0]
    c12all_2 = call[:,1]
    times_2 = f['times']
    f.close()    
    __, t_box_1, t_box_2  = np.intersect1d(times_box_1, times_box_2, assume_unique=True, return_indices = True)
    callbox_1 = callbox_1[t_box_1,:]
    callbox_2 = callbox_2[t_box_2,:]
    
    ############################## compare ############################
    f = np.load(folder + 'Results/' + file_name_1 + '_para.npz', allow_pickle = True)
    p1b_1 = f['p1b_1']
    p2b_1 = f['p2b_1']
    m1b_1 = f['m1b_1']
    m2b_1 = f['m2b_1']
    f.close()

    nb = 25
    numangsinttemp = 151
    x,y = np.meshgrid(np.linspace(0,3*np.pi,numangsinttemp-1), np.linspace(0,3*np.pi,numangsinttemp-1))
    nnans = ~np.isnan(m1b_1)
    mtot = m1b_1[nnans]%(2*np.pi)
    p = p1b_1.copy()
    x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
    pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
    pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
    p1 = 1
    pm00 = (p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi)
    if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
        p1 = -1
        pm00 = (2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi)

    nnans = ~np.isnan(m2b_1)
    mtot = m2b_1[nnans]%(2*np.pi)
    p = p2b_1.copy()
    x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
    pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
    pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
    p2 = 1
    pm01 = (p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi)
    if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
        p2 = -1
        pm01 = (2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi)

    cc1 = rot_coord(p1b_1,p2b_1, c11all_1, c12all_1, (p1,p2))
    if sess_name_0[:3] not in ('box'):
        cbox1 = rot_coord(p1b_1,p2b_1, callbox_1[:,0], callbox_1[:,1], (p1,p2))
    else:
        cbox1 = cc1.copy()

    f = np.load(folder + 'Results/' + file_name_2 + '_para.npz', allow_pickle = True)
    p1b_2 = f['p1b_1']
    p2b_2 = f['p2b_1']
    m1b_2 = f['m1b_1']
    m2b_2 = f['m2b_1']
    f.close()

    nb = 25
    numangsinttemp = 151
    x,y = np.meshgrid(np.linspace(0,3*np.pi,numangsinttemp-1), np.linspace(0,3*np.pi,numangsinttemp-1))
    nnans = ~np.isnan(m1b_2)
    mtot = m1b_2[nnans]%(2*np.pi)
    p = p1b_2.copy()
    x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
    pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
    pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
    p1 = 1
    pm11 = (p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi)
    if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
        p1 = -1
        pm11 = (2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi)


    nnans = ~np.isnan(m2b_2)
    mtot = m2b_2[nnans]%(2*np.pi)
    p = p2b_2.copy()
    x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
    pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
    pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
    p2 = 1
    pm12 = (p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi)
    if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
        p2 = -1
        pm12 = (2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi)
    cc2 = rot_coord(p1b_2,p2b_2, c11all_2, c12all_2, (p1,p2))

    if sess_name_0[:3] not in ('box'):
        cbox2 = rot_coord(p1b_2,p2b_2, callbox_2[:,0], callbox_2[:,1], (p1,p2))
    else:
        cbox2 = cc2.copy()

    pshift = np.arctan2(np.mean(np.sin(cbox1 - cbox2),0), np.mean(np.cos(cbox1 - cbox2),0))

    cbox1 = (cbox1 - pshift)%(2*np.pi)
    cc1 = (cc1 - pshift)%(2*np.pi)

    np.savez_compressed(folder + 'Results/' + file_name + '_alignment_dec',
                        pshift = pshift,
                        cbox = cbox1,
                        times_box = times_box_1[t_box_1],
                        csess = cc1,
                        times = times_1,
                        cbox_of = cbox2,
                        c_of = cc2,
                        times_of = times_2
                       )
    if bPlot:
        xx, yy, __, __, speed = load_pos(rat_name, 'OF', day_name, bSpeed = True, folder = folder)
        xx = xx[speed>2.5]
        yy = yy[speed>2.5]
        xx = xx[times_box_1[t_box_1]]
        yy = yy[times_box_1[t_box_1]]
        
        pm00, pm01, xedge,yedge = get_ang_hist(cbox1[:,0], 
            cbox1[:,1], xx,yy)
        plt.viridis()
        fig, ax = plt.subplots(1,1)
        ax.imshow(np.cos(pm00), origin = 'lower')
        ax.set_aspect('equal', 'box')
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        fig, ax = plt.subplots(1,1)
        ax.imshow(np.cos(pm01), origin = 'lower')
        ax.set_aspect('equal', 'box')
        ax.set_xticks([], [])
        ax.set_yticks([], [])



def rot_coord(params1,params2, c1, c2, p):    
    rot_mat = np.zeros((2,2))
    if np.abs(np.cos(params1[0])) < np.abs(np.cos(params2[0])):        
        cc1 = c2.copy()
        cc2 = c1.copy()
        y = params1.copy()
        x = params2.copy()
        p = np.flip(p)
    else:   
        cc1 = c1.copy()
        cc2 = c2.copy()
        x = params1.copy()
        y = params2.copy()  
    if p[1] ==-1:
        cc2 = (2*np.pi-cc2)
    if p[0] ==-1:
        cc1 = (2*np.pi-cc1)
    alpha = (y[0]-x[0])
    if (alpha < 0) & (np.abs(alpha) > np.pi/2):
        cctmp = cc2.copy()
        cc2 = cc1.copy()
        cc1 = cctmp
    if (alpha < 0) & (np.abs(alpha) < np.pi/2):
        cc1 = (2*np.pi-cc1 +  np.pi/3*cc2)
    elif np.abs(alpha) > np.pi/2:
        cc2 = (cc2 + np.pi/3*cc1)
    return np.concatenate((cc1[:,np.newaxis], cc2[:,np.newaxis]),1)%(2*np.pi)


def fit_para(rat_name, mod_name, sess_name, day_name, folder = ''):
    ############################## Fit parallelogram to space ############################

    file_name =  rat_name + '_' + mod_name + '_' + sess_name 
    if len(day_name)>0:
        file_name += '_' + day_name  
    ############################## Load 1 ############################
    f = np.load(folder + 'Results/' + file_name  + '_decoding.npz',
        allow_pickle = True)
    c11all_orig1 = f['coordsbox'][:,0]
    c12all_orig1 = f['coordsbox'][:,1]
    times = f['times_box']
    f.close()    
    xx,yy, __, __, speed = load_pos(rat_name, 'OF', day_name, bSpeed = True, folder = folder)   
    xx = xx[speed>2.5]
    yy = yy[speed>2.5]
    xx = xx[times]
    yy = yy[times]
    
    m1b_1, m2b_1, xedge,yedge = get_ang_hist(c11all_orig1, 
        c12all_orig1, xx,yy)
    p1b_1, f1 = fit_sine_wave(m1b_1)
    p2b_1, f2 = fit_sine_wave(m2b_1)

    np.savez_compressed(folder + 'Results/' + file_name + '_para', 
        p1b_1 = p1b_1, 
        p2b_1 = p2b_1,
        m1b_1 = m1b_1, 
        m2b_1 = m2b_1, 
        xedge = xedge,
        yedge = yedge,
        fun = (f1,f2))


def get_ratemaps(cc, spk):
    numangsint = 51
    numangsint_1 = numangsint-1
    bins = np.linspace(0,2*np.pi, numangsint)
    xv, yv = np.meshgrid(bins[0:-1] + (bins[1:] -bins[:-1])/2, 
                     bins[0:-1] + (bins[1:] -bins[:-1])/2)
    pos  = np.concatenate((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis]),1)
    ccos = np.cos(pos)
    csin = np.sin(pos)
    num_neurons = len(spk[0,:])
    mtottemp = np.zeros((num_neurons,len(bins)-1,len(bins)-1))
    mcstemp = np.zeros((num_neurons,2))
    for n in range(num_neurons):
        mtot_tmp, x_edge, y_edge,c2 = binned_statistic_2d(cc[:,0], cc[:,1], 
                                              spk[:,n], statistic='mean', 
                                              bins=bins, range=None, expand_binnumbers=True)
        mtot_tmp = np.rot90(mtot_tmp, 1).T
        mtottemp[n,:,:] = mtot_tmp.copy()
        mtot_tmp = mtot_tmp.flatten()
        nans  = ~np.isnan(mtot_tmp) 
        centcos = np.sum(np.multiply(ccos[nans,:],mtot_tmp[nans,np.newaxis]),0)
        centsin = np.sum(np.multiply(csin[nans,:],mtot_tmp[nans,np.newaxis]),0)
        mcstemp[n,:] = np.arctan2(centsin,centcos)%(2*np.pi)
    return mtottemp, mcstemp

def get_corr_dist(masscenters_1,masscenters_2, mtot_1, mtot_2):
    sig = 2.75
    num_shuffle = 1000
    numangsint = 51
    num_neurons = len(masscenters_1[:,0])
    cells_all = np.arange(num_neurons)
    corr = np.zeros(num_neurons)
    corr1 = np.zeros(num_neurons)
    corr2 = np.zeros(num_neurons)

    for n in cells_all:
        m1 = mtot_1[n,:,:].copy()
        m1[np.isnan(m1)] = np.mean(m1[~np.isnan(m1)])
        m2 = mtot_2[n,:,:].copy()
        m2[np.isnan(m2)] = np.mean(m2[~np.isnan(m2)])
        m1 = smooth_tuning_map(np.rot90(m1), numangsint, sig, bClose = False)
        m2 = smooth_tuning_map(np.rot90(m2), numangsint, sig, bClose = False)
        corr[n] = pearsonr(m1.flatten(), m2.flatten())[0]
        mtot_1[n,:,:]= m1
        mtot_2[n,:,:]= m2


    dist =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2),
                                  np.cos(masscenters_1 - masscenters_2))),1))
    dist_shuffle = np.zeros((num_shuffle, num_neurons))
    corr_shuffle = np.zeros((num_shuffle, num_neurons))
    np.random.seed(47)
    for i in range(num_shuffle):
        inds = np.arange(num_neurons)
        np.random.shuffle(inds)
        for n in cells_all:
            corr_shuffle[i,n] = pearsonr(mtot_1[n,:,:].flatten(), mtot_2[inds[n],:,:].flatten())[0]
        dist_shuffle[i,:] =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2[inds,:]),
                np.cos(masscenters_1 - masscenters_2[inds,:]))),1))

    return corr, dist, corr_shuffle, dist_shuffle


def plot_cumulative_stat(stat, stat_shuffle, stat_range, stat_scale, xs, ys, xlim, ylim):
    num_neurons = len(stat)
    num_shuffle = len(stat_shuffle)
    fig = plt.figure()
    ax = plt.axes()
    numbins = 30
    meantemp = np.zeros(numbins)
    stat_all = np.array([])
    mean_stat_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        stat_all = np.concatenate((stat_all, stat_shuffle[i]))
        mean_stat_all[i] = np.mean(stat_shuffle[i])

    meantemp1 = np.histogram(stat_all, range = stat_range, bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(stat, range = stat_range, bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= stat_scale
    ax.plot(x, meantemp, c = 'g', alpha = 0.8, lw = 5)
    ax.plot(x,y, c = 'r', alpha = 0.8, lw = 5)

    ax.set_xticks(xs)
    ax.set_xticklabels(np.zeros(len(xs),dtype=str))
    ax.xaxis.set_tick_params(width=1, length =5)
    ax.set_yticks(ys)
    ax.set_yticklabels(np.zeros(len(ys),dtype=str))
    ax.yaxis.set_tick_params(width=1, length =5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()

    ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*1.4))

    plt.gca().axes.spines['top'].set_visible(False)
    plt.gca().axes.spines['right'].set_visible(False)


def radialk(data_in, k = 10, startindex = -1):    
    n = data_in.shape[0]
    if startindex == -1:
        np.random.seed(41)
        startindex = np.random.randint(n)
    n = data_in.shape[0]
    inds = np.zeros((n,), dtype=int)
    i = startindex
    j = 1
    minDist = 0
    inds = np.zeros((n, ), dtype=int)
    inds1 = np.arange(n, dtype=int)
    cl = np.zeros((n+1,), dtype=int)
    while j < n+1:
        inds[i] = j
        dists = cdist(data_in[i, :].reshape(1, -1), data_in[inds1, :])[0]
        dists -= np.sort(dists)[k]
        cl[inds1[dists<=0]] = i
        inds1 = inds1[dists>0]
        j = j+1
        if len(inds1)>k:
             i = inds1[np.argmin(dists[dists>0])]
        else:
            break
    inds = np.where(inds)[0]
    return inds

def plot_neuron(ax, rat_name,  mod_name, day_name):
    spikes_OF, xx, yy, __, __ = get_spikes(rat_name, mod_name, day_name, 'OF', 
                                           bType = 'pure', bSmooth = False, bSpeed = True)
    spikes_OF = spikes_OF.astype(float)
    xmin = xx.min()
    xx -= xmin
    ymin = yy.min()
    yy -= ymin

    if (rat_name == 'S'):
        spktimes = np.argsort(gaussian_filter1d(spikes_OF[:,17],15))[-15000:] #np.where(sspk)[0
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.8, 0.8,0.8,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'Q') & (mod_name == '1'):
        spktimes = np.argsort(gaussian_filter1d(spikes_OF[:,54],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.8, 0.8,0.8,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'Q') & (mod_name == '2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_OF[:,55],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.5, 0.5,0.5,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'R') & (mod_name == '1') & (day_name == 'day2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_OF[:,18],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.8, 0.8,0.8,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'R') & (mod_name == '1') & (day_name == 'day1'):
        spktimes = np.argsort(gaussian_filter1d(spikes_OF[:,54],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.8, 0.8,0.8,1]], alpha = 0.6, s = 10)    
    elif (rat_name == 'R') & (mod_name == '2') & (day_name == 'day2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_OF[:,10],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.5, 0.5,0.5,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'R') & (mod_name == '2') & (day_name == 'day1'):
        spktimes = np.argsort(gaussian_filter1d(spikes_OF[:,27],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.5, 0.5,0.5,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'R') & (mod_name == '3') & (day_name == 'day2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_OF[:,89],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.2, 0.2,0.2,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'R') & (mod_name == '3') & (day_name == 'day1'):
        spktimes = np.argsort(gaussian_filter1d(spikes_OF[:,44],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.2, 0.2,0.2,1]], alpha = 0.6, s = 10)
    return ax



def plot_phase_distribution(masscenters_1, masscenters_2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')
    num_neurons = len(masscenters_1[:,0])
    for i in np.arange(num_neurons):
        ax.scatter(masscenters_1[i,0], masscenters_1[i,1], s = 10, c = 'r')
        ax.scatter(masscenters_2[i,0], masscenters_2[i,1], s = 10, c = 'g')
        line = masscenters_1[i,:] - masscenters_2[i,:]
        dline = line[1]/line[0]
        if line[0]< - np.pi and line[1] < -np.pi:
            line = (-2*np.pi + masscenters_2[i,:]) - masscenters_1[i,:]
            dline = line[1]/line[0]
            if (masscenters_1[i,1] + (- masscenters_1[i,0])*dline)>0:
                ax.plot([masscenters_1[i,0], 0],
                        [masscenters_1[i,1], masscenters_1[i,1] + (- masscenters_1[i,0])*dline],
                        c = 'b', alpha = 0.5)
               
                ax.plot([2*np.pi,2*np.pi + -(masscenters_1[i,1] + (- masscenters_1[i,0])*dline)/dline], 
                        [masscenters_1[i,1] + (- masscenters_1[i,0])*dline, 0],
                        c = 'b', alpha = 0.5)
               
                ax.plot([2*np.pi + -(masscenters_1[i,1] + (- masscenters_1[i,0])*dline)/dline, 
                         masscenters_2[i,0]], 
                        [2*np.pi,masscenters_2[i,1]],
                        c = 'b', alpha = 0.5)
            else:
                ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (- masscenters_1[i,1])/dline],
                        [masscenters_1[i,1], 0],
                        c = 'b', alpha = 0.5)
               
                ax.plot([masscenters_1[i,0] + (- masscenters_1[i,1])/dline, 0],
                        [2*np.pi, 2*np.pi + -(masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline], 
                        c = 'b', alpha = 0.5)
               
                ax.plot([2*np.pi, 
                         masscenters_2[i,0]], 
                        [2*np.pi + -(masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline,
                        masscenters_2[i,1]],
                        c = 'b', alpha = 0.5)
        elif line[0]> np.pi and line[1] >np.pi:
            line = (2*np.pi + masscenters_2[i,:]) - masscenters_1[i,:]
            dline = line[1]/line[0]
            if (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline)<2*np.pi:
               ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline],
                       
                       [masscenters_1[i,1],2*np.pi],
                       c = 'b', alpha = 0.5)
               ax.plot([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline, 2*np.pi],
                       [0,(2*np.pi - (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline))*dline], 
                       c = 'b', alpha = 0.5)
               ax.plot([0,masscenters_2[i,0]],
                       [(2*np.pi - (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline))*dline, 
                        masscenters_2[i,1]], 
                       c = 'b', alpha = 0.5)          
            else:
                ax.plot([masscenters_1[i,0],2*np.pi],
                        [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                        c = 'b', alpha = 0.5)
              
                ax.plot([0,(2*np.pi - (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline], 
                        [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, 2*np.pi],
                        c = 'b', alpha = 0.5)
               
                ax.plot([(2*np.pi - (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline, 
                         masscenters_2[i,0]], 
                        [0,masscenters_2[i,1]],
                        c = 'b', alpha = 0.5)#
        elif line[0]>np.pi and line[1] <-np.pi:  
            line = [2*np.pi + masscenters_2[i,0], -2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
            dline = line[1]/line[0]            
            if (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline)>0:
                ax.plot([masscenters_1[i,0],2*np.pi],
                        [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                        c = 'b', alpha = 0.5)
               
                ax.plot([0,(-(masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline], 
                        [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, 0],
                        c = 'b', alpha = 0.5)
                
                ax.plot([(-(masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline, 
                         masscenters_2[i,0]], 
                        [2*np.pi,masscenters_2[i,1]],
                        c = 'b', alpha = 0.5)

            else:
                line = [2*np.pi + masscenters_2[i,0], -2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
                dline = line[1]/line[0]
                ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (- masscenters_1[i,1])/dline],
                        [masscenters_1[i,1], 0],
                        c = 'b', alpha = 0.5)
               
                ax.plot([masscenters_1[i,0] + (- masscenters_1[i,1])/dline, 2*np.pi], 
                        [2*np.pi, 2*np.pi + (2*np.pi- masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline],
                        c = 'b', alpha = 0.5)
                
                ax.plot([0, masscenters_2[i,0]], 
                        [2*np.pi + (2*np.pi- masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline,masscenters_2[i,1]],
                        c = 'b', alpha = 0.5)
        elif line[0]<-np.pi and line[1] >np.pi:
            line = [-2*np.pi + masscenters_2[i,0], 2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
            dline = line[1]/line[0]
            if ((masscenters_1[i,1] + -(masscenters_1[i,0])*dline)<2*np.pi):

                ax.plot([masscenters_1[i,0],0],
                        [masscenters_1[i,1], masscenters_1[i,1] + -(masscenters_1[i,0])*dline],
                        c = 'b', alpha = 0.5)
               
                ax.plot([2*np.pi, 2*np.pi + (2*np.pi - (masscenters_1[i,1] + -(masscenters_1[i,0])*dline))/dline], 
                        [masscenters_1[i,1] + -(masscenters_1[i,0])*dline, 2*np.pi],
                        c = 'b', alpha = 0.5)
                
                ax.plot([2*np.pi + (2*np.pi - (masscenters_1[i,1] + -(masscenters_1[i,0])*dline))/dline, 
                         masscenters_2[i,0]], 
                        [0,masscenters_2[i,1]],
                        c = 'b', alpha = 0.5)
            else:
                ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline],
                        [masscenters_1[i,1], 2*np.pi],
                        c = 'b', alpha = 0.5)
               
                ax.plot([masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline, 0], 
                        [0, 0 + -(masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline)*dline],
                        c = 'b', alpha = 0.5)
                
                ax.plot([2*np.pi, masscenters_2[i,0]], 
                        [0 + -(masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline)*dline,masscenters_2[i,1]],
                        c = 'b', alpha = 0.5)

        elif line[0]< -np.pi:
            line = [(2*np.pi + masscenters_1[i,0]), masscenters_1[i,1]] - masscenters_2[i,:]
            dline = line[1]/line[0]
            ax.plot([masscenters_2[i,0],2*np.pi],
                    [masscenters_2[i,1], masscenters_2[i,1] + (2*np.pi - masscenters_2[i,0])*dline], 
                    alpha = 0.5, c = 'b')            
            ax.plot([0,masscenters_1[i,0]],
                    [masscenters_2[i,1] + (2*np.pi - masscenters_2[i,0])*dline, masscenters_1[i,1]], 
                    alpha = 0.5, c = 'b')
        elif line[0]> np.pi:
            line = [ masscenters_2[i,0]+ 2*np.pi, masscenters_2[i,1]] - masscenters_1[i,:]
            dline = line[1]/line[0]


            ax.plot([masscenters_1[i,0],2*np.pi],
                    [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                    c = 'b', alpha = 0.5)
            ax.plot([0,masscenters_2[i,0]],
                    [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, masscenters_2[i,1]], 
                    alpha = 0.5, c = 'b')
        elif line[1]< -np.pi:
            line = [ masscenters_1[i,0], (2*np.pi + masscenters_1[i,1])] - masscenters_2[i,:]
            dline = line[1]/line[0]

            ax.plot([masscenters_2[i,0], masscenters_2[i,0] + (2*np.pi - masscenters_2[i,1])/dline], 
                    [masscenters_2[i,1],2*np.pi], alpha = 0.5, c = 'b'),
            ax.plot([masscenters_1[i,0] - masscenters_1[i,1]/dline,masscenters_1[i,0]],
                    [0, masscenters_1[i,1]], 
                    alpha = 0.5, c = 'b')
        elif line[1]> np.pi:
            line = [ masscenters_2[i,0], masscenters_2[i,1]+ 2*np.pi] - masscenters_1[i,:]
            dline = line[1]/line[0]

            ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline], 
                    [masscenters_1[i,1], 2*np.pi], alpha = 0.5, c = 'b'),

            ax.plot([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline,masscenters_2[i,0]],
                    [0, masscenters_2[i,1]], 
                    alpha = 0.5, c = 'b')
        else:
            ax.plot([masscenters_1[i,0],masscenters_2[i,0]],
                    [masscenters_1[i,1],masscenters_2[i,1]], 
                    alpha = 0.5, c = 'b')

    ax.plot([0,0], [0,2*np.pi], c = 'k')
    ax.plot([0,2*np.pi], [0,0], c = 'k')
    ax.plot([2*np.pi,2*np.pi], [0,2*np.pi], c = 'k')
    ax.plot([0,2*np.pi], [2*np.pi,2*np.pi], c = 'k')

    r_box = transforms.Affine2D().skew_deg(15,15)
    for x in ax.images + ax.lines + ax.collections + ax.get_xticklabels() + ax.get_yticklabels():
        trans = x.get_transform()
        x.set_transform(r_box+trans) 
        if isinstance(x, PathCollection):
            transoff = x.get_offset_transform()
            x._transOffset = r_box+transoff 
    ax.set_xlim([0,2*np.pi + 3/5*np.pi])
    ax.set_ylim([0,2*np.pi + 3/5*np.pi])
    ax.set_aspect('equal', 'box')

def random_combinations(matrix, size):
    seen = set()
    n = len(matrix)
    while True:
        new_sample = tuple(sorted(random.sample(range(n), size)))
        if new_sample not in seen:
            seen.add(new_sample)
            yield tuple(matrix[i] for i in new_sample)

def getspatialpriorpairs(nn, periodicprior):
    pairs = []
    def addonein(iii, jjj):
        if(len(np.ravel(pairs))<1):
            pairs.append([iii, jjj])
            return
        for i in range(len(pairs)):
            aa = pairs[i]
            if(iii==aa[0] and jjj==aa[1]):
                return
            if(iii==aa[1] and jjj==aa[0]):
                return
        pairs.append([iii,jjj])

    for i in range(nn):
        for j in range(nn):
            for m in range(nn):
                for n in range(nn):
                    if(abs(i-m)==1 and (j-n)==0):
                        addonein(i*nn+j, m*nn+n)
                    if(abs(i-m)==0 and abs(j-n)==1):
                        addonein(i*nn+j, m*nn+n)
                    if(abs(i-m)==1 and abs(j-n)==1):
                        addonein(i*nn+j, m*nn+n)
                    if(periodicprior):
                        if(abs(i-m)==nn-1 and (j-n)==0):
                            addonein(i*nn+j, m*nn+n)
                        if(abs(i-m)==0 and abs(j-n)==nn-1):
                            addonein(i*nn+j, m*nn+n)
                        if(abs(i-m)==nn-1 and abs(j-n)==nn-1):
                            addonein(i*nn+j, m*nn+n)
                        if(abs(i-m)==1 and abs(j-n)==nn-1):
                            addonein(i*nn+j, m*nn+n)
                        if(abs(i-m)==nn-1 and abs(j-n)==1):
                            addonein(i*nn+j, m*nn+n)

    pairs = np.array(pairs)
    sortedpairs = []
    for i in range(nn*nn):
        kpairs = []
        for j in range(len(pairs[:,0])):
            ii, jj = pairs[j,:]
            if(i == ii or i == jj):
                kpairs.append(pairs[j,:])
        kpairs = np.array(kpairs)
        sortedpairs.append(kpairs)

    return(pairs, sortedpairs)

def fitmodel(y, covariates, LAM, periodicprior=False, GoGaussian = False):
    K = np.shape(covariates)[0]
    T = len(y)
    m = np.mean(y)
    BC = np.zeros(K)
    for j in range(K):
        BC[j] = np.mean(y * covariates[j,:])

    nn = int(np.floor(np.sqrt(float(K))))
    if(LAM==0):
        pairs, sortedpairs = ([], [])
    else:
        pairs, sortedpairs = getspatialpriorpairs(nn, periodicprior)
    if(GoGaussian == False):
        finthechat = np.sum(np.ravel(np.log(factorial(y))))

    def singleiter(vals, showvals=False):
        P = np.ravel(vals)
        H = np.dot(P,covariates)

        if(GoGaussian):
            guyH = H
        else:
            expH = np.exp(H)
            guyH = expH

        dP = np.zeros(K)
        for j in range(K):
            pp = 0.
            if LAM > 0:
                if(len(np.ravel(sortedpairs))>0):
                    kpairs = sortedpairs[j]
                    if(len(np.ravel(kpairs))>0):
                        for k in range(len(kpairs[:,0])):
                            ii, jj = kpairs[k,:]
                            if(j == ii):
                                pp += LAM*(P[ii] - P[jj])
                            if(j == jj):
                                pp += -1.*LAM*(P[ii] - P[jj])
            dP[j] = BC[j] - np.mean(guyH*covariates[j,:]) - pp/T

        pp = 0.
        if LAM > 0:
            for k in range(len(pairs[:,0])):
                pp += LAM*(P[(pairs[k,0])] - P[(pairs[k,1])])**2

        if(GoGaussian):
            L = -np.sum( (y-guyH)**2 ) - pp
        else:
            L = np.sum(np.ravel(y*H - expH)) - finthechat - pp
        return -L, -dP


    def simplegradientdescent(vals, numiters):
        P = vals
        for i in range(0,numiters,1):
            L, dvals = singleiter(vals, False)
            P -= 0.8 * dvals
            if(np.mod(i,1)==0):
                print(L, np.max(P), np.min(P))

        return P, L
    vals, Lmod = simplegradientdescent(np.zeros(K), 2)
    res = minimize(singleiter, vals, method='L-BFGS-B', jac = True, 
#                   options={'disp': True, 'maxiter': 2})
                   options={'ftol' : 1e-5, 'disp': False})
    vals = res.x + 0.
    vals, Lmod = simplegradientdescent(vals, 2)
    P = vals
    return P


def glm2d(xxss, ys, num_bins, periodicprior,  LAM, GoGaussian, nF):
    T = len(xxss[:,0])
    tmp = np.floor(T/nF)
    xxss[:,0] = normit(xxss[:,0])
    xxss[:,1] = normit(xxss[:,1])

    xvalscores = np.zeros(nF)
    P = np.zeros((num_bins**2, nF))
    LL = np.zeros(T)
    yt = np.zeros(T)
    tmp = np.floor(T/nF)
    for i in range(nF):
        fg = np.ones(T)
        fg = fg==1
        X_space = preprocess_dataX2(xxss[fg,:], num_bins)
        P[:, i] = fitmodel(ys[fg], X_space, LAM, periodicprior, GoGaussian)

        xt = xxss#[~fg,:]
        X_test = X_space#preprocess_dataX2(xt, num_bins)

        if(GoGaussian):
            yt[fg] = np.dot(P[:, i], X_test)        
            LL[fg] = -np.sum( (ys-yt)**2 )             
        else:
            H = np.dot(P[:, i], X_test)
            expH = np.exp(H)
            yt[fg] = expH
            finthechat = (np.ravel(np.log(factorial(ys[fg]))))
            LL[fg] = (np.ravel(ys[fg]*H - expH)) - finthechat
    if GoGaussian:
        leastsq = np.sum( (ys-yt)**2)
        ym = np.mean(ys)
        expl_deviance =  (1. - leastsq/np.sum((ys-ym)**2))
    else:
        LLnull = np.zeros(T)
        P_null = np.zeros((1, nF))
        yt = np.zeros(T)
        for i in range(nF):
            fg = np.ones(T)
            if(i<nF):
                fg[(int(tmp*i)):(int(tmp*(i+1)))] = 0
            else:
                fg[-(int(tmp)):] = 0
            fg = fg==1
            X_space = np.transpose(np.ones((sum(fg),1)))
            X_test = np.transpose(np.ones((sum(~fg),1)))
            P_null[:, i] = fitmodel(ys[fg], X_space, 0, False, GoGaussian)

            if(GoGaussian):
                yt[~fg] = np.dot(P_null[:, i], X_test)      
                LLnull[~fg] = -np.sum( (ys[~fg]-yt[~fg])**2 )     
            else:
                H = np.dot(P_null[:, i], X_test)
                expH = np.exp(H)
                yt[~fg] = expH
                finthechat = (np.ravel(np.log(factorial(ys[~fg]))))
                LLnull[~fg] = (np.ravel(ys[~fg]*H - expH)) - finthechat

        def getpoissonsaturatedloglike(S):
            Sguys = S[S>0.00000000000001] ## note this is same equation as below with Sguy = exp(H)
            return np.sum(np.ravel(Sguys*np.log(Sguys) - Sguys)) - np.sum(np.ravel(np.log(factorial(S))))
        if GoGaussian:
            LS = 0
        else:
            LS = getpoissonsaturatedloglike(ys[~np.isinf(LL)]) 
        
        expl_deviance = 1 - (np.sum(LS) - np.sum(LL[~np.isinf(LL)]))/(np.sum(LS) - np.sum(LLnull[~np.isinf(LL)]))
    return expl_deviance
