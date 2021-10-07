from datetime import datetime 
import functools
import glob
from gudhi.clustering.tomato import Tomato
import h5py
import matplotlib
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import numba
import numpy as np
from ripser import Rips, ripser
from scipy import stats, signal, optimize
from scipy.ndimage import gaussian_filter,  gaussian_filter1d, rotate
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.special import factorial
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.stats
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
import scipy.io as sio
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score
import sys
import time
from utils import *
from scipy.ndimage import binary_dilation, binary_closing
from scipy.stats import multivariate_normal


_LOG_2PI = np.log(2 * np.pi)

colors_envs = {}
colors_envs['roger_mod1_box'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod3_box'] = [[0.7,0.7,0]]
colors_envs['roger_mod4_box'] = [[0.7,0.,0.7]]
colors_envs['roger_mod1_box_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_box_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_box_rec2'] = [[0.7,0.,0.7]]
colors_envs['quentin_mod1_box'] = [[0.3,0.7,0.3]]
colors_envs['quentin_mod2_box'] = [[0.,0.7,0.7]]
colors_envs['shane_mod1_box'] = [[0.3,0.3,0.7]]
colors_envs['roger_mod1_maze'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod3_maze'] = [[0.7,0.7,0]]
colors_envs['roger_mod4_maze'] = [[0.7,0.,0.7]]
colors_envs['quentin_mod1_maze'] = [[0.3,0.7,0.3]]
colors_envs['quentin_mod2_maze'] = [[0.,0.7,0.7]]
colors_envs['shane_mod1_maze'] = [[0.3,0.3,0.7]]
colors_envs['roger_mod1_sws'] = 'b'
colors_envs['roger_mod2_sws'] = 'r'
colors_envs['roger_mod3_sws'] = 'c'
colors_envs['quentin_mod1_sws'] = 'm'
colors_envs['quentin_mod2_sws'] = 'g'
colors_envs['shane_mod1_sws'] = 'y'
colors_envs['roger_mod1_rem'] = 'b'
colors_envs['roger_mod2_rem'] = 'r'
colors_envs['roger_mod3_rem'] = 'c'
colors_envs['quentin_mod1_rem'] = 'm'
colors_envs['quentin_mod2_rem'] = 'g'
colors_envs['shane_mod1_rem'] = 'y'

colors_mods = {}
colors_mods['R1'] = [[0.7,0.3,0.3]]
colors_mods['R2'] = [[0.7,0.7,0]]
colors_mods['R3'] = [[0.7,0.,0.7]]
colors_mods['Q1'] = [[0.3,0.7,0.3]]
colors_mods['Q2'] = [[0.,0.7,0.7]]
colors_mods['S1'] = [[0.3,0.3,0.7]]


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
    return components, evecs, evals[:dim]


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
    dk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
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

## Note, the following funtion is imported from the scikit-tda
def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    colormap1 = "default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    rel_life= False,
    legend=True,
    show=False,
    ax=None,
    torus_colors = [],
    lw = 2.5,
    cs = ['#1f77b4','#ff7f0e', '#2ca02c', '#d62728']

):
    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = [
            "$H_0$",
            "$H_1$",
            "$H_2$",
            "$H_3$",
            "$H_4$",
            "$H_5$",
            "$H_6$",
            "$H_7$",
            "$H_8$",
        ]

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if len(plot_only)>0:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
    aspect = 'equal'
    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:
        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
#        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    i = 0
    for dgm, label in zip(diagrams, labels):
        c = cs[plot_only[i]]
        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none", c = c)
        i += 1
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    if len(torus_colors)>0:
        births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
        deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
        deaths1[np.isinf(deaths1)] = 0
        #lives1 = deaths1-births1
        #inds1 = np.argsort(lives1)
        inds1 = np.argsort(deaths1)
        ax.scatter(diagrams[1][inds1[-1], 0], diagrams[1][inds1[-1], 1], 
                   10*size, linewidth =lw, edgecolor=torus_colors[0], facecolor = "none")
        ax.scatter(diagrams[1][inds1[-2], 0], diagrams[1][inds1[-2], 1], 
                   10*size, linewidth =lw, edgecolor=torus_colors[1], facecolor = "none")
        
        
        births2 = diagrams[2][:, ] #the time of birth for the 1-dim classes
        deaths2 = diagrams[2][:, 1] #the time of death for the 1-dim classes
        deaths2[np.isinf(deaths2)] = 0
        #lives2 = deaths2-births2
        #inds2 = np.argsort(lives2)
        inds2 = np.argsort(deaths2)
#        print(lives2, births2[inds2[-1]],deaths2[inds2[-1]], diagrams[2][inds2[-1], 0], diagrams[2][inds2[-1], 1])
        ax.scatter(diagrams[2][inds2[-1], 0], diagrams[2][inds2[-1], 1], 
                   10*size, linewidth =lw, edgecolor=torus_colors[2], facecolor = "none")
        
        
    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect(aspect, 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="upper right")

    if show is True:
        plt.show()
    return


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

def load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bBox = False, bSpeed = True, bConj = False):    
    if bConj:
        if bSmooth:
            spkname = 'sspikes'
        else:
            spkname = 'spikes'
        if sess_name == 'sws_c0_rec2':
            f = np.load('Data/' + rat_name + '_' + mod_name + '_sws_rec2_spikes_conj.npz', allow_pickle = True)
            spk = f[spkname]
            f.close()
            f = np.load('Data/rog1_inds.npz', allow_pickle = True)
            ind = f['ind']
            f.close()
            spk = spk[:, ind==0]    
        elif sess_name[:3] in ('sws'):
            f = np.load('Data/' + rat_name + '_' + mod_name + '_' + sess_name + '_spikes_conj.npz', allow_pickle = True)
            spk = f[spkname]
            f.close()
        else:
            f = np.load('Data/' + rat_name + '_' + mod_name + '_' + sess_name + '_spikes_conj.npz', allow_pickle = True)
            spk = f[spkname]
            f.close()

        if (sess_name in ('maze', 'box', 'box_rec2')) & bSpeed:
            f = np.load('Data/tracking_' + rat_name + '_' + sess_name + '.npz', allow_pickle = True)
            xx = f['xx']
            yy = f['yy']
            f.close()
            xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
            yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
            dx = (xxs[1:] - xxs[:-1])*100
            dy = (yys[1:] - yys[:-1])*100
            speed = np.sqrt(dx**2+ dy**2)/0.01
            speed = np.concatenate(([speed[0]],speed))
            spk = spk[speed>=2.5,:]        
        if (rat_name == 'shane') & (sess_name in ('maze')):
            f = np.load('Data/' + rat_name + '_' + mod_name + '_' + 'maze2' + '_spikes_conj.npz', allow_pickle = True)
            spk1 = f[spkname]
            f.close()
            if bSpeed:
                f = np.load('Data/tracking_' + rat_name + '_' + 'maze2' + '.npz', allow_pickle = True)
                xx = f['xx']
                yy = f['yy']
                f.close()
                xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
                yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
                dx = (xxs[1:] - xxs[:-1])*100
                dy = (yys[1:] - yys[:-1])*100
                speed = np.sqrt(dx**2+ dy**2)/0.01
                speed = np.concatenate(([speed[0]],speed))
                spk1 = spk1[speed>=2.5,:]
            spk = np.concatenate((spk, spk1),0)    
        if bBox:
            if (sess_name in ('rem_rec2', 'sws_rec2','sws_c0_rec2', 'box_rec2')) & (rat_name == 'roger'):    
                f = np.load('Data/' + rat_name + '_' + mod_name + '_' + 'box_rec2_spikes_conj.npz', allow_pickle = True)
                spk_box = f[spkname]
                f.close()
                if sess_name == 'sws_c0_rec2':
                    f = np.load('Data/rog1_inds.npz', allow_pickle = True)
                    ind = f['ind']
                    f.close()
                    spk_box = spk_box[:, ind==0]    

                f = np.load('Data/tracking_' + rat_name + '_' + 'box_rec2.npz', allow_pickle = True)
                xx_box = f['xx']
                yy_box = f['yy']
                f.close()

            else:
                f = np.load('Data/' + rat_name + '_' + mod_name + '_' + 'box_spikes_conj.npz', allow_pickle = True)
                spk_box = f[spkname]
                f.close()

                f = np.load('Data/tracking_' + rat_name + '_' + 'box.npz', allow_pickle = True)
                xx_box = f['xx']
                yy_box = f['yy']
                f.close()
            xxs = gaussian_filter1d(xx_box-np.min(xx_box), sigma = 100)
            yys = gaussian_filter1d(yy_box-np.min(yy_box), sigma = 100)
            dx = (xxs[1:] - xxs[:-1])*100
            dy = (yys[1:] - yys[:-1])*100
            speed = np.sqrt(dx**2+ dy**2)/0.01
            speed = np.concatenate(([speed[0]],speed))

            xx_box = xx_box[speed>=2.5]
            yy_box = yy_box[speed>=2.5]
            spk_box = spk_box[speed>=2.5,:]
            if (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name in ('maze', 'box')):
                spk1, spk_box1, xx_box1, yy_box1 = load_spikes(rat_name, 'mod2', sess_name, bSmooth, bBox, bConj = True) 
                print(np.shape(spk1),np.shape(spk),np.shape(spk_box), np.shape(spk_box1))
                spk = np.concatenate((spk, spk1),1)
                spk_box = np.concatenate((spk_box, spk_box1),1)
            return spk, spk_box, xx_box, yy_box
        if (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name in ('maze', 'box')):
            spk1 = load_spikes(rat_name, 'mod2', sess_name, bSmooth, bBox, bConj = True) 
            spk = np.concatenate((spk, spk1),1)
        return spk
    else:
        if bSmooth:
            spkname = 'sspikes'
        else:
            spkname = 'spikes'
        if sess_name == 'sws_c0_rec2':
            f = np.load('Data/' + rat_name + '_' + mod_name + '_sws_rec2_spikes_03.npz', allow_pickle = True)
            spk = f[spkname]
            f.close()
            f = np.load('Data/rog1_inds.npz', allow_pickle = True)
            ind = f['ind']
            f.close()
            spk = spk[:, ind==0]    
        elif sess_name[:3] in ('sws'):
            f = np.load('Data/' + rat_name + '_' + mod_name + '_' + sess_name + '_spikes_03.npz', allow_pickle = True)
            spk = f[spkname]
            f.close()
        else:
            f = np.load('Data/' + rat_name + '_' + mod_name + '_' + sess_name + '_spikes_03_5.npz', allow_pickle = True)
            spk = f[spkname]
            f.close()

        if (sess_name in ('maze', 'box', 'box_rec2')) & bSpeed:
            f = np.load('Data/tracking_' + rat_name + '_' + sess_name + '.npz', allow_pickle = True)
            xx = f['xx']
            yy = f['yy']
            f.close()
            xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
            yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
            dx = (xxs[1:] - xxs[:-1])*100
            dy = (yys[1:] - yys[:-1])*100
            speed = np.sqrt(dx**2+ dy**2)/0.01
            speed = np.concatenate(([speed[0]],speed))
            spk = spk[speed>=2.5,:]        
        if (rat_name == 'shane') & (sess_name in ('maze')):
            f = np.load('Data/' + rat_name + '_' + mod_name + '_' + 'maze2' + '_spikes_03_5.npz', allow_pickle = True)
            spk1 = f[spkname]
            f.close()
            if bSpeed:
                f = np.load('Data/tracking_' + rat_name + '_' + 'maze2' + '.npz', allow_pickle = True)
                xx = f['xx']
                yy = f['yy']
                f.close()
                xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
                yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
                dx = (xxs[1:] - xxs[:-1])*100
                dy = (yys[1:] - yys[:-1])*100
                speed = np.sqrt(dx**2+ dy**2)/0.01
                speed = np.concatenate(([speed[0]],speed))
                spk1 = spk1[speed>=2.5,:]
            spk = np.concatenate((spk, spk1),0)    
        if bBox:
            if (sess_name in ('rem_rec2', 'sws_rec2','sws_c0_rec2', 'box_rec2')) & (rat_name == 'roger'):    
                f = np.load('Data/' + rat_name + '_' + mod_name + '_' + 'box_rec2_spikes_03_5.npz', allow_pickle = True)
                spk_box = f[spkname]
                f.close()
                if sess_name == 'sws_c0_rec2':
                    f = np.load('Data/rog1_inds.npz', allow_pickle = True)
                    ind = f['ind']
                    f.close()
                    spk_box = spk_box[:, ind==0]    

                f = np.load('Data/tracking_' + rat_name + '_' + 'box_rec2.npz', allow_pickle = True)
                xx_box = f['xx']
                yy_box = f['yy']
                f.close()

            else:
                f = np.load('Data/' + rat_name + '_' + mod_name + '_' + 'box_spikes_03_5.npz', allow_pickle = True)
                spk_box = f[spkname]
                f.close()

                f = np.load('Data/tracking_' + rat_name + '_' + 'box.npz', allow_pickle = True)
                xx_box = f['xx']
                yy_box = f['yy']
                f.close()
            if bSpeed:                
                xxs = gaussian_filter1d(xx_box-np.min(xx_box), sigma = 100)
                yys = gaussian_filter1d(yy_box-np.min(yy_box), sigma = 100)
                dx = (xxs[1:] - xxs[:-1])*100
                dy = (yys[1:] - yys[:-1])*100
                speed = np.sqrt(dx**2+ dy**2)/0.01
                speed = np.concatenate(([speed[0]],speed))

                xx_box = xx_box[speed>=2.5]
                yy_box = yy_box[speed>=2.5]
                spk_box = spk_box[speed>=2.5,:]
            if (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name in ('maze', 'box')):
                spk1, spk_box1, xx_box1, yy_box1 = load_spikes(rat_name, 'mod2', sess_name, bSmooth, bBox, bSpeed) 
                print(np.shape(spk1),np.shape(spk),np.shape(spk_box), np.shape(spk_box1))
                spk = np.concatenate((spk, spk1),1)
                spk_box = np.concatenate((spk_box, spk_box1),1)
            return spk, spk_box, xx_box, yy_box
        if (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name in ('maze', 'box')):
            spk1 = load_spikes(rat_name, 'mod2', sess_name, bSmooth, bBox, bSpeed) 
            spk = np.concatenate((spk, spk1),1)
        return spk


def load_pos(rat_name, sess_name, bSpeed = False):    
    f = np.load('Data/tracking_' + rat_name + '_' + sess_name + '.npz', allow_pickle = True)
    xx = f['xx']
    yy = f['yy']
    f.close()
        
    if bSpeed:
        xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
        yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
        dx = (xxs[1:] - xxs[:-1])*100
        dy = (yys[1:] - yys[:-1])*100
        speed = np.sqrt(dx**2+ dy**2)/0.01
        speed = np.concatenate(([speed[0]],speed))
        if (rat_name =='shane') & (sess_name == 'maze'):
            xx1, yy1, speed1 = load_pos(rat_name, 'maze2', True)
            xx = np.concatenate((xx,xx1))
            yy = np.concatenate((yy,yy1))
            speed = np.concatenate((speed,speed1))
        return xx, yy, speed
    if (rat_name =='shane') & (sess_name == 'maze'):
        xx1, yy1 = load_pos(rat_name, 'maze2')
        xx = np.concatenate((xx,xx1))
        yy = np.concatenate((yy,yy1))
    return xx,yy

def load_spikes_dec(rat_name, mod_name, sess_name, sig=0, bSmooth =True):
    file_name = rat_name + '_' + mod_name + '_' + sess_name
    if bSmooth:
        fname = glob.glob('Data/' + file_name + '_sspikes_' + str(sig) + '.npz')
    else:
        fname = glob.glob('Data/' + file_name + '_spikes_' + str(sig) + '.npz')
    if len(fname)>0:
        f = np.load(fname[0], allow_pickle = True)
        spk = f['spk']
        f.close()
    else:
        f = np.load('Data/' + file_name + '_spk_times.npz', allow_pickle = True) # or change to roger_mod3_spk_times.npz
        spikes_mod21 = f['spiketimes'][()]
        f.close()

        times_all = {}
        times_all['quentin_box'] = ((27826, 31223),)
        times_all['quentin_sleep'] = ((9576, 18812),)
        times_all['quentin_maze'] = ((18977, 25355),)
        times_all['shane_box'] = ((9939, 12363),)
        times_all['shane_maze'] = ((13670, 14847),)
        times_all['shane_maze2'] = ((23186, 24936),)
        times_all['roger_box_rec2'] = ((10617, 13004),)
        times_all['roger_sleep_rec2'] = ((396, 9941),)
        times_all['roger_box'] = ((7457, 16045),)
        times_all['roger_maze'] = ((16925, 20704),)
        times_all['quentin_rem'] = ((10424.38,10565.21),
            (11514.83,11716.55),
            (12780.35,13035.9),
            (14301.05,14475.470000000001),
            (15088.98,15310.66),
            (17739.35,17797.21))
        times_all['quentin_sws'] = ((9658.5,10419.7),
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
        times_all['roger_rem_rec2'] = ((1001.72,1053.43),
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
            
        times_all['roger_sws_rec2'] = ((524.52,771.8299999999999),
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

        times1 = times_all[rat_name + '_' + sess_name]
        spk = np.zeros((1,len(spikes_mod21)))
        for start1, end1 in times1: 
            res = 100000
            dt = 1000
            min_time = start1*res
            max_time = end1*res
            tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
            if bSmooth:
                sigma = sig*100
                thresh = sigma*5
                num_thresh = int(thresh/dt)
                num2_thresh = int(2*num_thresh)
                sig2 = 1/(2*(sigma/res)**2)
                ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
                kerwhere = np.arange(-num_thresh,num_thresh)*dt
                sspikes_mod21 = np.zeros((1,len(spikes_mod21)))
                spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod21)))    
                for n in spikes_mod21:
                    spike_times = np.array(spikes_mod21[n]*res-min_time, dtype = int)
                    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
                    spikes_mod = dt-spike_times%dt
                    spike_times= np.array(spike_times/dt, int)
                    for m, j in enumerate(spike_times):
                        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
                spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
                sspikes_mod21 = np.concatenate((sspikes_mod21, spikes_temp),0)
                sspikes_mod21 = sspikes_mod21[1:,:]
                sspikes_mod21 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

                if rat_name + sess_name == 'rogerbox':
                    valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                                                  np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)
                    sspikes_mod21 = sspikes_mod21[valid_times,:]
                elif rat_name + sess_name == 'rogermaze':
                    valid_times = np.concatenate((np.arange(0, (18026-16925)*res/dt),
                                                  np.arange((18183-16925)*res/dt, (20704-16925)*res/dt))).astype(int)
                    sspikes_mod21 = sspikes_mod21[valid_times,:]
                elif rat_name + sess_name == 'shanemaze':
                    file_name1 = rat_name + '_' + mod_name + '_' + sess_name + '2'
                    f = np.load('Data/' + file_name1 + '_spk_times.npz', allow_pickle = True) # or change to roger_mod3_spk_times.npz
                    spikes_mod22 = f['spiketimes'][()]
                    f.close()
                    start1, end1 = times_all[rat_name + '_' + sess_name + '2'][0]
                    min_time = start1*res
                    max_time = end1*res
                    sspikes_mod22 = np.zeros((1,len(spikes_mod22)))
                    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
                    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod21)))    
                    for n in spikes_mod22:
                        spike_times = np.array(spikes_mod22[n]*res-min_time, dtype = int)
                        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
                        spikes_mod = dt-spike_times%dt
                        spike_times= np.array(spike_times/dt, int)
                        for m, j in enumerate(spike_times):
                            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
                    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
                    sspikes_mod22 = np.concatenate((sspikes_mod22, spikes_temp),0)
                    sspikes_mod22 = sspikes_mod22[1:,:]
                    sspikes_mod22 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)
                    sspikes_mod21 = np.concatenate((sspikes_mod21, sspikes_mod22),0)
                spk =  np.concatenate((spk, sspikes_mod21),0)
            else:
                spikes_mod21_bin = np.zeros((1,len(spikes_mod21)))
                spikes_temp = np.zeros((len(tt), len(spikes_mod21)), dtype = int)    
                for n in spikes_mod21:
                    spike_times = np.array(spikes_mod21[n]*res-min_time, dtype = int)
                    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
                    spikes_mod = dt-spike_times%dt
                    spike_times= np.array(spike_times/dt, int)
                    for m, j in enumerate(spike_times):
                        spikes_temp[j, n] += 1
                spikes_mod21_bin = np.concatenate((spikes_mod21_bin, spikes_temp),0)
                spikes_mod21_bin = spikes_mod21_bin[1:,:]
                if rat_name + sess_name == 'rogerbox':
                    valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                                                  np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)
                    spikes_mod21_bin = spikes_mod21_bin[valid_times,:]
                elif rat_name + sess_name == 'rogermaze':
                    valid_times = np.concatenate((np.arange(0, (18026-16925)*res/dt),
                                                  np.arange((18183-16925)*res/dt, (20704-16925)*res/dt))).astype(int)
                    spikes_mod21_bin = spikes_mod21_bin[valid_times,:]
                elif rat_name + sess_name == 'shanemaze':
                    file_name1 = rat_name + '_' + mod_name + '_' + sess_name + '2'
                    f = np.load('Data/' + file_name1 + '_spk_times.npz', allow_pickle = True) # or change to roger_mod3_spk_times.npz
                    spikes_mod22 = f['spiketimes'][()]
                    f.close()
                    start1, end1 = times_all[rat_name + '_' + sess_name + '2'][0]
                    min_time = start1*res
                    max_time = end1*res
                    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
                    spikes_mod22_bin = np.zeros((1,len(spikes_mod22)))
                    spikes_temp = np.zeros((len(tt), len(spikes_mod22)), dtype = int)    
                    for n in spikes_mod22:
                        spike_times = np.array(spikes_mod22[n]*res-min_time, dtype = int)
                        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
                        spikes_mod = dt-spike_times%dt
                        spike_times= np.array(spike_times/dt, int)
                        for m, j in enumerate(spike_times):
                            spikes_temp[j, n] += 1
                    spikes_mod22_bin = np.concatenate((spikes_mod22_bin, spikes_temp),0)
                    spikes_mod22_bin = spikes_mod22_bin[1:,:]            
                    spikes_mod21_bin = np.concatenate((spikes_mod21_bin, spikes_mod22_bin),0)
                spk =  np.concatenate((spk, spikes_mod21_bin),0)
        spk = spk[1:,:]
        if bSmooth:
            np.savez_compressed('Data/' + file_name + '_sspikes_' + str(sig), spk = spk)
        else:
            np.savez_compressed('Data/' + file_name + '_spikes_' + str(sig), spk = spk)    
    return spk

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
        options={'disp': False, 'gtol': 1e-10})
    p = res['x']
    fun = res['fun']
    print('fun2:' + str(fun))
    t2 = time.time()
    return p, fun
    
def plot_stripes(xx,yy, p, mtot, name):
    nb = 25
    numangsint = 151
    x,y = np.meshgrid(np.linspace(0,3*np.pi,numangsint-1), np.linspace(0,3*np.pi,numangsint-1))
    x =rotate(x, p[0]*360/(2*np.pi), reshape= False)
    fig, ax = plt.subplots(1,1)
    ax.imshow(np.cos(p[2]*x[nb:-nb,nb:-nb]+p[1]), origin = 'lower', extent = [xx[0],xx[-1],yy[0],yy[-1]])
    ax.set_aspect('equal','box')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    fig.savefig('Figs/stripes/' + name + '_cosstripes', bbox_inches='tight', pad_inches=0.02)
        
    fig, ax = plt.subplots(1,1)
    ax.imshow(np.cos(mtot), origin = 'lower', extent = [xx[0],xx[-1],yy[0],yy[-1]])
    ax.set_aspect('equal','box')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    fig.savefig('Figs/stripes/' + name + '_boxstripes.png', bbox_inches='tight', pad_inches=0.02)
    fig.savefig('Figs/stripes/' + name + '_boxstripes.pdf', bbox_inches='tight', pad_inches=0.02)
    return

def rot_para(params1,params2, p1, p2):   
    if np.abs(np.cos(params1[0])) < np.abs(np.cos(params2[0])):
        y = params1.copy()
        x = params2.copy()
        py = p1
        px = p2      
    else:
        x = params1.copy()
        y = params2.copy()
        px = p1
        py = p2

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


def plot_neuron(ax, rat_name,  mod_name,sess_name):
    __, spikes_box, xx, yy = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bBox = True)
    xmin = xx.min()
    xx -= xmin
    ymin = yy.min()
    yy -= ymin
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
     
    if (rat_name == 'shane'):
        spktimes = np.argsort(gaussian_filter1d(spikes_box[:,17],15))[-15000:] #np.where(sspk)[0
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.8, 0.8,0.8,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'quentin') & (mod_name == 'mod1'):
        spktimes = np.argsort(gaussian_filter1d(spikes_box[:,54],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.8, 0.8,0.8,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'quentin') & (mod_name == 'mod2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_box[:,55],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.5, 0.5,0.5,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name[-4:] == 'rec2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_box[:,18],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.8, 0.8,0.8,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name[-4:] != 'rec2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_box[:,54],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.8, 0.8,0.8,1]], alpha = 0.6, s = 10)    
    elif (rat_name == 'roger') & (mod_name == 'mod2') & (sess_name[-4:] == 'rec2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_box[:,10],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.5, 0.5,0.5,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'roger') & (mod_name == 'mod3') & (sess_name[-4:] != 'rec2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_box[:,27],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.5, 0.5,0.5,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'roger') & (mod_name == 'mod3') & (sess_name[-4:] == 'rec2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_box[:,89],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.2, 0.2,0.2,1]], alpha = 0.6, s = 10)
    elif (rat_name == 'roger') & (mod_name == 'mod4') & (sess_name[-4:] != 'rec2'):
        spktimes = np.argsort(gaussian_filter1d(spikes_box[:,44],15))[-15000:] #np.where(sspk)[0]
        timeinds = radialk(np.concatenate((xx[spktimes,np.newaxis], yy[spktimes,np.newaxis]),1),k=3)
        dists = squareform(pdist(np.concatenate((xx[spktimes[timeinds],np.newaxis], yy[spktimes[timeinds],np.newaxis]),1)))
        timeinds = timeinds[np.argsort(np.sort(dists,1)[:,10])[:-500]]
        spktimes = spktimes[timeinds]
        ax.scatter(xx[spktimes], yy[spktimes], c = [[0.2, 0.2,0.2,1]], alpha = 0.6, s = 10)
    print(len(timeinds))
    return ax


def newdecoding(circ_coord_sampled, data, sampled_data):
    num_neurons = len(sampled_data[0,:])
    num_interpts = len(sampled_data[:,0])
    centcosall = np.zeros((num_neurons, 2, num_interpts))
    centsinall = np.zeros((num_neurons, 2, num_interpts))
    for neurid in range(num_neurons):
        spktemp = sampled_data[:,neurid]
        centcosall[neurid,:,:] = np.multiply(np.cos(cycle*2*np.pi),spktemp)
        centsinall[neurid,:,:] = np.multiply(np.sin(cycle*2*np.pi),spktemp)
     
    mtot1 = np.zeros((len(data[:,0]),2))
    mtot2 = np.zeros((len(data[:,0]),2))
    for i in range(num_interpts):
        mtot1 += np.dot(data[:,:],centcosall[:,:,i])
        mtot2 += np.dot(data[:,:],centsinall[:,:,i])
    coords = np.arctan2(mtot2,mtot1).T%(2*np.pi)
    return(coords)



def normit(xxx):
    xx = xxx-np.min(xxx)
    xx = xx/np.max(xx)
    return(xx)
    
def unique(list1):
    return(list(set(list1)))

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
                   options={'ftol' : 1e-5, 'gtol': 1e-5,  'disp': False})
    vals = res.x + 0.
    vals, Lmod = simplegradientdescent(vals, 2)
    P = vals
    return P

def dirtyglm(xxss, ys, num_bins, periodicprior,  LAM, GoGaussian, nF):
    T = len(xxss[:,0])
    tmp = np.floor(T/nF)
    xxss[:,0] = normit(xxss[:,0])
    xxss[:,1] = normit(xxss[:,1])

    xvalscores = np.zeros(nF)
    P = np.zeros((num_bins**2, nF))
    Lvals = np.zeros(T)
    yt = np.zeros(T)
    tmp = np.floor(T/nF)
    for i in range(nF):
        fg = np.ones(T)
        if(i<nF):
            fg[(int(tmp*i)):(int(tmp*(i+1)))] = 0
        else:
            fg[-(int(tmp)):] = 0
        fg = fg==1
        X_space = preprocess_dataX2(xxss[fg,:], num_bins)

        P[:, i] = fitmodel(ys[fg], X_space, LAM, periodicprior, GoGaussian)

        xt = xxss[~fg,:]
        X_test = preprocess_dataX2(xt, num_bins)

        if(GoGaussian):
            yt[~fg] = np.dot(P[:, i], X_test) 
            Lvals[~fg] = -np.sum( (ys-H)**2 )             
        else:
            H = np.dot(P[:, i], X_test)
            expH = np.exp(H)
            yt[~fg] = expH
            finthechat = (np.ravel(np.log(factorial(ys[~fg]))))
            Lvals[~fg] = (np.ravel(ys[~fg]*H - expH)) - finthechat

    leastsq = np.sum( (ys-yt)**2)
    #print('LEAST SQ', leastsq)
    ym = np.mean(ys)
    #return (np.sum((yt-ym)**2) / np.sum((ys-ym)**2))
    return yt, (1. - leastsq/np.sum((ys-ym)**2)), P, Lvals


def compute_deviance(xxss, ys, GoGaussian, nF, P, LL):
    T = len(ys)
    num_bins = 15
    tmp = np.floor(T/nF)
    if len(P) > 0:
        xxss[:,0] = normit(xxss[:,0])
        xxss[:,1] = normit(xxss[:,1])

        LL = np.zeros(T)
        yt = np.zeros(T)
        for i in range(nF):

            fg = np.ones(T)
            if(i<nF):
                fg[(int(tmp*i)):(int(tmp*(i+1)))] = 0
            else:
                fg[-(int(tmp)):] = 0
            fg = fg==1

            X_space = preprocess_dataX2(xxss[fg,:], num_bins)
            xt = xxss[~fg,:]
            X_test = preprocess_dataX2(xt, num_bins)

            if(GoGaussian):
                yt[~fg] = np.dot(P[:, i], X_test)        
            else:
                H = np.dot(P[:, i], X_test)
                expH = np.exp(H)
                yt[~fg] = expH
                finthechat = (np.ravel(np.log(factorial(ys[~fg]))))
                LL[~fg] = (np.ravel(ys[~fg]*H - expH)) - finthechat
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
            yt[~fg] = np.dot(P[:, i], X_test)        
        else:
            H = np.dot(P_null[:, i], X_test)
            expH = np.exp(H)
            yt[~fg] = expH
            finthechat = (np.ravel(np.log(factorial(ys[~fg]))))
            LLnull[~fg] = (np.ravel(ys[~fg]*H - expH)) - finthechat

    def getpoissonsaturatedloglike(S):
        Sguys = S[S>0.00000000000001] ## note this is same equation as below with Sguy = exp(H)
        return np.sum(np.ravel(Sguys*np.log(Sguys) - Sguys)) - np.sum(np.ravel(np.log(factorial(S))))
    
    LS = getpoissonsaturatedloglike(ys[~np.isinf(LL)]) 
    return 1 - (np.sum(LS) - np.sum(LL[~np.isinf(LL)]))/(np.sum(LS) - np.sum(LLnull[~np.isinf(LL)]))


def get_coords_box(rat_name, mod_name, sess_name, c1 = []):

    file_name = rat_name + '_' + mod_name + '_' + sess_name

    if sess_name[:6] == 'sws_c0':
        sspikes = load_spikes(rat_name, mod_name, 'sws_rec2', bSmooth = True, bBox = False, bSpeed = False)
        spikes = load_spikes(rat_name, mod_name, 'sws_rec2', bSmooth = False,bBox = False, bSpeed = False)
    else:
        sspikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bBox = False, bSpeed = False)
        spikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = False,bBox = False, bSpeed = False)

    if sess_name[:3] in ('box', 'maz'):
        xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
        sspikes = sspikes[speed>2.5,:]
        spikes = spikes[speed>2.5,:]
    print(np.shape(c1))
    if len(c1[:,0])>1200:
        times = np.where(np.sum(spikes>0, 1)>=1)[0]
        sspikes = sspikes[times,:]    
        spikes = spikes[times,:]    

        times_cube = np.arange(0,len(sspikes[:,0]),5)
        movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube,:],1))[-15000:])
        movetimes = times_cube[movetimes]
        indstemp = np.arange(len(movetimes))
        np.random.shuffle(indstemp)
        indstemp = indstemp[:1200]
        coords1 = c1[movetimes[indstemp],:].T
    elif len(c1)==1200:
        coords1 *=2*np.pi
    else:
        f = np.load('Results/Orig/' + file_name + '_pers_analysis.npz' , allow_pickle = True)
        movetimes = f['movetimes']
        indstemp = f['indstemp']
        coords1 = f['coords']*2*np.pi
        f.close()

    num_neurons = len(sspikes[0,:])
    centcosall = np.zeros((num_neurons, 2, 1200))
    centsinall = np.zeros((num_neurons, 2, 1200))
    dspk =preprocessing.scale(sspikes[movetimes[indstemp],:])

    k = 1200
    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
    #    spktemp = spktemp/np.sum(np.abs(spktemp))
        centcosall[neurid,:,:] = np.multiply(np.cos(coords1),spktemp)
        centsinall[neurid,:,:] = np.multiply(np.sin(coords1),spktemp)
    sig = 15
    boxname = 'box'
    if (rat_name == 'roger') and (sess_name[:3] in ('rem', 'sws')):
        boxname += '_rec2'
    elif (sess_name[:3] == 'box'):
        boxname = sess_name
    sspikes = load_spikes_dec(rat_name, mod_name,boxname, bSmooth = True, sig = sig)
    spikes = load_spikes_dec(rat_name, mod_name, boxname, bSmooth = False)

    xx,yy, speed = load_pos(rat_name, boxname, bSpeed = True)
    sspikes = sspikes[speed>2.5,:]
    spikes = spikes[speed>2.5,:]
    xx = xx[speed>2.5]
    yy = yy[speed>2.5]
    dspk =preprocessing.scale(sspikes)  
    times = np.where(np.sum(spikes>0, 1)>=1)[0]

    sspikes = sspikes[times,:]    
    dspk = dspk[times,:]
    xx = xx[times]
    yy = yy[times]

    a = np.zeros((len(xx), 2, num_neurons))
    for n in range(num_neurons):
        a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))

    c = np.zeros((len(xx), 2, num_neurons))
    for n in range(num_neurons):
        c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))

    mtot2 = np.sum(c,2)
    mtot1 = np.sum(a,2)
    coordsbox = np.arctan2(mtot2,mtot1)%(2*np.pi)

    return coordsbox