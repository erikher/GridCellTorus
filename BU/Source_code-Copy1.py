#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gudhi.clustering.tomato import Tomato

import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from scipy import stats
from datetime import datetime 
import time
import functools
from scipy import signal
from scipy import optimize
import sys
import numba
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from scipy import stats
from datetime import datetime 
import time
import functools
from scipy import signal
from scipy import optimize
import sys
import numba
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy import signal
import matplotlib
from scipy import stats
import sys
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
from utils import *


# In[2]:


import numpy as np
a = 1
np.savez('tst', a = a)


# ### Plot Barcodes

# In[ ]:


import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
import matplotlib.gridspec as grd


for (rat_name, mod_name, sess_name) in (('roger', 'mod1', 'box')):


    file_name = rat_name + '_' + mod_name + '_' + sess_name


    f = np.load('Results/Orig/' + file_name + '_H2.npz', allow_pickle = True)
    persistence = list(f['diagrams'])
    f.close()

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


    print('Number of rolls' + str(num_rolls))


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
    #fig,axesall = plt.subplots(len(dims), 1)
    fig = plt.figure()
    gs = grd.GridSpec(len(dims),1)#, height_ratios=[1,10], width_ratios=[6,1], wspace=0.1)

    indsall =  0
    labels = ["$H_0$", "$H_1$", "$H_2$"]
    for dit, dim in enumerate(dims):
        axes = plt.subplot(gs[dim])
        axes.axis('off')
        d = np.copy(persistence[dim])
        d[np.isinf(d[:,1]),1] = infinity
        dlife = (d[:,1] - d[:,0])
        dinds = np.argsort(dlife)[-30:]#[int(0.5*len(dlife)):]
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
    fig.tight_layout(pad=3.5, w_pad=0.1, h_pad=0.25)
    fig.savefig('' + file_name + 'persistence_barcode_inf' + str(int(round(infinity))), bbox_inches='tight')#, pad_inches=0.0)
    fig.savefig('' + file_name + 'persistence_barcode_inf' + str(int(round(infinity))) + '.pdf', bbox_inches='tight')#, pad_inches=0.0)


# ### Plot ratemaps

# In[ ]:



import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy import signal
import matplotlib
from scipy import stats
import sys
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
from utils import *

def get_spikes(rat_name, rec_name, mod_name, sess_name, bConj = True):
    if rec_name == 'rec1':
        tot_path = rat_name + '/'+ rec_name + '/' + rat_name + '_mod_final.mat'
    elif rec_name == 'rec2':
        tot_path = rat_name + '/'+ rec_name + '/' + rat_name + '_rec2_mod_final.mat'

    marvin = sio.loadmat(tot_path)
    m0 = np.zeros(len(marvin['v1'][:,0]), dtype=int)
    m00 = np.zeros(len(marvin['v1'][:,0]), dtype=int)
    for i,m in enumerate(marvin['v1'][:,0]):
        m0[i] = int(m[0][0])
        m00[i] = int(m[0][2:])

    m1 = np.zeros(len(marvin['v2'][:,0]), dtype=int)
    m11 = np.zeros(len(marvin['v2'][:,0]), dtype=int)
    for i,m in enumerate(marvin['v2'][:,0]):
        m1[i] = int(m[0][0])
        m11[i] = int(m[0][2:])

    m2 = np.zeros(len(marvin['v3'][:,0]), dtype=int)
    m22 = np.zeros(len(marvin['v3'][:,0]), dtype=int)
    for i,m in enumerate(marvin['v3'][:,0]):
        m2[i] = int(m[0][0])
        m22[i] = int(m[0][2:])
    if rat_name in ('roger', 'quentin'):
        m3 = np.zeros(len(marvin['v4'][:,0]), dtype=int)
        m33 = np.zeros(len(marvin['v4'][:,0]), dtype=int)
        for i,m in enumerate(marvin['v4'][:,0]):
            m3[i] = int(m[0][0])
            m33[i] = int(m[0][2:])
    if rat_name == 'roger':
        m4 = np.zeros(len(marvin['v5'][:,0]), dtype=int)
        m44 = np.zeros(len(marvin['v5'][:,0]), dtype=int)
        for i,m in enumerate(marvin['v5'][:,0]):
            m4[i] = int(m[0][0])
            m44[i] = int(m[0][2:])
        if sess_name in ('box', 'maze'):
            m5 = np.zeros(len(marvin['v6'][:,0]), dtype=int)
            m55 = np.zeros(len(marvin['v6'][:,0]), dtype=int)
            for i,m in enumerate(marvin['v6'][:,0]):
                m5[i] = int(m[0][0])
                m55[i] = int(m[0][2:])

    if rec_name == 'rec1':
        tot_path = rat_name + '/rec1/' + rat_name + '_all.mat'
    else:
        tot_path = rat_name + '/rec2/' + rat_name + '_all.mat'

    marvin = sio.loadmat(tot_path)
    mall = np.zeros(len(marvin['vv'][0,:]), dtype=int)
    mall1 = np.zeros(len(marvin['vv'][0,:]), dtype=int)
    for i,m in enumerate(marvin['vv'][0,:]):
        mall[i] = int(m[0][0])
        mall1[i] = int(m[0][2:])

    mind0 = np.zeros(len(m0), dtype = int)
    for i in range(len(m0)):
        mind0[i] = np.where((mall==m0[i]) & (mall1==m00[i]))[0]

    mind1 = np.zeros(len(m1), dtype = int)
    for i in range(len(m1)):
        mind1[i] = np.where((mall==m1[i]) & (mall1==m11[i]))[0]

    mind2 = np.zeros(len(m2), dtype = int)
    for i in range(len(m2)):
        mind2[i] = np.where((mall==m2[i]) & (mall1==m22[i]))[0]
    
    if rat_name in ('roger', 'quentin'):
        mind3 = np.zeros(len(m3), dtype = int)
        for i in range(len(m3)):
            mind3[i] = np.where((mall==m3[i]) & (mall1==m33[i]))[0]
    if rat_name == 'roger':
        mind4 = np.zeros(len(m4), dtype = int)
        for i in range(len(m4)):
            mind4[i] = np.where((mall==m4[i]) & (mall1==m44[i]))[0]
        if sess_name in ('box', 'maze'):
            mind5 = np.zeros(len(m5), dtype = int)
            for i in range(len(m5)):
                mind5[i] = np.where((mall==m5[i]) & (mall1==m55[i]))[0]

    tot_path = rat_name + '/' + rec_name + '/' + 'data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')
    
    numn = len(marvin["clusters"]["tc"][0])
    mvl_hd = np.zeros((numn))
    mvl_si = np.zeros((numn))
    hd_tun = np.zeros((numn, 60))
    for i in range(numn):
        if len(marvin[marvin["clusters"]["tc"][0][i]])==4:
            mvl_hd[i] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['mvl'][()]
            mvl_si[i] = marvin[marvin["clusters"]["tc"][0][i]]['pos']['si'][()]
            hd_tun[i,:] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['z'][:][0]
    
    if sess_name[:3] in ('rem', 'sws'):
        sess = 'sleep'
    else:
        sess = sess_name        

    min_time0 = times_all[rat_name + '_' + sess][0][0]
    max_time0 = times_all[rat_name + '_' + sess][0][1]
    
    spikes = {}
    if (rat_name == 'roger') & (rec_name == 'rec1'):
        if mod_name == 'mod1':
            mind = np.concatenate((mind2,mind3))
        elif mod_name == 'mod3':
            mind = mind4
        elif mod_name == 'mod4':
            mind = mind5
    else:
        if mod_name == 'mod1':
            mind = mind2
        elif mod_name == 'mod2':
            mind = mind3
        elif mod_name == 'mod3':
            mind = mind4

    for i,m in enumerate(mind):
        s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
        spikes[i] = np.array(s[(s>= min_time0) & (s< max_time0)])
    res = 100000
    dt = 1000
    min_time = min_time0*res
    max_time = max_time0*res
    sigma = 10000
    if sess_name[:3] == 'sws':
        sigma = 2500
    thresh = sigma*5
    num_thresh = int(thresh/dt)
    num2_thresh = int(2*num_thresh)
    sig2 = 1/(2*(sigma/res)**2)
    ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
    kerwhere = np.arange(-num_thresh,num_thresh)*dt

    ttall = []


    if sess_name[:3] in ('box', 'maz'):
        tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
        spikes_bin = np.zeros((1,len(spikes)))
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
    else:
        start = marvin['sleepTimes'][sess_name[:3]][()][0,:]*res
        start = start[start<max_time0*res]

        end = marvin['sleepTimes'][sess_name[:3]][()][1,:]*res
        end = end[end<=max_time0*res]

        spikes_bin = np.zeros((1,len(spikes)))
        for r in range(len(start)):
            min_time = start[r]
            max_time = end[r]

            tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
            ttall.extend(tt)
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
        
    if (rat_name =='shane') & (sess_name == 'maze'):
        spikes_bin = np.concatenate((spikes_bin, get_spikes(rat_name, rec_name, mod_name, 'maze2', bConj = True)),0)

    if (rat_name == 'roger') & (rec_name == 'rec1'):
        if sess_name =='box':
            valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                                          np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)
        elif sess_name == 'maze':
            valid_times = np.concatenate((np.arange(0, (18026-16925)*res/dt),
                                          np.arange((18183-16925)*res/dt, (20704-16925)*res/dt))).astype(int)
    else:
        valid_times = np.arange(len(spikes_bin[:,0]))
    print(valid_times.shape, sess_name)
    return spikes_bin[valid_times,:]

def load_pos(rat_name, rec_name, sess_name, bSpeed = False):    

    tot_path = rat_name + '/' + rec_name + '/' + 'data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')
    
    x = marvin['tracking']['x'][()][0,1:-1]
    y = marvin['tracking']['y'][()][0,1:-1]
    t = marvin['tracking']['t'][()][0,1:-1]
    z = marvin['tracking']['z'][()][0,1:-1]
    azimuth = marvin['tracking']['hd_azimuth'][()][0,1:-1]
    if sess_name[:3] in ('rem', 'sws'):
        sess = 'sleep'
    else:
        sess = sess_name        

    min_time0 = times_all[rat_name + '_' + sess][0][0]
    max_time0 = times_all[rat_name + '_' + sess][0][1]
    
    times = np.where((t>=min_time0) & (t<max_time0))
    x = x[times]
    y = y[times]
    t = t[times]
    z = z[times]
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

    rangesz =z[idx[:-1], np.newaxis] + np.multiply(ranges, (z[idx[1:]] - z[idx[:-1]])[:, np.newaxis])
    zz = rangesz[~np.isnan(ranges)] 

    rangesa =azimuth[idx[:-1], np.newaxis] + np.multiply(ranges, (azimuth[idx[1:]] - azimuth[idx[:-1]])[:, np.newaxis])
    aa = rangesa[~np.isnan(ranges)] 
    if (rat_name == 'roger') & (rec_name == 'rec1'):
        if sess_name =='box':
            valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                                          np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)
        elif sess_name == 'maze':
            valid_times = np.concatenate((np.arange(0, (18026-16925)*res/dt),
                                          np.arange((18183-16925)*res/dt, (20704-16925)*res/dt))).astype(int)
    else:
        valid_times = np.arange(len(xx))
        
    xx = xx[valid_times]
    yy = yy[valid_times]
    aa = aa[valid_times]
        
    if bSpeed:
        xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
        yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
        dx = (xxs[1:] - xxs[:-1])*100
        dy = (yys[1:] - yys[:-1])*100
        speed = np.sqrt(dx**2+ dy**2)/0.01
        speed = np.concatenate(([speed[0]],speed))
        if (rat_name =='shane') & (sess_name == 'maze'):
            xx1, yy1, aa1, speed1 = load_pos(rat_name, rec_name, 'maze2', True)
            xx = np.concatenate((xx,xx1))
            yy = np.concatenate((yy,yy1))
            aa = np.concatenate((aa,aa1))
            speed = np.concatenate((speed,speed1))
        return xx, yy, aa, speed
    if (rat_name =='shane') & (sess_name == 'maze'):
        xx1, yy1, aa1 = load_pos(rat_name, rec_name, 'maze2')
        xx = np.concatenate((xx,xx1))
        yy = np.concatenate((yy,yy1))
        aa = np.concatenate((aa,aa1))
    return xx,yy,aa

def get_acorr(rat_name, rec_name, mod_name, sess_name):
    spikes_box = {}
    tot_path = rat_name + '/' + rec_name + '/data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')  
    i = 0
    i0 = 0
    if sess_name[:3] not in ('sws','rem'):
        min_time0, max_time0 = times_all[rat_name + '_' + sess_name][0]
        t = marvin['tracking']['t'][()][0,1:-1]
        times = np.where((t>=min_time0) & (t<max_time0))
        t = t[times]

        x = marvin['tracking']['x'][()][0,1:-1]
        y = marvin['tracking']['y'][()][0,1:-1]
        x = x[times]
        y = y[times]
        xxs = gaussian_filter1d(x-np.min(x), sigma = 100)
        yys = gaussian_filter1d(y-np.min(y), sigma = 100)    
        dx = (xxs[1:] - xxs[:-1])*100
        dy = (yys[1:] - yys[:-1])*100
        speed = np.divide(np.sqrt(dx**2+ dy**2), t[1:] - t[:-1])
        speed = np.concatenate(([speed[0]],speed))
        ssp  = np.where(speed>2.5)[0]
        sw = np.where((ssp[1:] - ssp[:-1])>1)[0]
        if (rat_name == 'shane') & (mod_name in ('mod2','mod3')):
            return
        if (rat_name == 'quentin') & (mod_name in ('mod3')):
            return
        inds = get_inds(rat_name, mod_name, sess_name)
        i0 += len(inds)
        for m in inds:
            s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
            s = s[(s>= min_time0) & (s< max_time0)]
            min_time1 = t[ssp[0]-1]
            max_time1 = t[ssp[sw[0]]+1]
            sall = []
            for ss in range(len(sw)):
                stemp = s[(s>= min_time1) & (s< max_time1)]
                sall.extend(stemp)
                min_time1 = t[ssp[sw[ss-1]]-1]
                max_time1 = t[ssp[sw[ss]]+1]
            spikes_box[i] = np.array(sall)-min_time0
            i += 1
    else:
        min_time0, max_time0 = times_all[rat_name + '_sleep'][0]
        t = marvin['tracking']['t'][()][0,1:-1]
        times = np.where((t>=min_time0) & (t<max_time0))
        t = t[times]
        sw = times_all[rat_name + '_' + sess_name]
        if (rat_name == 'shane') & (mod_name in ('mod2','mod3')):
            return
        if (rat_name == 'quentin') & (mod_name in ('mod3')):
            return
        inds = get_inds(rat_name, mod_name, sess_name)
        i0 += len(inds)
        for m in inds:
            s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
            s = s[(s>= min_time0) & (s< max_time0)]
            sall = []
            for ss in range(len(sw)):
                min_time1 = sw[ss][0] #t[ssp[sw[ss-1]]-1]
                max_time1 = sw[ss][1]
                stemp = s[(s>= min_time1) & (s< max_time1)]
                sall.extend(stemp)
            spikes_box[i] = np.array(sall)-min_time0
            i += 1
    return get_isi_acorr(spikes_box, bLog = False, bOne = True, maxt = 0.2)


# In[ ]:


sess_name_0 = 'box'
sess_name_00 = 'box_rec2'
sess_name_1 = 'maze'
sess_name_2 = 'rem'
sess_name_3 = 'sws'
sess_name_33 = 'sws_c0'
f = np.load('autocorrelogram_classes_300621.npz', allow_pickle = True)
indsclasses = f['indsclasses'][:189]
f.close()

for (rat_name, rec_name, mod_name) in (('roger', 'rec1', 'mod1'),
                                       ('roger', 'rec1', 'mod3'),
                                       ('roger', 'rec1', 'mod4'),
                                       ('roger', 'rec2', 'mod1'),
                                       ('roger', 'rec2', 'mod1_inds'),
                                       #('roger', 'rec2', 'mod2'),
                                       ('roger', 'rec2', 'mod3'),
                                       ('quentin', 'rec1', 'mod1'),
                                       ('quentin', 'rec1', 'mod2'),
                                       ('shane', 'rec1', 'mod1'),                                        
                                      ):
    b_inds = False
    k_inds = [0,1]
    if mod_name == 'mod1_inds':
        mod_name = 'mod1'
        b_inds = True
        k_inds = [0,1,2]
        
    cs = {}
    xs = {}
    ys = {}
    aas = {}
    spks = {}
    if rat_name == 'quentin':
        ss = [sess_name_0, sess_name_1, sess_name_2, sess_name_3]
    elif rec_name == 'rec2':
        if mod_name == 'mod1':
            ss = [sess_name_00, sess_name_2, sess_name_33]
        else:
            ss = [sess_name_00, sess_name_2, sess_name_3]
    else:
        ss = [sess_name_0, sess_name_1]
    num_figs = 0

    file_name_all = rat_name +'_' + rec_name + '_' + mod_name
    for sess_name in ss:
        file_name_all += '_' + sess_name
        num_figs+= 1
        file_name = rat_name + '_' + mod_name + '_' + sess_name + '_' + ss[0]
        f = np.load('Results/Orig/' + file_name + '_alignment_dec.npz', allow_pickle = True)
        cs[sess_name] = f['csess']
        f.close()
        spks[sess_name] = get_spikes(rat_name, rec_name, mod_name, sess_name, bConj = True)
        if sess_name[:6] in ('sws_c0',):
            ts = np.where(np.sum(spks[sess_name][:,indsclasses==0],1)>0)[0]
        else:
            if (sess_name[:3] in ('rem','sws')) & (rat_name == 'roger'):
                spk = load_spikes(rat_name, mod_name, sess_name[:3] + '_rec2')
            else:
                spk = load_spikes(rat_name, mod_name, sess_name)
            ts = np.where(np.sum(spk>0, 1)>=1)[0]
        if sess_name[:3] in ('box', 'maz'):
            num_figs += 2
            xs[sess_name], ys[sess_name], aas[sess_name], speed0 = load_pos(rat_name, rec_name, sess_name, bSpeed = True)
            xs[sess_name] = xs[sess_name][speed0>2.5]
            ys[sess_name] = ys[sess_name][speed0>2.5]
            aas[sess_name] = aas[sess_name][speed0>2.5]
            xs[sess_name] = xs[sess_name][ts]
            ys[sess_name] = ys[sess_name][ts]
            aas[sess_name] = aas[sess_name][ts]
            spks[sess_name] = spks[sess_name][speed0>2.5]   
        spks[sess_name] = spks[sess_name][ts,:]
    acorr_curr = get_acorr(rat_name, rec_name, mod_name, ss[0])

    acorr_curr_box = np.zeros(acorr_curr.shape)
    for i in range(len(acorr_curr_box[:,0])):
        acorr_curr_box[i,:] = acorr_curr[i,:].astype(float).copy()/float(acorr_curr[i,0])
    acorr_curr_box[:,0] = 0
    r_box = transforms.Affine2D().skew_deg(15,15)
    numangsint = 51
    sig = 2.75

    tot_path = rat_name + '/' + rec_name + '/' + 'data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')
    numn = len(marvin["clusters"]["tc"][0])

    hd_tun = np.zeros((numn, 60))
    mvl_hd = np.zeros((numn))
    mvl_si = np.zeros((numn))

    for i in range(numn):
        if len(marvin[marvin["clusters"]["tc"][0][i]])==4:
            mvl_hd[i] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['mvl'][()]
            mvl_si[i] = marvin[marvin["clusters"]["tc"][0][i]]['pos']['si'][()]
            hd_tun[i,:] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['z'][()][0,:]

    inds = get_inds(rat_name, mod_name, ss[0])
    hd_tuning = hd_tun[inds,:]
    mvl_si = mvl_si[inds]
    mvl_hd = mvl_hd[inds]

    mmhd = marvin[marvin["clusters"]["tc"][0][i]]['hd']
    occ = mmhd['time'][()][0,:]
    occ0 = occ.copy()
    occ0 /= np.max(occ0)
    occ0 = np.concatenate((occ0, [occ0[0]]))

    phis = mmhd['phi'][()][0,:]
    phis = np.concatenate((phis, [phis[0]]))

    from matplotlib import gridspec
    get_ipython().run_line_magic('matplotlib', 'inline')
    from scipy.stats import binned_statistic
    numbins = 50
    bins = np.linspace(0,2*np.pi, numbins+1)
    sig_hd = np.sqrt(sig)
    num_neurons = len(spks[sess_name][0,:]) 
    mtots = {}
    num_figs +=1
    for s in ss:
        mtots[s] = np.zeros((num_neurons, numbins, numbins))
    plt.viridis()

    if len(ss) == 4:
        numfigs = 8
    else:
        numfigs = 6
    
    for k in k_inds:
        if b_inds:
            indstemp = np.where(indsclasses == k)[0]
        else:
            if k == 0:
                indstemp = np.where(mvl_hd<=0.3)[0]
            else:
                indstemp = np.where(mvl_hd>0.3)[0]
        numw = 4
#        numw = np.round(np.sqrt(len(indstemp)/numfigs)+1)
        numh = np.ceil(len(indstemp)/numw)
        gs = gridspec.GridSpec(int(numh), int(numw*numfigs+numw-1))
        fig = plt.figure(figsize=(np.ceil((numw*numfigs+numw-1)*1.05), np.ceil(numh*1.1)))
        indstemp = indstemp[np.argsort(mvl_si[indstemp])]
#        indstemp -= np.min(indstemp)
        for nn, n in enumerate(np.flip(indstemp[:])):
#            print(mvl_si[nn], mvl_si[n],nn,n)
            nnn = nn%numh
            mm = np.floor(nn/numh)
            if mm>0:
                mm =mm*numfigs + mm
            mm = int(mm)
            nnn = int(nnn)
            posnum = 0 
            if (rec_name == 'rec2'):
                xnum = 3
            else:
                xnum = 4

            for sess_name in ss:
                spk = spks[sess_name][:,n]
                cc1 = cs[sess_name]
                if sess_name[:3] in ('box', 'maz'):
                    xx = xs[sess_name]
                    yy = ys[sess_name]
                    binsx = np.linspace(np.min(xx)+ (np.max(xx) - np.min(xx))*0.0025,np.max(xx)- (np.max(xx) - np.min(xx))*0.0025, numbins+1)
                    binsy = np.linspace(np.min(yy)+ (np.max(yy) - np.min(yy))*0.0025,np.max(yy)- (np.max(yy) - np.min(yy))*0.0025, numbins+1)

                    posnum += 1 
                    ax = plt.subplot(gs[nnn,mm+posnum-1])
                    #Using the 1st row and 1st column for plotting heatmap

                    mtot1, x_edge, y_edge, circ = binned_statistic_2d(xx,yy,
                        spk, statistic='mean', bins=(binsx, binsy), range=None, expand_binnumbers=True)
                    #mtot2 = smooth_image(mtot1, sig)
                    nans = np.isnan(mtot1)
                    mtot1[nans] = np.mean(mtot1[~np.isnan(mtot1)])
                    mtot1 = gaussian_filter(mtot1, sigma = sig)
                    mtot1[nans] = -np.inf
                    ax.imshow(mtot1, vmin = 0, vmax = np.max(mtot1)*0.975, extent = [0,1,0,1])
                    ax.axis('off')
                    ax.set_aspect('equal', 'box')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1) 

                if sess_name[:3] in ('box'):
                    aa = aas[sess_name]            
                    mtot0 = hd_tuning[n,:].copy()
                    mtot0 /= np.max(mtot0)
                    posnum += 1 
                    ax = plt.subplot(gs[nnn,mm+posnum-1],projection=  'polar')  
                    ax.set_rticks([0.33, 0.66,1])
                    ax.plot(phis, np.concatenate((mtot0, [mtot0[0]])), lw = 3, c = 'k')
                    ax.plot(phis, occ0,lw = 3, c = [0.1,0.1,0.1], alpha = 0.4)  
                    ax.set_yticklabels('')
                    ax.set_xticklabels('')
                    ax.set_rlim([0,1.1])

                    posnum += 1 
                    ax = plt.subplot(gs[nnn,mm+posnum-1]) 
                    ax.bar(np.arange(200),acorr_curr_box[n,:], width = 1, color = 'k')
                    ax.set_xlim([0,200])
                    ax.set_xticks([0, 100, 200])
                    ax.set_yticks([0, 0.01, 0.02])
                    ax.set_yticklabels('')
                    ax.set_xticklabels('')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_aspect(1/ax.get_data_ratio())

                xnum += 1
                ax = plt.subplot(gs[nnn,mm+xnum-1])
                mtot_tmp, x_edge, y_edge,c2 = binned_statistic_2d(cc1[:,0],cc1[:,1], 
                                                      spk, statistic='mean', 
                                                      bins=bins, range=None, expand_binnumbers=True)
                nans = np.isnan(mtot_tmp)
                mtot_tmp[np.isnan(mtot_tmp)] = np.mean(mtot_tmp[~np.isnan(mtot_tmp)])
                mtot_tmp = smooth_tuning_map(np.rot90(mtot_tmp,1), numangsint, sig, bClose = False) 
                mtot_tmp[nans] = -np.inf

                mtots[sess_name][n,:,:] = mtot_tmp 
                ax.imshow(mtot_tmp, origin = 'lower', extent = [0,2*np.pi,0, 2*np.pi], vmin = 0, vmax = np.max(mtot_tmp) *0.975)
                ax.set_xticks([])
                ax.set_yticks([])

                for x in ax.images + ax.lines + ax.collections:
                    trans = x.get_transform()
                    x.set_transform(r_box+trans) 
                    if isinstance(x, PathCollection):
                        transoff = x.get_offset_transform()
                        x._transOffset = r_box+transoff     
                ax.set_xlim(0, 2*np.pi + 3*np.pi/5)
                ax.set_ylim(0, 2*np.pi + 3*np.pi/5)
                ax.set_aspect('equal', 'box') 
                ax.axis('off')
        if b_inds:
            mod_name = 'mod1_class'

        fig.tight_layout(pad=0, w_pad=0.5, h_pad=0.3)
        fig.savefig('' + rat_name + '_' + rec_name + '_' + mod_name + '_all_' + str(k) + '_4cols.png', 
                    bbox_inches='tight', pad_inches=0.2)
        fig.savefig('' + rat_name + '_' + rec_name + '_' + mod_name + '_all' + str(k) + '_4cols.pdf', 
                    bbox_inches='tight', pad_inches=0.2)
        plt.show()


# ### Information score and deviance

# Deviance scatter plot OF

# In[4]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin', 'shane']
mod_names = ['mod1', 'mod2', 'mod3', 'mod4']
sess_names = ['box']
#LAMs = [10]#,10,100,1000]
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

fig = plt.figure()
ax8 = fig.add_subplot(111)
minEV = np.inf
maxEV = -np.inf
Information_rate_Toroidal_OF = []
Information_rate_Spatial_OF = []

LOOtorscoresall = []
spacescoresall = []
Explained_deviance_Spatial_OF = []
Explained_deviance_Toroidal_OF = []
nils = 0 
for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if (((rat_name == 'shane') & (mod_name not in ('mod1'))) 
                | ((rat_name == 'shane') & (sess_name in ('sws', 'rem'))) 
                | ((rat_name == 'quentin') & (mod_name not in ('mod1', 'mod2'))) 
                | ((rat_name in ('quentin', 'shane')) & (sess_name in ('sws_c0','box_rec2'))) 
                | ((sess_name == 'sws_c0') & (mod_name not in ('mod1')))
                | ((rat_name == 'roger') & (sess_name in ('box', 'maze')) & (mod_name in ('mod2')))
                | ((rat_name == 'roger') & (sess_name in ('sws', 'rem', 'box_rec2')) & (mod_name in ('mod4')))
                | ((rat_name == 'roger') & (sess_name in ('sws')) & (mod_name in ('mod1')))):
                continue
            if sess_name[:3] not in ('box', 'maz'):
                continue

            file_name = rat_name + '_' + mod_name + '_' + sess_name
            spacescores = []
            LOOtorscores = []
            for i, LAM in enumerate(Lams):                
                f = np.load('GLM_info_res4/' + file_name + '_1_P_sd' + str(LAM) + '.npz')
                LOOtorscores.append(f['LOOtorscores'])
                spacescores.append(f['spacescores'])
                f.close()
            LOOtorscores = np.array(LOOtorscores)
            spacescores = np.array(spacescores)
            LOOtorscoresall.append(LOOtorscores)
            spacescoresall.append(spacescores)
            
            lamtor = np.argsort(np.sum(LOOtorscores,1))[-1]
            lamspace = np.argsort(np.sum(spacescores,1))[-1]

            spacescores = spacescores[lamspace,:]
            
            LOOtorscores = []
            for i in np.arange(10):                
                f = np.load('GLM_info_res4/' + file_name + '_1_P_seed' + str(i) + 'LAM' + str(Lams[lamtor]) +  '_deviance.npz')
                LOOtorscores.append(f['LOOtorscores'])
                if i == 0:
                    spacescores = f['spacescores']
                f.close()
                    
            LOOtorscores = np.array(LOOtorscores)
            LOOtorscores = np.mean(LOOtorscores[:,:],0)
            p  = wilcoxon(LOOtorscores-spacescores, alternative='greater')
            print(file_name)
            print(' n: ' + str(len(LOOtorscores)) )
            print(' p: ' + str(p) )
            print('')
            
            maxEV =  max(maxEV, max(np.max(LOOtorscores), np.max(spacescores)))
            minEV =  min(minEV, min(np.min(LOOtorscores), np.min(spacescores)))
            ax8.scatter(spacescores, LOOtorscores, s = 10, c = colors_envs[file_name])
            Explained_deviance_Toroidal_OF.extend([np.mean(LOOtorscores)])
            Explained_deviance_Toroidal_OF.extend([np.std(LOOtorscores)/np.sqrt(len(LOOtorscores))])
            Explained_deviance_Spatial_OF.extend([np.mean(spacescores)])
            Explained_deviance_Spatial_OF.extend([np.std(spacescores)/np.sqrt(len(spacescores))])


maxEV = 0.5
ax8.plot([minEV, maxEV], [minEV, maxEV], c='k', zorder = -5)
ax8.set_xlim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_ylim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_aspect('equal', 'box')
ax8.set_xticks([0.0,0.1,0.2,0.3,0.4,0.5])
ax8.set_xticklabels(('','','','','',''))
ax8.xaxis.set_tick_params(width=1, length =5)
ax8.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
ax8.set_yticklabels(('','','','','',''))
ax8.yaxis.set_tick_params(width=1, length =5)
print(minEV, maxEV)


# In[13]:


help(wilcoxon)


# In[62]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin', 'shane']
mod_names = ['mod1', 'mod2', 'mod3', 'mod4']
sess_names = ['box']
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)
minEV = np.inf
maxEV = -np.inf

LOOtorscoresall = []
spacescoresall = []
nils = 0 
data = []
data_names = []

for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if (((rat_name == 'shane') & (mod_name not in ('mod1'))) 
                | ((rat_name == 'shane') & (sess_name in ('sws', 'rem'))) 
                | ((rat_name == 'quentin') & (mod_name not in ('mod1', 'mod2'))) 
                | ((rat_name in ('quentin', 'shane')) & (sess_name in ('sws_c0','box_rec2'))) 
                | ((sess_name == 'sws_c0') & (mod_name not in ('mod1')))
                | ((rat_name == 'roger') & (sess_name in ('box', 'maze')) & (mod_name in ('mod2')))
                | ((rat_name == 'roger') & (sess_name in ('sws', 'rem', 'box_rec2')) & (mod_name in ('mod4')))
                | ((rat_name == 'roger') & (sess_name in ('sws')) & (mod_name in ('mod1')))):
                continue
            if sess_name[:3] not in ('box', 'maz'):
                continue

            file_name = rat_name + '_' + mod_name + '_' + sess_name
            spacescores = []
            LOOtorscores = []
            for i, LAM in enumerate(Lams):                
                f = np.load('GLM_info_res4/' + file_name + '_1_P_sd' + str(LAM) + '.npz')
                LOOtorscores.append(f['LOOtorscores'])
                spacescores.append(f['spacescores'])
                f.close()
#            print(np.shape(LOOtorscores))
            LOOtorscores = np.array(LOOtorscores)
            spacescores = np.array(spacescores)
            LOOtorscoresall.append(LOOtorscores)
            spacescoresall.append(spacescores)
            
            lamtor = np.argsort(np.sum(LOOtorscores,1))[-1]
            lamspace = np.argsort(np.sum(spacescores,1))[-1]

            spacescores = spacescores[lamspace,:]
            
            LOOtorscores = []
            for i in np.arange(10):                
                f = np.load('GLM_info_res4/' + file_name + '_1_P_seed' + str(i) + 'LAM' + str(Lams[lamtor]) +  '_deviance.npz')
                LOOtorscores.append(f['LOOtorscores'])
                if i == 0:
                    spacescores = f['spacescores']
                f.close()
                    
            LOOtorscores = np.array(LOOtorscores)
            LOOtorscores = np.mean(LOOtorscores[:,:],0)
            if sess_name == 'maze':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_WW'
            elif sess_name == 'box':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_OF'
            else:
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_' + np.str.upper(sess_name[:3])
            data.append(pd.Series(spacescores))
            data_names.extend([sheet_name + '_physical_space'])
            data.append(pd.Series(LOOtorscores))
            data_names.extend([sheet_name + '_torus'])            
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
df.to_excel("expl_dev_scatter_OF.xlsx", sheet_name='expl_dev_scatter_OF')  


# Information scatter plot OF

# In[9]:


import numpy as np 
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import functools
from scipy import signal
from scipy import optimize
import sys
import glob
import matplotlib
from utils import *

rat_names = ['roger', 'quentin', 'shane']
mod_names = ['mod1', 'mod2', 'mod3', 'mod4']
sess_names = ['rem', 'sws', 'sws_c0', 'box','maze']#, 'box_rec2']
#sess_names = ['box','maze']#, 'box_rec2']
sess_names = ['box']#, 'box_rec2']

fig = plt.figure()
ax8 = fig.add_subplot(111)
minEV = np.inf
maxEV = -np.inf

Information_rate_Toroidal_OF = []
Information_rate_Spatial_OF = []
for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if (((rat_name == 'shane') & (mod_name not in ('mod1'))) 
                | ((rat_name == 'shane') & (sess_name in ('sws', 'rem'))) 
                | ((rat_name == 'quentin') & (mod_name not in ('mod1', 'mod2'))) 
                | ((rat_name in ('quentin', 'shane')) & (sess_name in ('sws_c0','box_rec2'))) 
                | ((sess_name == 'sws_c0') & (mod_name not in ('mod1')))
                | ((rat_name == 'roger') & (sess_name in ('box', 'maze')) & (mod_name in ('mod2')))
                | ((rat_name == 'roger') & (sess_name in ('sws', 'rem', 'box_rec2')) & (mod_name in ('mod4')))
                | ((rat_name == 'roger') & (sess_name in ('sws')) & (mod_name in ('mod1')))):
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            spk = load_spikes(rat_name, mod_name, sess_name, bSpeed = True)
            spk = spk[np.sum(spk,1)>0, :]
#            print(file_name, np.sum(spk,0))
            f = np.load('GLM_info2/' + file_name + '_info.npz')
            LOOtorscores = np.divide(f['I_5_noise'], np.mean(spk,0))
            spacescores = np.divide(f['I_1'], np.mean(spk,0))
            f.close()

            maxEV =  max(maxEV, max(np.max(LOOtorscores), np.max(spacescores)))
            minEV =  min(minEV, min(np.min(LOOtorscores), np.min(spacescores)))
            ax8.scatter(spacescores, LOOtorscores, s = 10, c = colors_envs[file_name])
            p  = wilcoxon(LOOtorscores-spacescores, alternative='greater')
            print(file_name)
            print(' n: ' + str(len(LOOtorscores)) )
            print(' p: ' + str(p) )
            print('')

#            Information_rate_Toroidal_OF.extend([np.mean(LOOtorscores)])
#            Information_rate_Toroidal_OF.extend([np.std(LOOtorscores)/np.sqrt(len(LOOtorscores))])
#            Information_rate_Spatial_OF.extend([np.mean(spacescores)])
#            Information_rate_Spatial_OF.extend([np.std(spacescores)/np.sqrt(len(spacescores))])
ax8.plot([minEV, maxEV], [minEV, maxEV], c='k', zorder = -5)
ax8.set_xlim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_ylim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_aspect('equal', 'box')
ax8.xaxis.set_tick_params(width=1, length =5)
ax8.yaxis.set_tick_params(width=1, length =5)


# In[92]:


import numpy as np 
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import functools
from scipy import signal
from scipy import optimize
import sys
import glob
import matplotlib
from utils import *

rat_names = ['roger', 'quentin', 'shane']
mod_names = ['mod1', 'mod2', 'mod3', 'mod4']
sess_names = ['rem', 'sws', 'sws_c0', 'box','maze']#, 'box_rec2']
sess_names = ['box']#, 'box_rec2']
minEV = np.inf
maxEV = -np.inf
data = []
data_names = []

for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if (((rat_name == 'shane') & (mod_name not in ('mod1'))) 
                | ((rat_name == 'shane') & (sess_name in ('sws', 'rem'))) 
                | ((rat_name == 'quentin') & (mod_name not in ('mod1', 'mod2'))) 
                | ((rat_name in ('quentin', 'shane')) & (sess_name in ('sws_c0','box_rec2'))) 
                | ((sess_name == 'sws_c0') & (mod_name not in ('mod1')))
                | ((rat_name == 'roger') & (sess_name in ('box', 'maze')) & (mod_name in ('mod2')))
                | ((rat_name == 'roger') & (sess_name in ('sws', 'rem', 'box_rec2')) & (mod_name in ('mod4')))
                | ((rat_name == 'roger') & (sess_name in ('sws')) & (mod_name in ('mod1')))):
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            spk = load_spikes(rat_name, mod_name, sess_name, bSpeed = True)
            spk = spk[np.sum(spk,1)>0, :]
            f = np.load('GLM_info2/' + file_name + '_info.npz')
            LOOtorscores = np.divide(f['I_5_noise'], np.mean(spk,0))
            spacescores = np.divide(f['I_1'], np.mean(spk,0))
            f.close()

            if sess_name == 'maze':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_WW'
            elif sess_name == 'box':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_OF'
            else:
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_' + np.str.upper(sess_name[:3])
            data.append(pd.Series(spacescores))
            data_names.extend([sheet_name + '_physical_space'])
            data.append(pd.Series(LOOtorscores))
            data_names.extend([sheet_name + '_torus'])            
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "info_scatter_OF.xlsx"
df.to_excel(fname, sheet_name=fname)  


# Information scatter plot WW

# In[10]:


import numpy as np 
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import functools
from scipy import signal
from scipy import optimize
import sys
import glob
import matplotlib
from utils import *

rat_names = ['roger', 'quentin', 'shane']
mod_names = ['mod1', 'mod2', 'mod3', 'mod4']
sess_names = ['rem', 'sws', 'sws_c0', 'box','maze']#, 'box_rec2']
#sess_names = ['box','maze']#, 'box_rec2']
sess_names = ['maze']#, 'box_rec2']

fig = plt.figure()
ax8 = fig.add_subplot(111)
minEV = np.inf
maxEV = -np.inf

Information_rate_Toroidal_C = []
Information_rate_Spatial_C = []

for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if (((rat_name == 'shane') & (mod_name not in ('mod1'))) 
                | ((rat_name == 'shane') & (sess_name in ('sws', 'rem'))) 
                | ((rat_name == 'quentin') & (mod_name not in ('mod1', 'mod2'))) 
                | ((rat_name in ('quentin', 'shane')) & (sess_name in ('sws_c0','box_rec2'))) 
                | ((sess_name == 'sws_c0') & (mod_name not in ('mod1')))
                | ((rat_name == 'roger') & (sess_name in ('box', 'maze')) & (mod_name in ('mod2')))
                | ((rat_name == 'roger') & (sess_name in ('sws', 'rem', 'box_rec2')) & (mod_name in ('mod4')))
                | ((rat_name == 'roger') & (sess_name in ('sws')) & (mod_name in ('mod1')))):
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            spk = load_spikes(rat_name, mod_name, sess_name, bSpeed = True)
            spk = spk[np.sum(spk,1)>0, :]
            f = np.load('GLM_info2/' + file_name + '_info.npz')
            LOOtorscores = np.divide(f['I_5_noise'], np.mean(spk,0))
            spacescores = np.divide(f['I_1'], np.mean(spk,0))
            f.close()

            maxEV =  max(maxEV, max(np.max(LOOtorscores), np.max(spacescores)))
            minEV =  min(minEV, min(np.min(LOOtorscores), np.min(spacescores)))
            ax8.scatter(spacescores, LOOtorscores, s = 10, c = colors_envs[file_name])
            p  = wilcoxon(LOOtorscores-spacescores, alternative='greater')
            
            print(file_name)
            print(' n: ' + str(len(LOOtorscores)) )
            print(' p: ' + str(p) )
            print('')
            
#            Information_rate_Toroidal_C.extend([np.mean(LOOtorscores)])
#            Information_rate_Toroidal_C.extend([np.std(LOOtorscores)/np.sqrt(len(LOOtorscores))])
#            Information_rate_Spatial_C.extend([np.mean(spacescores)])
#            Information_rate_Spatial_C.extend([np.std(spacescores)/np.sqrt(len(spacescores))])

            

ax8.plot([minEV, maxEV], [minEV, maxEV], c='k', zorder = -5)
ax8.set_xlim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_ylim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_aspect('equal', 'box')
ax8.set_xticks([0,100,200,300])
ax8.set_xticklabels(('','','','','','',''))
ax8.xaxis.set_tick_params(width=1, length =5)
ax8.set_yticks([0,100,200,300])
ax8.set_yticklabels(('','','','','','',''))
ax8.yaxis.set_tick_params(width=1, length =5)


# In[363]:


import numpy as np 
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import functools
from scipy import signal
from scipy import optimize
import sys
import glob
import matplotlib
from utils import *

data = []
data_names = []

rat_names = ['roger', 'quentin', 'shane']
mod_names = ['mod1', 'mod2', 'mod3', 'mod4']
sess_names = ['rem', 'sws', 'sws_c0', 'box','maze']#, 'box_rec2']
sess_names = ['maze']#, 'box_rec2']
for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if (((rat_name == 'shane') & (mod_name not in ('mod1'))) 
                | ((rat_name == 'shane') & (sess_name in ('sws', 'rem'))) 
                | ((rat_name == 'quentin') & (mod_name not in ('mod1', 'mod2'))) 
                | ((rat_name in ('quentin', 'shane')) & (sess_name in ('sws_c0','box_rec2'))) 
                | ((sess_name == 'sws_c0') & (mod_name not in ('mod1')))
                | ((rat_name == 'roger') & (sess_name in ('box', 'maze')) & (mod_name in ('mod2')))
                | ((rat_name == 'roger') & (sess_name in ('sws', 'rem', 'box_rec2')) & (mod_name in ('mod4')))
                | ((rat_name == 'roger') & (sess_name in ('sws')) & (mod_name in ('mod1')))):
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            spk = load_spikes(rat_name, mod_name, sess_name, bSpeed = True)
            spk = spk[np.sum(spk,1)>0, :]
            f = np.load('GLM_info2/' + file_name + '_info.npz')
            LOOtorscores = np.divide(f['I_5_noise'], np.mean(spk,0))
            spacescores = np.divide(f['I_1'], np.mean(spk,0))
            f.close()

            if sess_name == 'maze':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_WW'
            elif sess_name == 'box':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_OF'
            else:
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_' + np.str.upper(sess_name[:3])
            data.append(pd.Series(spacescores))
            data_names.extend([sheet_name + '_physical_space'])
            data.append(pd.Series(LOOtorscores))
            data_names.extend([sheet_name + '_torus'])            
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "info_scatter_WW1.xlsx"
df.to_excel(fname, sheet_name=fname)  


# Deviance scatter plot WW

# In[12]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin', 'shane']
mod_names = ['mod1', 'mod2', 'mod3', 'mod4']
sess_names = ['maze']
#LAMs = [10]#,10,100,1000]
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

fig = plt.figure()
ax8 = fig.add_subplot(111)
minEV = np.inf
maxEV = -np.inf
Explained_deviance_Toroidal_C = []
Explained_deviance_Spatial_C = []
LOOtorscoresall = []
spacescoresall = []
nils = 0 
for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if (((rat_name == 'shane') & (mod_name not in ('mod1'))) 
                | ((rat_name == 'shane') & (sess_name in ('sws', 'rem'))) 
                | ((rat_name == 'quentin') & (mod_name not in ('mod1', 'mod2'))) 
                | ((rat_name in ('quentin', 'shane')) & (sess_name in ('sws_c0','box_rec2'))) 
                | ((sess_name == 'sws_c0') & (mod_name not in ('mod1')))
                | ((rat_name == 'roger') & (sess_name in ('box', 'maze')) & (mod_name in ('mod2')))
                | ((rat_name == 'roger') & (sess_name in ('sws', 'rem', 'box_rec2')) & (mod_name in ('mod4')))
                | ((rat_name == 'roger') & (sess_name in ('sws')) & (mod_name in ('mod1')))):
                continue
            if sess_name[:3] not in ('box', 'maz'):
                continue

            file_name = rat_name + '_' + mod_name + '_' + sess_name
            spacescores = []
            LOOtorscores = []
            for i, LAM in enumerate(Lams):                
                f = np.load('GLM_info_res4/' + file_name + '_1_P_sd' + str(LAM) + '.npz')
                LOOtorscores.append(f['LOOtorscores'])
                spacescores.append(f['spacescores'])
                f.close()
#            print(np.shape(LOOtorscores))
            LOOtorscores = np.array(LOOtorscores)
            spacescores = np.array(spacescores)
            LOOtorscoresall.append(LOOtorscores)
            spacescoresall.append(spacescores)
            
            lamtor = np.argsort(np.sum(LOOtorscores,1))[-1]
            lamspace = np.argsort(np.sum(spacescores,1))[-1]

            spacescores = spacescores[lamspace,:]
            
            LOOtorscores = []
            for i in np.arange(10):                
                f = np.load('GLM_info_res4/' + file_name + '_1_P_seed' + str(i) + 'LAM' + str(Lams[lamtor]) +  '_deviance.npz')
                LOOtorscores.append(f['LOOtorscores'])
                if i == 0:
                    spacescores = f['spacescores']
                f.close()

            LOOtorscores = np.array(LOOtorscores)
            LOOtorscores = np.mean(LOOtorscores[:,:],0)
            p  = wilcoxon(LOOtorscores-spacescores, alternative='greater')
            print(file_name)
            print(' n: ' + str(len(LOOtorscores)) )
            print(' p: ' + str(p) )
            print('')
            
            maxEV =  max(maxEV, max(np.max(LOOtorscores), np.max(spacescores)))
            minEV =  min(minEV, min(np.min(LOOtorscores), np.min(spacescores)))
            ax8.scatter(spacescores, LOOtorscores, s = 10, c = colors_envs[file_name])
            
            
maxEV = 0.5
ax8.plot([minEV, maxEV], [minEV, maxEV], c='k', zorder = -5)
ax8.set_xlim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_ylim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_aspect('equal', 'box')
ax8.set_xticks([0.0,0.1,0.2,0.3,0.4,0.5])
ax8.set_xticklabels(('','','','','',''))
ax8.xaxis.set_tick_params(width=1, length =5)
ax8.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
ax8.set_yticklabels(('','','','','',''))
ax8.yaxis.set_tick_params(width=1, length =5)


# In[61]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin', 'shane']
mod_names = ['mod1', 'mod2', 'mod3', 'mod4']
sess_names = ['maze']
#LAMs = [10]#,10,100,1000]
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

minEV = np.inf
maxEV = -np.inf

LOOtorscoresall = []
spacescoresall = []
nils = 0 

data = []
data_names = []
for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if (((rat_name == 'shane') & (mod_name not in ('mod1'))) 
                | ((rat_name == 'shane') & (sess_name in ('sws', 'rem'))) 
                | ((rat_name == 'quentin') & (mod_name not in ('mod1', 'mod2'))) 
                | ((rat_name in ('quentin', 'shane')) & (sess_name in ('sws_c0','box_rec2'))) 
                | ((sess_name == 'sws_c0') & (mod_name not in ('mod1')))
                | ((rat_name == 'roger') & (sess_name in ('box', 'maze')) & (mod_name in ('mod2')))
                | ((rat_name == 'roger') & (sess_name in ('sws', 'rem', 'box_rec2')) & (mod_name in ('mod4')))
                | ((rat_name == 'roger') & (sess_name in ('sws')) & (mod_name in ('mod1')))):
                continue
            if sess_name[:3] not in ('box', 'maz'):
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            spacescores = []
            LOOtorscores = []
            for i, LAM in enumerate(Lams):                
                f = np.load('GLM_info_res4/' + file_name + '_1_P_sd' + str(LAM) + '.npz')
                LOOtorscores.append(f['LOOtorscores'])
                spacescores.append(f['spacescores'])
                f.close()
            LOOtorscores = np.array(LOOtorscores)
            spacescores = np.array(spacescores)
            LOOtorscoresall.append(LOOtorscores)
            spacescoresall.append(spacescores)
            
            lamtor = np.argsort(np.sum(LOOtorscores,1))[-1]
            lamspace = np.argsort(np.sum(spacescores,1))[-1]

            spacescores = spacescores[lamspace,:]
            
            LOOtorscores = []
            for i in np.arange(10):                
                f = np.load('GLM_info_res4/' + file_name + '_1_P_seed' + str(i) + 'LAM' + str(Lams[lamtor]) +  '_deviance.npz')
                LOOtorscores.append(f['LOOtorscores'])
                if i == 0:
                    spacescores = f['spacescores']
                f.close()
            LOOtorscores = np.array(LOOtorscores)
            LOOtorscores = np.mean(LOOtorscores[:,:],0)
            if sess_name == 'maze':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_WW'
            elif sess_name == 'box':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_OF'
            else:
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_' + np.str.upper(sess_name[:3])
            data.append(pd.Series(spacescores))
            data_names.extend([sheet_name + '_physical_space'])
            data.append(pd.Series(LOOtorscores))
            data_names.extend([sheet_name + '_torus'])            

df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
df.to_excel("expl_dev_scatter_WW.xlsx", sheet_name='expl_dev_scatter_WW')  


# #### Results of Kruskal-wallis test on bursty, theta and non-bursty classes

# Medians: 0.187774019 0.080010298 0.050510688,  Sizes: 523 229 95
# 
# KruskalResult(statistic=204.74202167772637, pvalue=3.474040569782457e-45)
# 
# Dunn test with bonferroni correction
# 
#               1             2             3
# 
# 1  1.000000e+00  1.447610e-27  2.660121e-29
# 
# 2  1.447610e-27  1.000000e+00  3.719186e-03
# 
# 3  2.660121e-29  3.719186e-03  1.000000e+00
# 
# 
# File: deviance_contrem_030821.txt
# 
# Medians: 0.087694294 0.0495752 0.018610073, Sizes: 495 169 43
# 
# KruskalResult(statistic=96.32765353474042, pvalue=1.2098062048551787e-21)
# 
# Dunn test with bonferroni correction
# 
#               1             2             3
# 
# 1  1.000000e+00  3.963385e-13  3.189210e-13
# 
# 2  3.963385e-13  1.000000e+00  6.716849e-03
# 
# 3  3.189210e-13  6.716849e-03  1.000000e+00
# 
# 
# File: deviance_contsws_030821.txt
# 
# Medians: 0.100028357 0.043692024 0.018167421, Sizes: 495 169 43
# 
# KruskalResult(statistic=171.78496570938887, pvalue=4.981597780713629e-38)
# 
# Dunn test with bonferroni correction
# 
#               1             2             3
# 
# 1  1.000000e+00  6.661049e-25  2.255278e-20
# 
# 2  6.661049e-25  1.000000e+00  3.027236e-03
# 
# 3  2.255278e-20  3.027236e-03  1.000000e+00

# histogram deviance REM

# In[64]:


colors_envs['roger_mod1_box_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_box_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_box_rec2'] = [[0.7,0.,0.7]]

colors_envs['roger_mod1_rem_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_rem_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_rem_rec2'] = [[0.7,0.,0.7]]
colors_envs['quentin_mod1_rem'] = [[0.3,0.7,0.3]]
colors_envs['quentin_mod2_rem'] = [[0.,0.7,0.7]]
#colors_envs['shane_mod1_box'] = [[0.3,0.3,0.7]]
colors_envs['roger_mod1_sws_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_sws_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_sws_rec2'] = [[0.7,0.,0.7]]
colors_envs['quentin_mod1_sws'] = [[0.3,0.7,0.3]]
colors_envs['quentin_mod2_sws'] = [[0.,0.7,0.7]]
#colors_envs['shane_mod1_box'] = [[0.3,0.3,0.7]]


# In[65]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin']
#rat_names = ['quentin']
mod_names = ['mod1', 'mod2', 'mod3']
sess_names = ['rem']
#LAMs = [10]#,10,100,1000]
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

fig = plt.figure()
ax = fig.add_subplot(111)
minEV = np.inf
maxEV = -np.inf
colors_envs['roger_mod1_box_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_box_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_box_rec2'] = [[0.7,0.,0.7]]

#numbins = 15
lins = np.arange(-0.02,0.5,0.035)
numbins = len(lins)
width = (lins[1]-lins[0])/2
b1_max = -np.inf

LOOtorscoresall = []
spacescoresall = []
nils = 0 
for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if rat_name == 'roger':
                sess_name += '_rec2'
            if rat_name+mod_name == 'quentinmod3':
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            if file_name == 'roger_mod1_sws_rec2':
                continue
            spacescores = []
            LOOtorscores = []
            for i, LAM in enumerate(Lams):                
                f = np.load('GLM8/' + file_name + '_1_P_sd_LAM' + str(LAM) + '_deviance.npz')
                LOOtorscores.append(f['LOOtorscores'])
                f.close()
            LOOtorscores = np.array(LOOtorscores)
            
            lamtor = np.argsort(np.sum(LOOtorscores,1))[-1]
            LOOtorscores = LOOtorscores[lamtor,:]
            
            nilstmp = (LOOtorscores<0)
            
            maxEV =  max(maxEV, np.max(LOOtorscores))
            minEV =  min(minEV, np.min(LOOtorscores))
            i1 = np.digitize(LOOtorscores, lins)-1
            b1 = np.bincount(i1,minlength = numbins)/len(LOOtorscores,)
            b1_max = max(b1_max,np.max(b1))
            ax.plot(lins + width, b1,  alpha = 0.7, c = colors_envs[file_name][0], lw = 3)
b1_max = 0.4
ax.set_ylim([0,b1_max])
ax.set_xlim([np.min(lins),np.max(lins)])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xticks([0.0,0.1,0.2,0.3,0.4,0.5])
ax.set_xticklabels(('','','','','',''))
ax.xaxis.set_tick_params(width=1, length =5)
ax.set_yticks([0,0.1,0.2,0.3,0.4])
ax.set_yticklabels(('','','',''))
ax.yaxis.set_tick_params(width=1, length =5)

            


# In[67]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin']
mod_names = ['mod1', 'mod2', 'mod3']
sess_names = ['rem']
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

minEV = np.inf
maxEV = -np.inf
lins = np.arange(-0.02,0.5,0.035)
numbins = len(lins)
width = (lins[1]-lins[0])/2
b1_max = -np.inf

LOOtorscoresall = []
spacescoresall = []
nils = 0 
data = []
data_names = []

for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if rat_name == 'roger':
                sess_name += '_rec2'
            if rat_name+mod_name == 'quentinmod3':
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            if file_name == 'roger_mod1_sws_rec2':
                continue
            spacescores = []
            LOOtorscores = []
            for i, LAM in enumerate(Lams):                
                f = np.load('GLM8/' + file_name + '_1_P_sd_LAM' + str(LAM) + '_deviance.npz')
                LOOtorscores.append(f['LOOtorscores'])
                f.close()
            LOOtorscores = np.array(LOOtorscores)
            
            lamtor = np.argsort(np.sum(LOOtorscores,1))[-1]
            LOOtorscores = LOOtorscores[lamtor,:]
            
            nilstmp = (LOOtorscores<0)
            
            maxEV =  max(maxEV, np.max(LOOtorscores))
            minEV =  min(minEV, np.min(LOOtorscores))
            i1 = np.digitize(LOOtorscores, lins)-1
            b1 = np.bincount(i1,minlength = numbins)/len(LOOtorscores,)
            b1_max = max(b1_max,np.max(b1))
            ax.plot(lins + width, b1,  alpha = 0.7, c = colors_envs[file_name][0], lw = 3)

            if sess_name == 'maze':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_WW'
            elif sess_name == 'box':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_OF'
            else:
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_' + np.str.upper(sess_name[:3])
            data.append(pd.Series(b1))
            data_names.extend([sheet_name + '_deviance'])
            data.append(pd.Series(lins + width))
            data_names.extend([sheet_name + '_bins'])            
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
df.to_excel("expl_dev_hist_REM.xlsx", sheet_name='expl_dev_hist_REM')  


# histogram deviance SWS

# In[69]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin']
#rat_names = ['quentin']
mod_names = ['mod1', 'mod2', 'mod3']
sess_names = ['sws']
#LAMs = [10]#,10,100,1000]
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

fig = plt.figure()
ax = fig.add_subplot(111)
minEV = np.inf
maxEV = -np.inf
colors_envs['roger_mod1_box_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_box_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_box_rec2'] = [[0.7,0.,0.7]]

#numbins = 15
lins = np.arange(-0.02,0.5,0.035)
numbins = len(lins)
width = (lins[1]-lins[0])/2
b1_max = -np.inf

LOOtorscoresall = []
spacescoresall = []
nils = 0 
for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if rat_name == 'roger':
                sess_name += '_rec2'
            if rat_name+mod_name == 'quentinmod3':
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            if file_name == 'roger_mod1_sws_rec2':
                continue
            spacescores = []
            LOOtorscores = []
            for i, LAM in enumerate(Lams):                
                f = np.load('GLM8/' + file_name + '_1_P_sd_LAM' + str(LAM) + '_deviance.npz')
                LOOtorscores.append(f['LOOtorscores'])
                print(np.sum(f['LOOtorscores']<0),np.sum(f['LOOtorscores']),np.median(f['LOOtorscores']))
#                print(np.min(f['LOOtorscores']),np.max(f['LOOtorscores']))
                f.close()
            LOOtorscores = np.array(LOOtorscores)
            
            lamtor = np.argsort(np.sum(LOOtorscores,1))[-1]
            LOOtorscores = LOOtorscores[lamtor,:]
            
            nilstmp = (LOOtorscores<0)
            print(file_name, np.sum(nilstmp), lamtor)
            #nils += np.sum(nilstmp) 
            #LOOtorscores = LOOtorscores[~nilstmp]
            
            maxEV =  max(maxEV, np.max(LOOtorscores))
            minEV =  min(minEV, np.min(LOOtorscores))
            #ax.scatter(spacescores, LOOtorscores, s = 10, c = colors_envs[file_name])
            i1 = np.digitize(LOOtorscores, lins)-1
            b1 = np.bincount(i1,minlength = numbins)/len(LOOtorscores,)
            b1_max = max(b1_max,np.max(b1))
            ax.plot(lins + width, b1,  alpha = 0.7, c = colors_envs[file_name][0], lw = 3)
print(b1_max)
b1_max = 0.4
#ax.plot([0,0], [0, b1_max], c = 'k', ls = '-' )

ax.set_ylim([0,b1_max])
ax.set_xlim([np.min(lins),np.max(lins)])
#ax.set_xticks([],[])
#ax.set_yticks([],[]) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.set_aspect('equal', 'box')

ax.set_xticks([0.0,0.1,0.2,0.3,0.4,0.5])
ax.set_xticklabels(('','','','','',''))
ax.xaxis.set_tick_params(width=1, length =5)
ax.set_yticks([0,0.1,0.2,0.3,0.4])
ax.set_yticklabels(('','','',''))
ax.yaxis.set_tick_params(width=1, length =5)

#plt.savefig('hist_EV_sws', bbox_inches='tight', pad_inches=0.03)
#plt.savefig('hist_EV_sws.pdf', bbox_inches='tight', pad_inches=0.03)
#print(b1_max)
            


# In[70]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin']
#rat_names = ['quentin']
mod_names = ['mod1', 'mod2', 'mod3']
sess_names = ['sws']
#LAMs = [10]#,10,100,1000]
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

minEV = np.inf
maxEV = -np.inf

lins = np.arange(-0.02,0.5,0.035)
numbins = len(lins)
width = (lins[1]-lins[0])/2
b1_max = -np.inf

LOOtorscoresall = []
spacescoresall = []
nils = 0 
data = []
data_names = []

for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if rat_name == 'roger':
                sess_name += '_rec2'
            if rat_name+mod_name == 'quentinmod3':
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            if file_name == 'roger_mod1_sws_rec2':
                continue
            spacescores = []
            LOOtorscores = []
            for i, LAM in enumerate(Lams):                
                f = np.load('GLM8/' + file_name + '_1_P_sd_LAM' + str(LAM) + '_deviance.npz')
                LOOtorscores.append(f['LOOtorscores'])
                f.close()
            LOOtorscores = np.array(LOOtorscores)
            
            lamtor = np.argsort(np.sum(LOOtorscores,1))[-1]
            LOOtorscores = LOOtorscores[lamtor,:]
            
            nilstmp = (LOOtorscores<0)
            maxEV =  max(maxEV, np.max(LOOtorscores))
            minEV =  min(minEV, np.min(LOOtorscores))
            i1 = np.digitize(LOOtorscores, lins)-1
            b1 = np.bincount(i1,minlength = numbins)/len(LOOtorscores,)
            b1_max = max(b1_max,np.max(b1))
            if sess_name == 'maze':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_WW'
            elif sess_name == 'box':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_OF'
            else:
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_' + np.str.upper(sess_name[:3])
            data.append(pd.Series(b1))
            data_names.extend([sheet_name + '_deviance'])
            data.append(pd.Series(lins + width))
            data_names.extend([sheet_name + '_bins'])            
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
df.to_excel("expl_dev_hist_SWS.xlsx", sheet_name='expl_dev_hist_SWS')  


# histogram information content SWS

# In[68]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin']
#rat_names = ['quentin']
mod_names = ['mod1', 'mod2', 'mod3']
sess_names = ['sws']
#LAMs = [10]#,10,100,1000]
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

fig = plt.figure()
ax = fig.add_subplot(111)
minEV = np.inf
maxEV = -np.inf
colors_envs['roger_mod1_box_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_box_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_box_rec2'] = [[0.7,0.,0.7]]

lins = np.linspace(0,2.5,10)#np.arange(0.0,4,0.25)
print(lins)
numbins = len(lins)
width = (lins[1]-lins[0])/2
b1_max = -np.inf

LOOtorscoresall = []
spacescoresall = []
nils = 0 
j = 0
I_mean =  np.zeros(4)
for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if rat_name == 'roger':
                sess_name += '_rec2'
            if rat_name+mod_name == 'quentinmod3':
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            if file_name == 'roger_mod1_sws_rec2':
                continue
            spacescores = []
            LOOtorscores = []
            
            spk = load_spikes(rat_name, mod_name, sess_name, bSpeed = True)
            spk = spk[np.sum(spk,1)>0, :]
            

            file_name = rat_name + '_' + mod_name + '_' + sess_name
            f = np.load('GLM_info_res7/' + file_name + '_info_sleep.npz')
            LOOtorscores = np.divide(f['I_5'], np.mean(spk,0))
            I_torus_sws_shuf = np.divide(f['I_5_noise'], np.mean(spk,0))
            f.close()

            maxEV =  max(maxEV, np.max(LOOtorscores))
            minEV =  min(minEV, np.min(LOOtorscores))
            #ax.scatter(spacescores, LOOtorscores, s = 10, c = colors_envs[file_name])
            i1 = np.digitize(LOOtorscores, lins)-1
            b1 = np.bincount(i1,minlength = numbins)/len(LOOtorscores,)
            b1_max = max(b1_max,np.max(b1))
            ax.plot(lins + width, b1,  alpha = 0.7, c = colors_envs[file_name][0], lw = 3)
            I_mean[j] = np.mean(I_torus_sws_shuf)
            j += 1




b1_max = 0.4

ax.plot([np.mean(I_mean),np.mean(I_mean)], [0, b1_max], c = 'k', ls = '--', lw = 1 )
#ax.plot([0,0], [0, b1_max], c = 'k', ls = '-' )

ax.set_ylim([0,b1_max])
ax.set_xlim([np.min(lins),np.max(lins)])
#ax.set_xticks([],[])
#ax.set_yticks([],[]) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.set_aspect('equal', 'box')

ax.set_xticks([0,0.5,1,1.5,2,2.5])
ax.set_xticklabels(('','','','','',''))
ax.xaxis.set_tick_params(width=1, length =5)
ax.set_yticks([0,0.1,0.2,0.3])
ax.set_yticklabels(('','','',''))
ax.yaxis.set_tick_params(width=1, length =5)            
print(b1_max)


# In[76]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin']
#rat_names = ['quentin']
mod_names = ['mod1', 'mod2', 'mod3']
sess_names = ['sws']
#LAMs = [10]#,10,100,1000]
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)
minEV = np.inf
maxEV = -np.inf

lins = np.linspace(0,2.5,10)#np.arange(0.0,4,0.25)
numbins = len(lins)
width = (lins[1]-lins[0])/2
b1_max = -np.inf

LOOtorscoresall = []
spacescoresall = []
nils = 0 
j = 0
I_mean =  np.zeros(4)
data = []
data_names = []

for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if rat_name == 'roger':
                sess_name += '_rec2'
            if rat_name+mod_name == 'quentinmod3':
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            if file_name == 'roger_mod1_sws_rec2':
                continue
            spacescores = []
            LOOtorscores = []
            
            spk = load_spikes(rat_name, mod_name, sess_name, bSpeed = True)
            spk = spk[np.sum(spk,1)>0, :]
            
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            f = np.load('GLM_info_res7/' + file_name + '_info_sleep.npz')
            LOOtorscores = np.divide(f['I_5'], np.mean(spk,0))
            I_torus_sws_shuf = np.divide(f['I_5_noise'], np.mean(spk,0))
            f.close()

            maxEV =  max(maxEV, np.max(LOOtorscores))
            minEV =  min(minEV, np.min(LOOtorscores))
            i1 = np.digitize(LOOtorscores, lins)-1
            b1 = np.bincount(i1,minlength = numbins)/len(LOOtorscores,)
            b1_max = max(b1_max,np.max(b1))
            ax.plot(lins + width, b1,  alpha = 0.7, c = colors_envs[file_name][0], lw = 3)
            I_mean[j] = np.mean(I_torus_sws_shuf)
            j += 1
            if sess_name == 'maze':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_WW'
            elif sess_name == 'box':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_OF'
            else:
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_' + np.str.upper(sess_name[:3])
            data.append(pd.Series(b1))
            data_names.extend([sheet_name + '_info'])
            data.append(pd.Series(lins + width))
            data_names.extend([sheet_name + '_bins'])         
            
data.append(pd.Series(np.mean(I_mean)))
data_names.extend(['shuffled_value'])

df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
df.to_excel("info_hist_SWS.xlsx", sheet_name='info_hist_SWS')  


# histogram information contenct REM

# In[84]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin']
#rat_names = ['quentin']
mod_names = ['mod1', 'mod2', 'mod3']
sess_names = ['rem']
#LAMs = [10]#,10,100,1000]
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

fig = plt.figure()
ax = fig.add_subplot(111)
minEV = np.inf
maxEV = -np.inf
colors_envs['roger_mod1_box_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_box_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_box_rec2'] = [[0.7,0.,0.7]]

lins = np.linspace(0,2.5,10)#np.arange(0.0,4,0.25)
numbins = len(lins)
width = (lins[1]-lins[0])/2
b1_max = -np.inf

LOOtorscoresall = []
spacescoresall = []
nils = 0 
j = 0
I_mean =  np.zeros(5)
for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if rat_name == 'roger':
                sess_name += '_rec2'
            if rat_name+mod_name == 'quentinmod3':
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            if file_name == 'roger_mod1_sws_rec2':
                continue
            spacescores = []
            LOOtorscores = []
            
            #print(file_name, np.sum(nilstmp), lamtor)
            #nils += np.sum(nilstmp) 
            #LOOtorscores = LOOtorscores[~nilstmp]
            spk = load_spikes(rat_name, mod_name, sess_name, bSpeed = True)
            spk = spk[np.sum(spk,1)>0, :]
            


            file_name = rat_name + '_' + mod_name + '_' + sess_name
            f = np.load('GLM_info_res7/' + file_name + '_info_sleep.npz')
            LOOtorscores = np.divide(f['I_5'], np.mean(spk,0))
            I_torus_sws_shuf = np.divide(f['I_5_noise'], np.mean(spk,0))
            f.close()
            maxEV =  max(maxEV, np.max(LOOtorscores))
            minEV =  min(minEV, np.min(LOOtorscores))
            #ax.scatter(spacescores, LOOtorscores, s = 10, c = colors_envs[file_name])
            i1 = np.digitize(LOOtorscores, lins)-1
            b1 = np.bincount(i1,minlength = numbins)/len(LOOtorscores,)
            b1_max = max(b1_max,np.max(b1))
            ax.plot(lins + width, b1,  alpha = 0.7, c = colors_envs[file_name][0], lw = 3)
            I_mean[j] = np.mean(I_torus_sws_shuf)
            j += 1




b1_max = 0.4

ax.plot([np.mean(I_mean),np.mean(I_mean)], [0, b1_max], c = 'k', ls = '--', lw = 1 )
#ax.plot([0,0], [0, b1_max], c = 'k', ls = '-' )

ax.set_ylim([0,b1_max])
ax.set_xlim([np.min(lins),np.max(lins)])
#ax.set_xticks([],[])
#ax.set_yticks([],[]) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.set_aspect('equal', 'box')

ax.set_xticks([0,0.5,1,1.5,2,2.5])
ax.set_xticklabels(('','','','','',''))
ax.xaxis.set_tick_params(width=1, length =5)
ax.set_yticks([0,0.1,0.2,0.3])
ax.set_yticklabels(('','','',''))
ax.yaxis.set_tick_params(width=1, length =5)


# In[78]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin']
#rat_names = ['quentin']
mod_names = ['mod1', 'mod2', 'mod3']
sess_names = ['rem']
#LAMs = [10]#,10,100,1000]
Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

fig = plt.figure()
ax = fig.add_subplot(111)
minEV = np.inf
maxEV = -np.inf
colors_envs['roger_mod1_box_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_box_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_box_rec2'] = [[0.7,0.,0.7]]

lins = np.linspace(0,2.5,10)#np.arange(0.0,4,0.25)
numbins = len(lins)
width = (lins[1]-lins[0])/2
b1_max = -np.inf

LOOtorscoresall = []
spacescoresall = []
nils = 0 
j = 0
I_mean =  np.zeros(5)

data = []
data_names = []

for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in sess_names:
            if rat_name == 'roger':
                sess_name += '_rec2'
            if rat_name+mod_name == 'quentinmod3':
                continue
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            if file_name == 'roger_mod1_sws_rec2':
                continue
            spacescores = []
            LOOtorscores = []
            
            spk = load_spikes(rat_name, mod_name, sess_name, bSpeed = True)
            spk = spk[np.sum(spk,1)>0, :]

            file_name = rat_name + '_' + mod_name + '_' + sess_name
            f = np.load('GLM_info_res7/' + file_name + '_info_sleep.npz')
            LOOtorscores = np.divide(f['I_5'], np.mean(spk,0))
            I_torus_sws_shuf = np.divide(f['I_5_noise'], np.mean(spk,0))
            f.close()

            maxEV =  max(maxEV, np.max(LOOtorscores))
            minEV =  min(minEV, np.min(LOOtorscores))
            i1 = np.digitize(LOOtorscores, lins)-1
            b1 = np.bincount(i1,minlength = numbins)/len(LOOtorscores,)
            b1_max = max(b1_max,np.max(b1))
            I_mean[j] = np.mean(I_torus_sws_shuf)
            j += 1
            if sess_name == 'maze':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_WW'
            elif sess_name == 'box':
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_OF'
            else:
                sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_' + np.str.upper(sess_name[:3])
            data.append(pd.Series(b1))
            data_names.extend([sheet_name + '_info'])
            data.append(pd.Series(lins + width))
            data_names.extend([sheet_name + '_bins'])         
            
data.append(pd.Series(np.mean(I_mean)))
data_names.extend(['shuffled_value'])

df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
df.to_excel("info_hist_REM.xlsx", sheet_name='info_hist_REM')  


# Mean vals

# In[96]:


import numpy as np
import matplotlib.pyplot as plt

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
ylims['EV_OF'] = [0,0.2]
ylims['EV_C'] = [0,0.2]
ylims['cov'] = [40,100]
ylims['DEV_OF'] = [0,0.06]
ylims['DEV_C'] = [0,0.06]

ytics = {}
ytics['corr'] = [0.0, 0.25, 0.5, 0.75, 1]
ytics['dist'] = [0, 50, 100, 150]
ytics['EV_OF'] = [0, 0.05,0.1, 0.15, 0.2]
ytics['EV_C'] = [0, 0.05,0.1, 0.15, 0.2]
ytics['cov'] = [40,100]
ytics['DEV_OF'] = [0,0.06]
ytics['DEV_C'] = [0,0.06]
ytics['info'] = [0.0, 25, 50, 75,100]

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

plot_stats(Information_rate_Toroidal_OF, Information_rate_Spatial_OF,  
    Information_rate_OF_lbls, 'info', 'OF')
plot_stats(Information_rate_Toroidal_C, Information_rate_Spatial_C,  
    Information_rate_C_lbls, 'info', 'C')

plot_stats(Explained_deviance_Toroidal_OF, Explained_deviance_Spatial_OF,  
    Explained_variance_OF_lbls, 'EV_OF', 'OF')

plot_stats(Explained_deviance_Toroidal_C, Explained_deviance_Spatial_C,  
    Explained_variance_C_lbls, 'EV_C', 'C')


# In[102]:



    
    
#plot_stats(Information_rate_Toroidal_OF, Information_rate_Spatial_OF,  
#    Information_rate_OF_lbls, 'info', 'OF')
data = []
data_names = []
data.append(pd.Series(Information_rate_Toroidal_OF, index = Information_rate_OF_lbls))
data.append(pd.Series(Information_rate_Spatial_OF, index = Information_rate_OF_lbls))
data_names.extend(['Toroidal'])
data_names.extend(['Spatial'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "info_mean_OF.xlsx"
df.to_excel(fname, sheet_name=fname)  

#plot_stats(Information_rate_Toroidal_C, Information_rate_Spatial_C,  
#    Information_rate_C_lbls, 'info', 'C')

data = []
data_names = []
data.append(pd.Series(Information_rate_Toroidal_C, index = Information_rate_C_lbls))
data.append(pd.Series(Information_rate_Spatial_C, index = Information_rate_C_lbls))
data_names.extend(['Toroidal'])
data_names.extend(['Spatial'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "info_mean_C.xlsx"
df.to_excel(fname, sheet_name=fname)  


#plot_stats(Explained_deviance_Toroidal_OF, Explained_deviance_Spatial_OF,  
#    Explained_variance_OF_lbls, 'EV_OF', 'OF')

data = []
data_names = []
data.append(pd.Series(Explained_deviance_Toroidal_OF, index = Explained_variance_OF_lbls))
data.append(pd.Series(Explained_deviance_Spatial_OF, index = Explained_variance_OF_lbls))
data_names.extend(['Toroidal'])
data_names.extend(['Spatial'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "Expl_dev_mean_OF.xlsx"
df.to_excel(fname, sheet_name=fname)  


#plot_stats(Explained_deviance_Toroidal_C, Explained_deviance_Spatial_C,  
#    Explained_variance_C_lbls, 'EV_C', 'C')

data = []
data_names = []
data.append(pd.Series(Explained_deviance_Toroidal_C, index = Explained_variance_C_lbls))
data.append(pd.Series(Explained_deviance_Spatial_C, index = Explained_variance_C_lbls))
data_names.extend(['Toroidal'])
data_names.extend(['Spatial'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "Expl_dev_mean_C.xlsx"
df.to_excel(fname, sheet_name=fname)  


# ### Phase Distributions

# In[105]:



from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import *

for bnew in [True, False]:
    rat_name = 'roger'#args[1].strip()
    mod_name = 'mod3'#args[2].strip()
    sess_name_0 = 'maze'#args[3].strip()
    sess_name_1 = 'box'#args[4].strip()


    file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1
    if bnew == True:
        f = np.load('Results/Orig/' + file_name + '_alignment3.npz', allow_pickle = True)
        masscenters_1_all = [f['masscenters_1']]
        masscenters_2_all = [f['masscenters_2']]
    #    mtot_1 = f['mtot_1']
    #    mtot_2 = f['mtot_2']
        f.close()
        addtext = ''

    else:
        f = np.load('Results/Orig/' + file_name + '_alignment_same_proj.npz', allow_pickle = True)
        masscenters_1_all = [f['masscenters_11']]
        masscenters_2_all = [f['masscenters_21']]

        masscenters_1_all.append(f['masscenters_12'])
        masscenters_2_all.append(f['masscenters_22'])    
        f.close()
        addtext = '_same'

    ##################### Parameters ############################

    num_neurons = np.shape(masscenters_1_all[0])[0]

    ##################### Plot Phase distribution ############################
    for m in range(len(masscenters_1_all)):
        masscenters_1 = masscenters_1_all[m]
        masscenters_2 = masscenters_2_all[m]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axis('off')
        labels = [sess_name_0, sess_name_1]

        for i in np.arange(num_neurons):

            if i == 0:
                ax.scatter(masscenters_1[i,0], masscenters_1[i,1], s = 10, c = 'r', label = labels[0])
                ax.scatter(masscenters_2[i,0], masscenters_2[i,1], s = 10, c = 'g', label = labels[1])
            else:
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
    #    plt.savefig('Figs/'+ file_name + 'phase_distr' + addtext + str(m) + '.png', bbox_inches='tight', pad_inches=0)
    #    plt.savefig('Figs/'+ file_name + 'phase_distr' + addtext + str(m) + '.pdf', bbox_inches='tight', pad_inches=0)


# In[375]:



from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import *

rat_name = 'roger'#args[1].strip()
mod_name = 'mod3'#args[2].strip()
sess_name_0 = 'maze'#args[3].strip()
sess_name_1 = 'box'#args[4].strip()
file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1

f = np.load('Results/Orig/' + file_name + '_alignment_same_proj.npz', allow_pickle = True)
masscenters_1_all = [f['masscenters_11']]
masscenters_2_all = [f['masscenters_21']]
f.close()

data = []
data_names = []
data.append(pd.Series(masscenters_1_all[0][:,0]*180/np.pi))
data.append(pd.Series(masscenters_1_all[0][:,1]*180/np.pi))
data.append(pd.Series(masscenters_2_all[0][:,0]*180/np.pi))
data.append(pd.Series(masscenters_2_all[0][:,1]*180/np.pi))
data_names.extend([sess_name_1 + '_coord1'])
data_names.extend([sess_name_1 + '_coord2'])
data_names.extend([sess_name_0 + '_coord1'])
data_names.extend([sess_name_0 + '_coord2'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "phase_distr_R3_OF_WW_common.xlsx"
df.to_excel(fname, sheet_name=fname[:-5])  


# In[381]:


for (rat_name, mod_name, sess_name_1,sess_name_0) in (('roger', 'mod1', 'box', 'maze'),
                                                      ('roger', 'mod1', 'box_rec2', 'rem'),
                                                      ('roger', 'mod1', 'box_rec2', 'sws_c0'),
                                                      ('roger', 'mod3', 'box', 'maze'),
                                                      ('roger', 'mod2', 'box_rec2', 'rem'),
                                                      ('roger', 'mod2', 'box_rec2', 'sws'),
                                                      ('roger', 'mod2', 'box_rec2', 'rem'),
                                                      ('roger', 'mod2', 'box_rec2', 'sws'),
                                                      ('roger', 'mod4', 'box', 'maze'),
                                                      ('roger', 'mod3', 'box_rec2', 'rem'),
                                                      ('roger', 'mod3', 'box_rec2', 'sws'),
                                                      ('quentin', 'mod1', 'box', 'maze'),
                                                      ('quentin', 'mod1', 'box', 'rem'),
                                                      ('quentin', 'mod1', 'box', 'sws'),
                                                      ('quentin', 'mod2', 'box', 'maze'),
                                                      ('quentin', 'mod2', 'box', 'rem'),
                                                      ('quentin', 'mod2', 'box', 'sws'),
                                                      ('shane', 'mod1', 'box', 'maze')):
    file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1
    f = np.load('Results/Orig/' + file_name + '_alignment3.npz', allow_pickle = True)
    masscenters_1_all = [f['masscenters_1']]
    masscenters_2_all = [f['masscenters_2']]
    f.close()
    addtext = ''
    if (rat_name == 'roger') & (sess_name_0 in ('maze', 'box')):
        if (mod_name == 'mod3'):
            mod_name = 'mod2'
        elif (mod_name == 'mod4'):
            mod_name = 'mod3'           
    if sess_name_0 == 'maze':
        sess_name_0 = 'WW'
    elif sess_name_0 == 'box':
        sess_name_0 = 'OF'
    elif sess_name_0 == 'box_rec2':
        sess_name_0 = 'OF2'
    elif sess_name_0 == 'sws_class1':
        sess_name_0 = 'SWS_Bursty'
    else:
        sess_name_0 = np.str.upper(sess_name_0[:3])        
    if sess_name_1 == 'maze':
        sess_name_1 = 'WW'
    elif sess_name_1 == 'box':
        sess_name_1 = 'OF'
    elif sess_name_1 == 'box_rec2':
        sess_name_1 = 'OF2'
    elif sess_name_1 == 'sws_class1':
        sess_name_1 = 'SWS_Bursty'
    else:
        sess_name_1 = np.str.upper(sess_name[:3])
    
    data = []
    data_names = []
    data.append(pd.Series(masscenters_1_all[0][:,0]*180/np.pi))
    data.append(pd.Series(masscenters_1_all[0][:,1]*180/np.pi))
    data.append(pd.Series(masscenters_2_all[0][:,0]*180/np.pi))
    data.append(pd.Series(masscenters_2_all[0][:,1]*180/np.pi))
    data_names.extend([sess_name_1 + '_coord1'])
    data_names.extend([sess_name_1 + '_coord2'])
    data_names.extend([sess_name_0 + '_coord1'])
    data_names.extend([sess_name_0 + '_coord2'])
    df = pd.concat(data, ignore_index=True, axis=1)            
    df.columns = data_names
    fname = "phase_distr_" + np.str.upper(rat_name[0]) + mod_name[3] + '_' + sess_name_1 + sess_name_0
    df.to_excel(fname + '.xlsx', sheet_name=fname)  


# In[107]:



from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import *

args = sys.argv
#rat_name = args[1].strip()
#mod_name = args[2].strip()
#sess_name_0 = args[3].strip()
#sess_name_1 = args[4].strip()
rat_name = 'roger'#args[1].strip()
mod_name = 'mod3'#args[2].strip()
sess_name_0 = 'maze'#args[3].strip()
sess_name_1 = 'box'#args[4].strip()
 
file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1
f = np.load('Results/Orig/' + file_name + '_alignment.npz', allow_pickle = True)
mtot_1 = f['mtot_1']
mtot_2 = f['mtot_2']
masscenters_1 = f['masscenters_1']
#masscenters_2 = f['masscenters_2']
f.close()




from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import *
  
file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1

f = np.load('Results/Orig/' + file_name + '_alignment3.npz', allow_pickle = True)
masscenters_1 = f['masscenters_1']
masscenters_2 = f['masscenters_2']
f.close()
num_neurons = np.shape(masscenters_1)[0]


num_shuffle = 1000
_2_PI = 2*np.pi

dist_shuffle = np.zeros((num_shuffle, num_neurons))
np.random.seed(47)
for i in range(num_shuffle):
    inds = np.arange(num_neurons)
    np.random.shuffle(inds)
    masscenters_1 = masscenters_1[inds,:]
    dist_shuffle[i,:] =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2[inds]),
            np.cos(masscenters_1 - masscenters_2[inds]))),1))


mean_dist_all = np.mean(dist_shuffle, 1)
dist_mean = np.mean(mean_dist_all)

masscenters_2 = np.zeros_like(masscenters_1)
for i in range(num_neurons):
    theta = np.random.rand()*2*np.pi
    r1 = np.cos(theta)
    r2 = np.sin(theta)
    #if np.random.rand()>=0.5:
    #    r1*=-1
    #if np.random.rand()>=0.5:
    #    r2*=-1

    masscenters_2[i,0] = (masscenters_1[i,0] + dist_mean*r1)%(2*np.pi)
    masscenters_2[i,1] = (masscenters_1[i,1] + dist_mean*r2)%(2*np.pi)

print(np.mean(np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2),
                              np.cos(masscenters_1 - masscenters_2))),1))))

##################### Parameters ############################

fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis('off')
labels = [sess_name_0, sess_name_1]
for i in np.arange(num_neurons):
    if i == 0:
        ax.scatter(masscenters_1[i,0], masscenters_1[i,1], s = 10, c = 'r', label = labels[0])
        ax.scatter(masscenters_2[i,0], masscenters_2[i,1], s = 10, c = 'g', label = labels[1])
    else:
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
#plt.savefig('Figs/'+ file_name + 'phase_distr_shuffle_mean.png', bbox_inches='tight', pad_inches=0)
#plt.savefig('Figs/'+ file_name + 'phase_distr_shuffle_mean.pdf', bbox_inches='tight', pad_inches=0)


# In[376]:



from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import *

args = sys.argv
#rat_name = args[1].strip()
#mod_name = args[2].strip()
#sess_name_0 = args[3].strip()
#sess_name_1 = args[4].strip()
rat_name = 'roger'#args[1].strip()
mod_name = 'mod3'#args[2].strip()
sess_name_0 = 'maze'#args[3].strip()
sess_name_1 = 'box'#args[4].strip()
 
file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1
f = np.load('Results/Orig/' + file_name + '_alignment.npz', allow_pickle = True)
mtot_1 = f['mtot_1']
mtot_2 = f['mtot_2']
masscenters_1 = f['masscenters_1']
#masscenters_2 = f['masscenters_2']
f.close()




from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import *
  
file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1

f = np.load('Results/Orig/' + file_name + '_alignment3.npz', allow_pickle = True)
masscenters_1 = f['masscenters_1']
masscenters_2 = f['masscenters_2']
f.close()
num_neurons = np.shape(masscenters_1)[0]


num_shuffle = 1000
_2_PI = 2*np.pi

dist_shuffle = np.zeros((num_shuffle, num_neurons))
np.random.seed(47)
for i in range(num_shuffle):
    inds = np.arange(num_neurons)
    np.random.shuffle(inds)
    masscenters_1 = masscenters_1[inds,:]
    dist_shuffle[i,:] =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2[inds]),
            np.cos(masscenters_1 - masscenters_2[inds]))),1))


mean_dist_all = np.mean(dist_shuffle, 1)
dist_mean = np.mean(mean_dist_all)

masscenters_2 = np.zeros_like(masscenters_1)
for i in range(num_neurons):
    theta = np.random.rand()*2*np.pi
    r1 = np.cos(theta)
    r2 = np.sin(theta)
    #if np.random.rand()>=0.5:
    #    r1*=-1
    #if np.random.rand()>=0.5:
    #    r2*=-1

    masscenters_2[i,0] = (masscenters_1[i,0] + dist_mean*r1)%(2*np.pi)
    masscenters_2[i,1] = (masscenters_1[i,1] + dist_mean*r2)%(2*np.pi)
        
data = []
data_names = []
data.append(pd.Series(masscenters_1_all[0][:,0]*180/np.pi))
data.append(pd.Series(masscenters_1_all[0][:,1]*180/np.pi))
data.append(pd.Series(masscenters_2_all[0][:,0]*180/np.pi))
data.append(pd.Series(masscenters_2_all[0][:,1]*180/np.pi))
data_names.extend([sess_name_1 + '_coord1'])
data_names.extend([sess_name_1 + '_coord2'])
data_names.extend([sess_name_0 + '_coord1'])
data_names.extend([sess_name_0 + '_coord2'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "phase_distr_R3_OF_WW_shuf.xlsx"
df.to_excel(fname, sheet_name=fname[:-5])  


# ### Cumulative distributions dist and corr

# In[116]:



from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import *

args = sys.argv
rat_name = 'roger'
mod_name = 'mod3'
sess_name_0 = 'maze'
sess_name_1 = 'box'

for (rat_name, mod_name, sess_name_1,sess_name_0) in (('roger', 'mod1', 'box', 'maze'),
                                                      ('roger', 'mod1', 'box_rec2', 'rem'),
                                                      ('roger', 'mod1', 'box_rec2', 'sws_c0'),
                                                      ('roger', 'mod3', 'box', 'maze'),
                                                      ('roger', 'mod2', 'box_rec2', 'rem'),
                                                      ('roger', 'mod2', 'box_rec2', 'sws'),
                                                      ('roger', 'mod2', 'box_rec2', 'rem'),
                                                      ('roger', 'mod2', 'box_rec2', 'sws'),
                                                      ('roger', 'mod4', 'box', 'maze'),
                                                      ('roger', 'mod3', 'box_rec2', 'rem'),
                                                      ('roger', 'mod3', 'box_rec2', 'sws'),
                                                      ('quentin', 'mod1', 'box', 'maze'),
                                                      ('quentin', 'mod1', 'box', 'rem'),
                                                      ('quentin', 'mod1', 'box', 'sws'),
                                                      ('quentin', 'mod2', 'box', 'maze'),
                                                      ('quentin', 'mod2', 'box', 'rem'),
                                                      ('quentin', 'mod2', 'box', 'sws'),
                                                      ('shane', 'mod1', 'box', 'maze')):
    file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1
    f = np.load('Results/Orig/' + file_name + '_alignment2.npz', allow_pickle = True)
    mtot_1 = f['mtot_1']
    mtot_2 = f['mtot_2']
    f.close()
    f = np.load('Results/Orig/' + file_name + '_alignment3.npz', allow_pickle = True)
    #mtot_1 = f['mtot_1']
    #mtot_2 = f['mtot_2']
    masscenters_1 = f['masscenters_1']
    masscenters_2 = f['masscenters_2']
    dist = f['dist']
    corr = f['corr']
    f.close()
    print(dist, corr)
    num_neurons = np.shape(masscenters_1)[0]


    f = np.load('Results/Orig/' + file_name + '_alignment_same_proj2.npz', allow_pickle = True)
    mtot_11 = f['mtot_11']
    mtot_22 = f['mtot_22']
    mtot_12 = f['mtot_12']
    mtot_21 = f['mtot_21']
    f.close()

    f = np.load('Results/Orig/' + file_name + '_alignment_same_proj.npz', allow_pickle = True)
    masscenters_11 = f['masscenters_11']
    masscenters_22 = f['masscenters_22']
    masscenters_12 = f['masscenters_12']
    masscenters_21 = f['masscenters_21']
    f.close()

    sig = 2.75
    num_shuffle = 1000

    _2_PI = 2*np.pi
    numangsint = 51
    numangsint_1 = numangsint-1
    bins = np.linspace(0,_2_PI, numangsint)

    cells_all = np.arange(num_neurons)
    bNew = True
    if bNew:
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

            m12 = mtot_12[n,:,:].copy()
            m12[np.isnan(m12)] = np.mean(m12[~np.isnan(m12)])
            m12 = smooth_tuning_map(np.rot90(m12), numangsint, sig, bClose = False)
            m22 = mtot_22[n,:,:].copy()
            m22[np.isnan(m22)] = np.mean(m22[~np.isnan(m22)])
            m22 = smooth_tuning_map(np.rot90(m22), numangsint, sig, bClose = False)

            corr1[n] = pearsonr(m12.flatten(), m22.flatten())[0]

            m11 = mtot_11[n,:,:].copy()
            m11[np.isnan(m11)] = np.mean(m11[~np.isnan(m11)])
            m21 = mtot_21[n,:,:].copy()
            m21[np.isnan(m21)] = np.mean(m21[~np.isnan(m21)])
            m11 = smooth_tuning_map(np.rot90(m11), numangsint, sig, bClose = False)
            m21 = smooth_tuning_map(np.rot90(m21), numangsint, sig, bClose = False)

            corr2[n] = pearsonr(m11.flatten(), m21.flatten())[0]
            mtot_1[n,:,:]= m1
            mtot_2[n,:,:]= m2

        dist =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2),
                                      np.cos(masscenters_1 - masscenters_2))),1))
        dist1 =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_11 - masscenters_21),
                                      np.cos(masscenters_11 - masscenters_21))),1))
        dist2 =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_12 - masscenters_22),
                                      np.cos(masscenters_12 - masscenters_22))),1))

    ##################### Cumulative distribution Correlation ############################
        num_neurons = len(masscenters_1[:,0])
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
    #    np.savez_compressed('Results/Orig/' + file_name + '_alignment_stats',
    #                dist = dist,
    #                corr = corr,
    #                dist_shuffle = dist_shuffle,
    #                corr_shuffle = corr_shuffle,
    #               )
    else:
        f = np.load('Results/Orig/' + file_name + '_alignment_stats.npz', allow_pickle = True)
        dist_shuffle = f['dist_shuffle']
        corr_shuffle = f['corr_shuffle']
        f.close()

    fig = plt.figure()
    ax = plt.axes()
    numbins = 30
    meantemp = np.zeros(numbins)
    corr_all = np.array([])
    mean_corr_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        corr_all = np.concatenate((corr_all, corr_shuffle[i]))
        mean_corr_all[i] = np.mean(corr_shuffle[i])

    meantemp1 = np.histogram(corr_all, range = (-1,1), bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(corr, range = (-1,1), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    ax.plot(x, meantemp, c = 'g', alpha = 0.8, lw = 5)
    ax.plot(x,y, c = 'r', alpha = 0.8, lw = 5)

    y,x = np.histogram(corr1, range = (-1,1), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    ax.plot(x,y, c = [0.3,0.3,0.3], alpha = 0.8, lw = 5)

    y,x = np.histogram(corr2, range = (-1,1), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    ax.plot(x,y, c = [0.3,0.3,0.3], alpha = 0.8, lw = 5)

    xs = [-1, -0.5, 0.0, 0.5, 1.0]
    ax.set_xticks(xs)
    ax.set_xticklabels(np.zeros(len(xs),dtype=str))
    ax.xaxis.set_tick_params(width=1, length =5)
    ys = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_yticks(ys)
    ax.set_yticklabels(np.zeros(len(ys),dtype=str))
    ax.yaxis.set_tick_params(width=1, length =5)

    ax.set_xlim([-1,1])
    ax.set_ylim([-0.05,1.05])

    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()

    ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*1.4))

    plt.gca().axes.spines['top'].set_visible(False)
    plt.gca().axes.spines['right'].set_visible(False)


    ##################### Cumulative distribution Distance ############################
    fig = plt.figure()
    ax = plt.axes()
    numbins = 30
    meantemp = np.zeros(numbins)
    dist_all = np.array([])
    mean_dist_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        dist_all = np.concatenate((dist_all, dist_shuffle[i]))
        mean_dist_all[i] = np.mean(dist_shuffle[i])

    meantemp1 = np.histogram(dist_all, range = (0,np.sqrt(2)*np.pi), bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(dist, range = (0,np.sqrt(2)*np.pi), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= 180/np.pi
    ax.plot(x, meantemp, c = 'g', alpha = 0.8, lw = 5)
    ax.plot(x,y, c = 'r', alpha = 0.8, lw = 5)

    y,x = np.histogram(dist1, range = (0,np.sqrt(2)*np.pi), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= 180/np.pi
    ax.plot(x,y, c = [0.3,0.3,0.3], alpha = 0.8, lw = 5)

    y,x = np.histogram(dist2, range = (0,np.sqrt(2)*np.pi), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= 180/np.pi
    ax.plot(x,y, c = [0.3,0.3,0.3], alpha = 0.8, lw = 5)


    ys = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_yticks(ys)
    ax.set_yticklabels(np.zeros(len(ys),dtype=str))
    ax.yaxis.set_tick_params(width=1, length =5)

    xs = [0, 62.5, 125, 187.5, 250]
    ax.set_xticks(xs)
    ax.set_xticklabels(np.zeros(len(xs),dtype=str))
    ax.xaxis.set_tick_params(width=1, length =5)

    ax.set_xlim([0,255])
    ax.set_ylim([-0.05,1.05])

    plt.gca().axes.spines['top'].set_visible(False)
    plt.gca().axes.spines['right'].set_visible(False)

    #ax.set_xticks([], [])
    #ax.set_yticks([], [])
    #plt.axis('off')
    #mean_dist = np.mean(dist)
    #p_val = (sum(mean_dist_all<mean_dist)+1)/(num_shuffle+1)

    # Plot expected
    #plt.plot([0,0], [0, np.max(y)], c = 'k', ls = '--', lw = 1)
    #plt.title('p: ' + str(p_val))
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()

    ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*1.4))


# In[383]:


fig = plt.figure()
ax = plt.axes()
numbins = 30
meantemp = np.zeros(numbins)
corr_all = np.array([])
mean_corr_all = np.zeros(num_shuffle)
for i in range(num_shuffle):
    corr_all = np.concatenate((corr_all, corr_shuffle[i]))
    mean_corr_all[i] = np.mean(corr_shuffle[i])

meantemp1 = np.histogram(corr_all, range = (-1,1), bins = numbins)[0]
meantemp = np.cumsum(meantemp1)
meantemp = np.divide(meantemp, num_shuffle)
meantemp = np.divide(meantemp, num_neurons)
y,x = np.histogram(corr, range = (-1,1), bins = numbins)
y = np.cumsum(y)
y = np.divide(y, num_neurons)
x = x[1:]-(x[1]-x[0])/2
ax.plot(x, meantemp, c = 'g', alpha = 0.8, lw = 5)
ax.plot(x,y, c = 'r', alpha = 0.8, lw = 5)

y,x = np.histogram(corr1, range = (-1,1), bins = numbins)
y = np.cumsum(y)
y = np.divide(y, num_neurons)
x = x[1:]-(x[1]-x[0])/2
ax.plot(x,y, c = [0.3,0.3,0.3], alpha = 0.8, lw = 5)

y,x = np.histogram(corr2, range = (-1,1), bins = numbins)
y = np.cumsum(y)
y = np.divide(y, num_neurons)
x = x[1:]-(x[1]-x[0])/2
ax.plot(x,y, c = [0.3,0.3,0.3], alpha = 0.8, lw = 5)

xs = [-1, -0.5, 0.0, 0.5, 1.0]
ax.set_xticks(xs)
ax.set_xticklabels(np.zeros(len(xs),dtype=str))
ax.xaxis.set_tick_params(width=1, length =5)
ys = [0.0, 0.25, 0.5, 0.75, 1.0]
ax.set_yticks(ys)
ax.set_yticklabels(np.zeros(len(ys),dtype=str))
ax.yaxis.set_tick_params(width=1, length =5)

ax.set_xlim([-1,1])
ax.set_ylim([-0.05,1.05])

x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()

ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*1.4))

plt.gca().axes.spines['top'].set_visible(False)
plt.gca().axes.spines['right'].set_visible(False)


##################### Cumulative distribution Distance ############################
fig = plt.figure()
ax = plt.axes()
numbins = 30
meantemp = np.zeros(numbins)
dist_all = np.array([])
mean_dist_all = np.zeros(num_shuffle)
for i in range(num_shuffle):
    dist_all = np.concatenate((dist_all, dist_shuffle[i]))
    mean_dist_all[i] = np.mean(dist_shuffle[i])

meantemp1 = np.histogram(dist_all, range = (0,np.sqrt(2)*np.pi), bins = numbins)[0]
meantemp = np.cumsum(meantemp1)
meantemp = np.divide(meantemp, num_shuffle)
meantemp = np.divide(meantemp, num_neurons)
y,x = np.histogram(dist, range = (0,np.sqrt(2)*np.pi), bins = numbins)
y = np.cumsum(y)
y = np.divide(y, num_neurons)
x = x[1:]-(x[1]-x[0])/2
x *= 180/np.pi
ax.plot(x, meantemp, c = 'g', alpha = 0.8, lw = 5)
ax.plot(x,y, c = 'r', alpha = 0.8, lw = 5)

y,x = np.histogram(dist1, range = (0,np.sqrt(2)*np.pi), bins = numbins)
y = np.cumsum(y)
y = np.divide(y, num_neurons)
x = x[1:]-(x[1]-x[0])/2
x *= 180/np.pi
ax.plot(x,y, c = [0.3,0.3,0.3], alpha = 0.8, lw = 5)

y,x = np.histogram(dist2, range = (0,np.sqrt(2)*np.pi), bins = numbins)
y = np.cumsum(y)
y = np.divide(y, num_neurons)
x = x[1:]-(x[1]-x[0])/2
x *= 180/np.pi
ax.plot(x,y, c = [0.3,0.3,0.3], alpha = 0.8, lw = 5)


ys = [0.0, 0.25, 0.5, 0.75, 1.0]
ax.set_yticks(ys)
ax.set_yticklabels(np.zeros(len(ys),dtype=str))
ax.yaxis.set_tick_params(width=1, length =5)

xs = [0, 62.5, 125, 187.5, 250]
ax.set_xticks(xs)
ax.set_xticklabels(np.zeros(len(xs),dtype=str))
ax.xaxis.set_tick_params(width=1, length =5)

ax.set_xlim([0,255])
ax.set_ylim([-0.05,1.05])

plt.gca().axes.spines['top'].set_visible(False)
plt.gca().axes.spines['right'].set_visible(False)

#ax.set_xticks([], [])
#ax.set_yticks([], [])
#plt.axis('off')
#mean_dist = np.mean(dist)
#p_val = (sum(mean_dist_all<mean_dist)+1)/(num_shuffle+1)

# Plot expected
#plt.plot([0,0], [0, np.max(y)], c = 'k', ls = '--', lw = 1)
#plt.title('p: ' + str(p_val))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()

ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*1.4))


# In[384]:


file_name


# In[157]:



from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import *

Peak_distance_Toroidal_OFvsC = []
Peak_distance_Shuffled_OFvsC  = []
Pearson_correlation_Toroidal_OFvsC = [] 
Pearson_correlation_Shuffled_OFvsC = []

Peak_distance_Toroidal_OFvsREM = []
Peak_distance_Shuffled_OFvsREM  = []
Pearson_correlation_Toroidal_OFvsREM = [] 
Pearson_correlation_Shuffled_OFvsREM = []

Peak_distance_Toroidal_OFvsSWS = []
Peak_distance_Shuffled_OFvsSWS  = []
Pearson_correlation_Toroidal_OFvsSWS = [] 
Pearson_correlation_Shuffled_OFvsSWS = []


for (rat_name, mod_name, sess_name_1,sess_name_0) in (('roger', 'mod1', 'box', 'maze'),
                                                      ('roger', 'mod1', 'box_rec2', 'rem'),
                                                      ('roger', 'mod1', 'box_rec2', 'sws_c0'),
                                                      ('roger', 'mod3', 'box', 'maze'),
                                                      ('roger', 'mod2', 'box_rec2', 'rem'),
                                                      ('roger', 'mod2', 'box_rec2', 'sws'),
                                                      ('roger', 'mod2', 'box_rec2', 'rem'),
                                                      ('roger', 'mod2', 'box_rec2', 'sws'),
                                                      ('roger', 'mod4', 'box', 'maze'),
                                                      ('roger', 'mod3', 'box_rec2', 'rem'),
                                                      ('roger', 'mod3', 'box_rec2', 'sws'),
                                                      ('quentin', 'mod1', 'box', 'maze'),
                                                      ('quentin', 'mod1', 'box', 'rem'),
                                                      ('quentin', 'mod1', 'box', 'sws'),
                                                      ('quentin', 'mod2', 'box', 'maze'),
                                                      ('quentin', 'mod2', 'box', 'rem'),
                                                      ('quentin', 'mod2', 'box', 'sws'),
                                                      ('shane', 'mod1', 'box', 'maze')):
    file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1
    f = np.load('Results/Orig/' + file_name + '_alignment2.npz', allow_pickle = True)
    mtot_1 = f['mtot_1']
    mtot_2 = f['mtot_2']
    f.close()
    f = np.load('Results/Orig/' + file_name + '_alignment3.npz', allow_pickle = True)
    #mtot_1 = f['mtot_1']
    #mtot_2 = f['mtot_2']
    masscenters_1 = f['masscenters_1']
    masscenters_2 = f['masscenters_2']
    dist = f['dist']
    corr = f['corr']
    f.close()
    print(dist, corr)
    num_neurons = np.shape(masscenters_1)[0]


    f = np.load('Results/Orig/' + file_name + '_alignment_same_proj2.npz', allow_pickle = True)
    mtot_11 = f['mtot_11']
    mtot_22 = f['mtot_22']
    mtot_12 = f['mtot_12']
    mtot_21 = f['mtot_21']
    f.close()

    f = np.load('Results/Orig/' + file_name + '_alignment_same_proj.npz', allow_pickle = True)
    masscenters_11 = f['masscenters_11']
    masscenters_22 = f['masscenters_22']
    masscenters_12 = f['masscenters_12']
    masscenters_21 = f['masscenters_21']
    f.close()

    sig = 2.75
    num_shuffle = 1000

    _2_PI = 2*np.pi
    numangsint = 51
    numangsint_1 = numangsint-1
    bins = np.linspace(0,_2_PI, numangsint)

    cells_all = np.arange(num_neurons)
    bNew = True
    if bNew:
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

            m12 = mtot_12[n,:,:].copy()
            m12[np.isnan(m12)] = np.mean(m12[~np.isnan(m12)])
            m12 = smooth_tuning_map(np.rot90(m12), numangsint, sig, bClose = False)
            m22 = mtot_22[n,:,:].copy()
            m22[np.isnan(m22)] = np.mean(m22[~np.isnan(m22)])
            m22 = smooth_tuning_map(np.rot90(m22), numangsint, sig, bClose = False)

            corr1[n] = pearsonr(m12.flatten(), m22.flatten())[0]

            m11 = mtot_11[n,:,:].copy()
            m11[np.isnan(m11)] = np.mean(m11[~np.isnan(m11)])
            m21 = mtot_21[n,:,:].copy()
            m21[np.isnan(m21)] = np.mean(m21[~np.isnan(m21)])
            m11 = smooth_tuning_map(np.rot90(m11), numangsint, sig, bClose = False)
            m21 = smooth_tuning_map(np.rot90(m21), numangsint, sig, bClose = False)

            corr2[n] = pearsonr(m11.flatten(), m21.flatten())[0]
            mtot_1[n,:,:]= m1
            mtot_2[n,:,:]= m2

        dist =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2),
                                      np.cos(masscenters_1 - masscenters_2))),1))
        dist1 =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_11 - masscenters_21),
                                      np.cos(masscenters_11 - masscenters_21))),1))
        dist2 =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_12 - masscenters_22),
                                      np.cos(masscenters_12 - masscenters_22))),1))

    ##################### Cumulative distribution Correlation ############################
        num_neurons = len(masscenters_1[:,0])
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
        distshuffmeans = np.zeros(num_shuffle)
        corrshuffmeans = np.zeros(num_shuffle)
        for i in range(num_shuffle):
            distshuffmeans[i] = np.mean(dist_shuffle[i])
            corrshuffmeans[i] = np.mean(corr_shuffle[i])

        if sess_name_0 == 'maze':
            Peak_distance_Toroidal_OFvsC.extend([np.mean(dist)])
            Peak_distance_Toroidal_OFvsC.extend([np.std(dist)/np.sqrt(len(dist))])
            Peak_distance_Shuffled_OFvsC.extend([np.mean(distshuffmeans)])
            Peak_distance_Shuffled_OFvsC.extend([np.std(distshuffmeans)])
            Pearson_correlation_Toroidal_OFvsC.extend([np.mean(corr)])
            Pearson_correlation_Toroidal_OFvsC.extend([np.std(corr)/np.sqrt(len(corr))])
            Pearson_correlation_Shuffled_OFvsC.extend([np.mean(corrshuffmeans)])
            Pearson_correlation_Shuffled_OFvsC.extend([np.std(corrshuffmeans)])
        elif sess_name_0[:3] == 'rem':
            Peak_distance_Toroidal_OFvsREM.extend([np.mean(dist)])
            Peak_distance_Toroidal_OFvsREM.extend([np.std(dist)/np.sqrt(len(dist))])
            Peak_distance_Shuffled_OFvsREM.extend([np.mean(distshuffmeans)])
            Peak_distance_Shuffled_OFvsREM.extend([np.std(distshuffmeans)])
            Pearson_correlation_Toroidal_OFvsREM.extend([np.mean(corr)])
            Pearson_correlation_Toroidal_OFvsREM.extend([np.std(corr)/np.sqrt(len(corr))])
            Pearson_correlation_Shuffled_OFvsREM.extend([np.mean(corrshuffmeans)])
            Pearson_correlation_Shuffled_OFvsREM.extend([np.std(corrshuffmeans)])
        elif sess_name_0[:3] == 'sws':
            Peak_distance_Toroidal_OFvsSWS.extend([np.mean(dist)])
            Peak_distance_Toroidal_OFvsSWS.extend([np.std(dist)/np.sqrt(len(dist))])
            Peak_distance_Shuffled_OFvsSWS.extend([np.mean(distshuffmeans)])
            Peak_distance_Shuffled_OFvsSWS.extend([np.std(distshuffmeans)])
            Pearson_correlation_Toroidal_OFvsSWS.extend([np.mean(corr)])
            Pearson_correlation_Toroidal_OFvsSWS.extend([np.std(corr)/np.sqrt(len(corr))])
            Pearson_correlation_Shuffled_OFvsSWS.extend([np.mean(corrshuffmeans)])
            Pearson_correlation_Shuffled_OFvsSWS.extend([np.std(corrshuffmeans)])


# In[123]:




for (rat_name, mod_name, sess_name_1,sess_name_0) in (#('roger', 'mod1', 'box', 'maze'),
                                                      ('roger', 'mod1', 'box_rec2', 'rem'),
                                                      ('roger', 'mod1', 'box_rec2', 'sws_c0'),
    
                                                      ('roger', 'mod3', 'box', 'maze'),
                                                      ('roger', 'mod2', 'box_rec2', 'rem'),
                                                      ('roger', 'mod2', 'box_rec2', 'sws'),
    
                                                      ('roger', 'mod4', 'box', 'maze'),
                                                      ('roger', 'mod3', 'box_rec2', 'rem'),
                                                      ('roger', 'mod3', 'box_rec2', 'sws'),
    
                                                      ('quentin', 'mod1', 'box', 'maze'),
                                                      ('quentin', 'mod1', 'box', 'rem'),
                                                      ('quentin', 'mod1', 'box', 'sws'),
                                                      ('quentin', 'mod2', 'box', 'maze'),
                                                      ('quentin', 'mod2', 'box', 'rem'),
                                                      ('quentin', 'mod2', 'box', 'sws'),
                                                      ('shane', 'mod1', 'box', 'maze')):
    file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1
    f = np.load('Results/Orig/' + file_name + '_alignment2.npz', allow_pickle = True)
    mtot_1 = f['mtot_1']
    mtot_2 = f['mtot_2']
    f.close()
    f = np.load('Results/Orig/' + file_name + '_alignment3.npz', allow_pickle = True)
    #mtot_1 = f['mtot_1']
    #mtot_2 = f['mtot_2']
    masscenters_1 = f['masscenters_1']
    masscenters_2 = f['masscenters_2']
    dist = f['dist']
    corr = f['corr']
    f.close()
    print(dist, corr)
    num_neurons = np.shape(masscenters_1)[0]


    f = np.load('Results/Orig/' + file_name + '_alignment_same_proj2.npz', allow_pickle = True)
    mtot_11 = f['mtot_11']
    mtot_22 = f['mtot_22']
    mtot_12 = f['mtot_12']
    mtot_21 = f['mtot_21']
    f.close()

    f = np.load('Results/Orig/' + file_name + '_alignment_same_proj.npz', allow_pickle = True)
    masscenters_11 = f['masscenters_11']
    masscenters_22 = f['masscenters_22']
    masscenters_12 = f['masscenters_12']
    masscenters_21 = f['masscenters_21']
    f.close()

    sig = 2.75
    num_shuffle = 1000

    _2_PI = 2*np.pi
    numangsint = 51
    numangsint_1 = numangsint-1
    bins = np.linspace(0,_2_PI, numangsint)

    cells_all = np.arange(num_neurons)
    bNew = True
    if bNew:
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

            m12 = mtot_12[n,:,:].copy()
            m12[np.isnan(m12)] = np.mean(m12[~np.isnan(m12)])
            m12 = smooth_tuning_map(np.rot90(m12), numangsint, sig, bClose = False)
            m22 = mtot_22[n,:,:].copy()
            m22[np.isnan(m22)] = np.mean(m22[~np.isnan(m22)])
            m22 = smooth_tuning_map(np.rot90(m22), numangsint, sig, bClose = False)

            corr1[n] = pearsonr(m12.flatten(), m22.flatten())[0]

            m11 = mtot_11[n,:,:].copy()
            m11[np.isnan(m11)] = np.mean(m11[~np.isnan(m11)])
            m21 = mtot_21[n,:,:].copy()
            m21[np.isnan(m21)] = np.mean(m21[~np.isnan(m21)])
            m11 = smooth_tuning_map(np.rot90(m11), numangsint, sig, bClose = False)
            m21 = smooth_tuning_map(np.rot90(m21), numangsint, sig, bClose = False)

            corr2[n] = pearsonr(m11.flatten(), m21.flatten())[0]
            mtot_1[n,:,:]= m1
            mtot_2[n,:,:]= m2

        dist =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2),
                                      np.cos(masscenters_1 - masscenters_2))),1))
        dist1 =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_11 - masscenters_21),
                                      np.cos(masscenters_11 - masscenters_21))),1))
        dist2 =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_12 - masscenters_22),
                                      np.cos(masscenters_12 - masscenters_22))),1))

    ##################### Cumulative distribution Correlation ############################
        num_neurons = len(masscenters_1[:,0])
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
        distshuffmeans = np.zeros(num_shuffle)
        corrshuffmeans = np.zeros(num_shuffle)
        for i in range(num_shuffle):
            distshuffmeans[i] = np.mean(dist_shuffle[i])
            corrshuffmeans[i] = np.mean(corr_shuffle[i])

        if sess_name_0 == 'maze':
            Peak_distance_Toroidal_OFvsC.extend([np.mean(dist)])
            Peak_distance_Toroidal_OFvsC.extend([np.std(dist)/np.sqrt(len(dist))])
            Peak_distance_Shuffled_OFvsC.extend([np.mean(distshuffmeans)])
            Peak_distance_Shuffled_OFvsC.extend([np.std(distshuffmeans)])
            Pearson_correlation_Toroidal_OFvsC.extend([np.mean(corr)])
            Pearson_correlation_Toroidal_OFvsC.extend([np.std(corr)/np.sqrt(len(corr))])
            Pearson_correlation_Shuffled_OFvsC.extend([np.mean(corrshuffmeans)])
            Pearson_correlation_Shuffled_OFvsC.extend([np.std(corrshuffmeans)])
        elif sess_name_0[:3] == 'rem':
            Peak_distance_Toroidal_OFvsREM.extend([np.mean(dist)])
            Peak_distance_Toroidal_OFvsREM.extend([np.std(dist)/np.sqrt(len(dist))])
            Peak_distance_Shuffled_OFvsREM.extend([np.mean(distshuffmeans)])
            Peak_distance_Shuffled_OFvsREM.extend([np.std(distshuffmeans)])
            Pearson_correlation_Toroidal_OFvsREM.extend([np.mean(corr)])
            Pearson_correlation_Toroidal_OFvsREM.extend([np.std(corr)/np.sqrt(len(corr))])
            Pearson_correlation_Shuffled_OFvsREM.extend([np.mean(corrshuffmeans)])
            Pearson_correlation_Shuffled_OFvsREM.extend([np.std(corrshuffmeans)])
        elif sess_name_0[:3] == 'sws':
            Peak_distance_Toroidal_OFvsSWS.extend([np.mean(dist)])
            Peak_distance_Toroidal_OFvsSWS.extend([np.std(dist)/np.sqrt(len(dist))])
            Peak_distance_Shuffled_OFvsSWS.extend([np.mean(distshuffmeans)])
            Peak_distance_Shuffled_OFvsSWS.extend([np.std(distshuffmeans)])
            Pearson_correlation_Toroidal_OFvsSWS.extend([np.mean(corr)])
            Pearson_correlation_Toroidal_OFvsSWS.extend([np.std(corr)/np.sqrt(len(corr))])
            Pearson_correlation_Shuffled_OFvsSWS.extend([np.mean(corrshuffmeans)])
            Pearson_correlation_Shuffled_OFvsSWS.extend([np.std(corrshuffmeans)])

    numbins = 30
    meantemp = np.zeros(numbins)
    corr_all = np.array([])
    mean_corr_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        corr_all = np.concatenate((corr_all, corr_shuffle[i]))
        mean_corr_all[i] = np.mean(corr_shuffle[i])

    meantemp1 = np.histogram(corr_all, range = (-1,1), bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(corr, range = (-1,1), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
        
    data = []
    data_names = []
    data.append(pd.Series(x))
    data.append(pd.Series(meantemp))
    data.append(pd.Series(y))
    data_names.extend(['Bins'])
    data_names.extend(['Shuffled'])
    data_names.extend(['Original'])

    y,x = np.histogram(corr1, range = (-1,1), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    
    data.append(pd.Series(y))
    data_names.extend(['Common_' + sess_name_1])


    y,x = np.histogram(corr2, range = (-1,1), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    data.append(pd.Series(y))
    data_names.extend(['Common_' + sess_name_0])
    
    
    if (rat_name == 'roger') & (sess_name_0 in ('maze', 'box')):
        if (mod_name == 'mod3'):
            mod_name = 'mod2'
        elif (mod_name == 'mod4'):
            mod_name = 'mod3'           
    if sess_name_0 == 'maze':
        sess_name_0 = 'WW'
    elif sess_name_0 == 'box':
        sess_name_0 = 'OF'
    elif sess_name_0 == 'box_rec2':
        sess_name_0 = 'OF2'
    elif sess_name_0 == 'sws_class1':
        sess_name_0 = 'SWS_Bursty'
    else:
        sess_name_0 = np.str.upper(sess_name_0[:3])        
    if sess_name_1 == 'maze':
        sess_name_1 = 'WW'
    elif sess_name_1 == 'box':
        sess_name_1 = 'OF'
    elif sess_name_1 == 'box_rec2':
        sess_name_1 = 'OF2'
    elif sess_name_1 == 'sws_class1':
        sess_name_1 = 'SWS_Bursty'
    else:
        sess_name_1 = np.str.upper(sess_name[:3])
    
    df = pd.concat(data, ignore_index=True, axis=1)            
    df.columns = data_names
    fname = "corr_" + np.str.upper(rat_name[0]) + mod_name[3] + '_' + sess_name_1 + sess_name_0
    df.to_excel(fname + '.xlsx', sheet_name=fname)  

    ##################### Cumulative distribution Distance ############################
    fig = plt.figure()
    ax = plt.axes()
    numbins = 30
    meantemp = np.zeros(numbins)
    dist_all = np.array([])
    mean_dist_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        dist_all = np.concatenate((dist_all, dist_shuffle[i]))
        mean_dist_all[i] = np.mean(dist_shuffle[i])

    meantemp1 = np.histogram(dist_all, range = (0,np.sqrt(2)*np.pi), bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(dist, range = (0,np.sqrt(2)*np.pi), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= 180/np.pi
    
    data = []
    data_names = []
    data.append(pd.Series(x))
    data.append(pd.Series(meantemp))
    data.append(pd.Series(y))
    data_names.extend(['Bins'])
    data_names.extend(['Shuffled'])
    data_names.extend(['Original'])

    y,x = np.histogram(dist1, range = (0,np.sqrt(2)*np.pi), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= 180/np.pi
    data.append(pd.Series(y))
    data_names.extend(['Common_' + sess_name_1])

    y,x = np.histogram(dist2, range = (0,np.sqrt(2)*np.pi), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= 180/np.pi
    data.append(pd.Series(y))
    data_names.extend(['Common_' + sess_name_0])

    df = pd.concat(data, ignore_index=True, axis=1)            
    df.columns = data_names
    fname = "dist_" + np.str.upper(rat_name[0]) + mod_name[3] + '_' + sess_name_1 + sess_name_0
    df.to_excel(fname + '.xlsx', sheet_name=fname)  


# In[382]:




for (rat_name, mod_name, sess_name_1,sess_name_0) in (
                                                      ('roger', 'mod2', 'box_rec2', 'rem'),
                                                      ('roger', 'mod2', 'box_rec2', 'sws'),):
    file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1
    f = np.load('Results/Orig/' + file_name + '_alignment2.npz', allow_pickle = True)
    mtot_1 = f['mtot_1']
    mtot_2 = f['mtot_2']
    f.close()
    f = np.load('Results/Orig/' + file_name + '_alignment3.npz', allow_pickle = True)
    #mtot_1 = f['mtot_1']
    #mtot_2 = f['mtot_2']
    masscenters_1 = f['masscenters_1']
    masscenters_2 = f['masscenters_2']
    dist = f['dist']
    corr = f['corr']
    f.close()
    print(dist, corr)
    num_neurons = np.shape(masscenters_1)[0]


    f = np.load('Results/Orig/' + file_name + '_alignment_same_proj2.npz', allow_pickle = True)
    mtot_11 = f['mtot_11']
    mtot_22 = f['mtot_22']
    mtot_12 = f['mtot_12']
    mtot_21 = f['mtot_21']
    f.close()

    f = np.load('Results/Orig/' + file_name + '_alignment_same_proj.npz', allow_pickle = True)
    masscenters_11 = f['masscenters_11']
    masscenters_22 = f['masscenters_22']
    masscenters_12 = f['masscenters_12']
    masscenters_21 = f['masscenters_21']
    f.close()

    sig = 2.75
    num_shuffle = 1000

    _2_PI = 2*np.pi
    numangsint = 51
    numangsint_1 = numangsint-1
    bins = np.linspace(0,_2_PI, numangsint)

    cells_all = np.arange(num_neurons)
    bNew = True
    if bNew:
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

            m12 = mtot_12[n,:,:].copy()
            m12[np.isnan(m12)] = np.mean(m12[~np.isnan(m12)])
            m12 = smooth_tuning_map(np.rot90(m12), numangsint, sig, bClose = False)
            m22 = mtot_22[n,:,:].copy()
            m22[np.isnan(m22)] = np.mean(m22[~np.isnan(m22)])
            m22 = smooth_tuning_map(np.rot90(m22), numangsint, sig, bClose = False)

            corr1[n] = pearsonr(m12.flatten(), m22.flatten())[0]

            m11 = mtot_11[n,:,:].copy()
            m11[np.isnan(m11)] = np.mean(m11[~np.isnan(m11)])
            m21 = mtot_21[n,:,:].copy()
            m21[np.isnan(m21)] = np.mean(m21[~np.isnan(m21)])
            m11 = smooth_tuning_map(np.rot90(m11), numangsint, sig, bClose = False)
            m21 = smooth_tuning_map(np.rot90(m21), numangsint, sig, bClose = False)

            corr2[n] = pearsonr(m11.flatten(), m21.flatten())[0]
            mtot_1[n,:,:]= m1
            mtot_2[n,:,:]= m2

        dist =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2),
                                      np.cos(masscenters_1 - masscenters_2))),1))
        dist1 =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_11 - masscenters_21),
                                      np.cos(masscenters_11 - masscenters_21))),1))
        dist2 =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_12 - masscenters_22),
                                      np.cos(masscenters_12 - masscenters_22))),1))

    ##################### Cumulative distribution Correlation ############################
        num_neurons = len(masscenters_1[:,0])
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
        distshuffmeans = np.zeros(num_shuffle)
        corrshuffmeans = np.zeros(num_shuffle)
        for i in range(num_shuffle):
            distshuffmeans[i] = np.mean(dist_shuffle[i])
            corrshuffmeans[i] = np.mean(corr_shuffle[i])
    numbins = 30
    meantemp = np.zeros(numbins)
    corr_all = np.array([])
    mean_corr_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        corr_all = np.concatenate((corr_all, corr_shuffle[i]))
        mean_corr_all[i] = np.mean(corr_shuffle[i])

    meantemp1 = np.histogram(corr_all, range = (-1,1), bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(corr, range = (-1,1), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
        
    data = []
    data_names = []
    data.append(pd.Series(x))
    data.append(pd.Series(meantemp))
    data.append(pd.Series(y))
    data_names.extend(['Bins'])
    data_names.extend(['Shuffled'])
    data_names.extend(['Original'])

    y,x = np.histogram(corr1, range = (-1,1), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    
    data.append(pd.Series(y))
    data_names.extend(['Common_' + sess_name_1])


    y,x = np.histogram(corr2, range = (-1,1), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    data.append(pd.Series(y))
    data_names.extend(['Common_' + sess_name_0])
    
    
    if (rat_name == 'roger') & (sess_name_0 in ('maze', 'box')):
        if (mod_name == 'mod3'):
            mod_name = 'mod2'
        elif (mod_name == 'mod4'):
            mod_name = 'mod3'           
    if sess_name_0 == 'maze':
        sess_name_0 = 'WW'
    elif sess_name_0 == 'box':
        sess_name_0 = 'OF'
    elif sess_name_0 == 'box_rec2':
        sess_name_0 = 'OF2'
    elif sess_name_0 == 'sws_class1':
        sess_name_0 = 'SWS_Bursty'
    else:
        sess_name_0 = np.str.upper(sess_name_0[:3])        
    if sess_name_1 == 'maze':
        sess_name_1 = 'WW'
    elif sess_name_1 == 'box':
        sess_name_1 = 'OF'
    elif sess_name_1 == 'box_rec2':
        sess_name_1 = 'OF2'
    elif sess_name_1 == 'sws_class1':
        sess_name_1 = 'SWS_Bursty'
    else:
        sess_name_1 = np.str.upper(sess_name[:3])
    
    df = pd.concat(data, ignore_index=True, axis=1)            
    df.columns = data_names
    fname = "corr_" + np.str.upper(rat_name[0]) + mod_name[3] + '_' + sess_name_1 + sess_name_0
    df.to_excel(fname + '.xlsx', sheet_name=fname)  

    ##################### Cumulative distribution Distance ############################
    fig = plt.figure()
    ax = plt.axes()
    numbins = 30
    meantemp = np.zeros(numbins)
    dist_all = np.array([])
    mean_dist_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        dist_all = np.concatenate((dist_all, dist_shuffle[i]))
        mean_dist_all[i] = np.mean(dist_shuffle[i])

    meantemp1 = np.histogram(dist_all, range = (0,np.sqrt(2)*np.pi), bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(dist, range = (0,np.sqrt(2)*np.pi), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= 180/np.pi
    
    data = []
    data_names = []
    data.append(pd.Series(x))
    data.append(pd.Series(meantemp))
    data.append(pd.Series(y))
    data_names.extend(['Bins'])
    data_names.extend(['Shuffled'])
    data_names.extend(['Original'])

    y,x = np.histogram(dist1, range = (0,np.sqrt(2)*np.pi), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= 180/np.pi
    data.append(pd.Series(y))
    data_names.extend(['Common_' + sess_name_1])

    y,x = np.histogram(dist2, range = (0,np.sqrt(2)*np.pi), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= 180/np.pi
    data.append(pd.Series(y))
    data_names.extend(['Common_' + sess_name_0])

    df = pd.concat(data, ignore_index=True, axis=1)            
    df.columns = data_names
    fname = "dist_" + np.str.upper(rat_name[0]) + mod_name[3] + '_' + sess_name_1 + sess_name_0
    df.to_excel(fname + '.xlsx', sheet_name=fname)  


# In[121]:


F = np.load('Results/Orig/' + file_name + '_alignment_same_proj2.npz', allow_pickle = True)


# In[122]:


mtot_11 = F['mtot_11']
mtot_22 = F['mtot_22']


# In[169]:


import numpy as np
import matplotlib.pyplot as plt


Peak_distance_OFvsC_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM','S1','S1_SEM' ]
Peak_distance_OFvsREM_lbls = ['R1','R1_SEM','R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM']
Peak_distance_OFvsSWS_lbls = ['R2','R2_SEM','R3','R3_SEM', 'Q1','Q1_SEM','Q2','Q2_SEM' ]

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
ylims['EV_OF'] = [0,0.2]
ylims['EV_C'] = [0,0.2]
ylims['cov'] = [40,100]
ylims['DEV_OF'] = [0,0.06]
ylims['DEV_C'] = [0,0.06]

ytics = {}
ytics['corr'] = [0.0, 0.25, 0.5, 0.75, 1]
ytics['dist'] = [0, 50, 100, 150]
ytics['EV_OF'] = [0, 0.05,0.1, 0.15, 0.2]
ytics['EV_C'] = [0, 0.05,0.1, 0.15, 0.2]
ytics['cov'] = [40,100]
ytics['DEV_OF'] = [0,0.06]
ytics['DEV_C'] = [0,0.06]
ytics['info'] = [0.0, 25, 50, 75,100]

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
                print(stats_tor[i],stats_space[i], lbls[i])
                ax.scatter(xs, [stats_tor[i],stats_space[i]], s = 250, marker='.', #lw = 3, 
                    color = colors_mods[lbls[i]], zorder=-1)
                ax.plot(xs, [stats_tor[i],stats_space[i]], lw = 4, c = colors_mods[lbls[i]][0], alpha = 0.5, zorder = -2)

        if bSems:
            for i in range(num_mods):
                i*=2
                print(stats_tor[i],stats_space[i], lbls[i])
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

plot_stats(Peak_distance_Toroidal_OFvsC, Peak_distance_Shuffled_OFvsC,  
    Peak_distance_OFvsC_lbls, 'dist', 'OFvsC')
plot_stats(Peak_distance_Toroidal_OFvsREM, Peak_distance_Shuffled_OFvsREM,  
    Peak_distance_OFvsREM_lbls, 'dist', 'OFvsREM')
plot_stats(Peak_distance_Toroidal_OFvsSWS, Peak_distance_Shuffled_OFvsSWS,  
    Peak_distance_OFvsSWS_lbls, 'dist', 'OFvsSWS')

plot_stats(Pearson_correlation_Toroidal_OFvsC, Pearson_correlation_Shuffled_OFvsC,  
    Pearson_correlation_OFvsC_lbls, 'corr', 'OFvsC')
plot_stats(Pearson_correlation_Toroidal_OFvsREM, Pearson_correlation_Shuffled_OFvsREM,  
    Pearson_correlation_OFvsREM_lbls, 'corr', 'OFvsREM')
plot_stats(Pearson_correlation_Toroidal_OFvsSWS, Pearson_correlation_Shuffled_OFvsSWS,  
    Pearson_correlation_OFvsSWS_lbls, 'corr', 'OFvsSWS')


# In[172]:


#plot_stats(Peak_distance_Toroidal_OFvsC, Peak_distance_Shuffled_OFvsC,  
#    Peak_distance_OFvsC_lbls, 'dist', 'OFvsC')

data = []
data_names = []
data.append(pd.Series(Peak_distance_Toroidal_OFvsC, index = Peak_distance_OFvsC_lbls))
data.append(pd.Series(Peak_distance_Shuffled_OFvsC, index = Peak_distance_OFvsC_lbls))
data_names.extend(['Original'])
data_names.extend(['Shuffled'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "dist_mean_OF_WW.xlsx"
df.to_excel(fname, sheet_name=fname[:-5])  

#plot_stats(Peak_distance_Toroidal_OFvsREM, Peak_distance_Shuffled_OFvsREM,  
#    Peak_distance_OFvsREM_lbls, 'dist', 'OFvsREM')

data = []
data_names = []
data.append(pd.Series(Peak_distance_Toroidal_OFvsREM, index = Peak_distance_OFvsREM_lbls))
data.append(pd.Series(Peak_distance_Shuffled_OFvsREM, index = Peak_distance_OFvsREM_lbls))
data_names.extend(['Original'])
data_names.extend(['Shuffled'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "dist_mean_OF_REM.xlsx"
df.to_excel(fname, sheet_name=fname[:-5])  

#plot_stats(Peak_distance_Toroidal_OFvsSWS, Peak_distance_Shuffled_OFvsSWS,  
#    Peak_distance_OFvsSWS_lbls, 'dist', 'OFvsSWS')

data = []
data_names = []
data.append(pd.Series(Peak_distance_Toroidal_OFvsSWS, index = Peak_distance_OFvsSWS_lbls))
data.append(pd.Series(Peak_distance_Shuffled_OFvsSWS, index = Peak_distance_OFvsSWS_lbls))
data_names.extend(['Original'])
data_names.extend(['Shuffled'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "dist_mean_OF_SWS.xlsx"
df.to_excel(fname, sheet_name=fname[:-5])  

#plot_stats(Pearson_correlation_Toroidal_OFvsC, Pearson_correlation_Shuffled_OFvsC,  
#    Pearson_correlation_OFvsC_lbls, 'corr', 'OFvsC')

data = []
data_names = []
data.append(pd.Series(Pearson_correlation_Toroidal_OFvsC, index = Pearson_correlation_OFvsC_lbls))
data.append(pd.Series(Pearson_correlation_Shuffled_OFvsC, index = Pearson_correlation_OFvsC_lbls))
data_names.extend(['Original'])
data_names.extend(['Shuffled'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "corr_mean_OF_WW.xlsx"
df.to_excel(fname, sheet_name=fname[:-5])  

#plot_stats(Pearson_correlation_Toroidal_OFvsREM, Pearson_correlation_Shuffled_OFvsREM,  
#    Pearson_correlation_OFvsREM_lbls, 'corr', 'OFvsREM')

data = []
data_names = []
data.append(pd.Series(Pearson_correlation_Toroidal_OFvsREM, index = Pearson_correlation_OFvsREM_lbls))
data.append(pd.Series(Pearson_correlation_Shuffled_OFvsREM, index = Pearson_correlation_OFvsREM_lbls))
data_names.extend(['Original'])
data_names.extend(['Shuffled'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "corr_mean_OF_REM.xlsx"
df.to_excel(fname, sheet_name=fname[:-5])  

#plot_stats(Pearson_correlation_Toroidal_OFvsSWS, Pearson_correlation_Shuffled_OFvsSWS,  
#    Pearson_correlation_OFvsSWS_lbls, 'corr', 'OFvsSWS')

data = []
data_names = []
data.append(pd.Series(Pearson_correlation_Toroidal_OFvsSWS, index = Pearson_correlation_OFvsSWS_lbls))
data.append(pd.Series(Pearson_correlation_Shuffled_OFvsSWS, index = Pearson_correlation_OFvsSWS_lbls))
data_names.extend(['Original'])
data_names.extend(['Shuffled'])
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = "corr_mean_OF_SWS.xlsx"
df.to_excel(fname, sheet_name=fname[:-5])  


# ### Simulation barcodes

# In[176]:


import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
import matplotlib.gridspec as grd

file_name = 'sqr_d'

f = np.load('Results/Orig/' + file_name + '_H2.npz', allow_pickle = True)
persistence = list(f['diagrams'])
f.close()

diagrams_roll = {}
filenames=glob.glob('Main/Results/Roll/' + file_name + '_H2_roll_*')
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

num_rolls = 0

print('Number of rolls' + str(num_rolls))


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
#fig,axesall = plt.subplots(len(dims), 1)
fig = plt.figure()
gs = grd.GridSpec(len(dims),1)#, height_ratios=[1,10], width_ratios=[6,1], wspace=0.1)

indsall =  0
labels = ["$H_0$", "$H_1$", "$H_2$"]
for dit, dim in enumerate(dims):
    axes = plt.subplot(gs[dim])
    axes.axis('off')
    d = np.copy(persistence[dim])
    d[np.isinf(d[:,1]),1] = infinity
    dlife = (d[:,1] - d[:,0])
    dinds = np.argsort(dlife)[-30:]#[int(0.5*len(dlife)):]
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

#        axes.set_ylabel('between y1 and 0')

#        axes.plot(ytemp, 0.5+np.arange(len(dinds)), 
#                c = cs[(dim)], lw = 2,  linestyle = ':', label=labels_roll[0])

    axes.plot([0,0], [0, indsall], c = 'k', linestyle = '-', lw = 1)
    axes.plot([0,infinity],[0,0], c = 'k', linestyle = '-', lw = 1)
    axes.set_xlim([0, infinity])


# In[385]:


import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
import matplotlib.gridspec as grd

file_name = 'sqr_d'

f = np.load('Results/Orig/' + file_name + '_H2.npz', allow_pickle = True)
persistence = list(f['diagrams'])
f.close()

cs = np.repeat([[0,0.55,0.2]],3).reshape(3,3).T
alpha=1
inf_delta=0.1
legend=True
colormap=cs
maxdim = len(persistence)-1
dims =np.arange(maxdim+1)

min_birth, max_death = 0,0            
for dim in dims:
    persistence_dim = persistence[dim][~np.isinf(persistence[dim][:,1]),:]
    min_birth = min(min_birth, np.min(persistence_dim))
    max_death = max(max_death, np.max(persistence_dim))
delta = (max_death - min_birth) * inf_delta
infinity = max_death + delta
axis_start = min_birth - delta            
plotind = (dims[-1]+1)*100 + 10 +1

data = {}

indsall =  0
labels = ["$H_0$", "$H_1$", "$H_2$"]
for dit, dim in enumerate(dims):
    d = np.copy(persistence[dim])
    d[np.isinf(d[:,1]),1] = infinity
    dlife = (d[:,1] - d[:,0])
    dinds = np.argsort(dlife)[-30:]#[int(0.5*len(dlife)):]
    if dim>0:
        dinds = dinds[np.flip(np.argsort(d[dinds,0]))]
    data['Dim_' + str(dim) + '_lifetime'] = dlife[dinds]
    data['Dim_' + str(dim) + '_births'] = d[dinds,0]
    df = pd.DataFrame.from_dict(data)
    df.to_excel('Square_torus.xlsx', sheet_name='Square_torus')


# In[386]:


import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
import matplotlib.gridspec as grd

file_name = 'hex_d'

f = np.load('Results/Orig/' + file_name + '_H2.npz', allow_pickle = True)
persistence = list(f['diagrams'])
f.close()

cs = np.repeat([[0,0.55,0.2]],3).reshape(3,3).T
alpha=1
inf_delta=0.1
legend=True
colormap=cs
maxdim = len(persistence)-1
dims =np.arange(maxdim+1)

min_birth, max_death = 0,0            
for dim in dims:
    persistence_dim = persistence[dim][~np.isinf(persistence[dim][:,1]),:]
    min_birth = min(min_birth, np.min(persistence_dim))
    max_death = max(max_death, np.max(persistence_dim))
delta = (max_death - min_birth) * inf_delta
infinity = max_death + delta
axis_start = min_birth - delta            
plotind = (dims[-1]+1)*100 + 10 +1

data = {}

indsall =  0
labels = ["$H_0$", "$H_1$", "$H_2$"]
for dit, dim in enumerate(dims):
    d = np.copy(persistence[dim])
    d[np.isinf(d[:,1]),1] = infinity
    dlife = (d[:,1] - d[:,0])
    dinds = np.argsort(dlife)[-30:]#[int(0.5*len(dlife)):]
    if dim>0:
        dinds = dinds[np.flip(np.argsort(d[dinds,0]))]
    data['Dim_' + str(dim) + '_lifetime'] = dlife[dinds]
    data['Dim_' + str(dim) + '_births'] =d[dinds,0]
    df = pd.DataFrame.from_dict(data)
    df.to_excel('Hexagonal_torus.xlsx', sheet_name='hexagonal_torus')


# In[388]:


import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
import matplotlib.gridspec as grd

file_name = 'couey_d'

f = np.load('Results/Orig/' + file_name + '_H2.npz', allow_pickle = True)
persistence = list(f['diagrams'])
f.close()

cs = np.repeat([[0,0.55,0.2]],3).reshape(3,3).T
alpha=1
inf_delta=0.1
legend=True
colormap=cs
maxdim = len(persistence)-1
dims =np.arange(maxdim+1)

min_birth, max_death = 0,0            
for dim in dims:
    persistence_dim = persistence[dim][~np.isinf(persistence[dim][:,1]),:]
    min_birth = min(min_birth, np.min(persistence_dim))
    max_death = max(max_death, np.max(persistence_dim))
delta = (max_death - min_birth) * inf_delta
infinity = max_death + delta
axis_start = min_birth - delta            
plotind = (dims[-1]+1)*100 + 10 +1

data = {}

indsall =  0
labels = ["$H_0$", "$H_1$", "$H_2$"]
for dit, dim in enumerate(dims):
    d = np.copy(persistence[dim])
    d[np.isinf(d[:,1]),1] = infinity
    dlife = (d[:,1] - d[:,0])
    dinds = np.argsort(dlife)[-30:]#[int(0.5*len(dlife)):]
    if dim>0:
        dinds = dinds[np.flip(np.argsort(d[dinds,0]))]
    data['Dim_' + str(dim) + '_lifetime'] = dlife[dinds]
    data['Dim_' + str(dim) + '_births'] =d[dinds,0]
    df = pd.DataFrame.from_dict(data)
    df.to_excel('Couey_torus.xlsx', sheet_name='Couey_torus')


# In[389]:


import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
import matplotlib.gridspec as grd

file_name = 'guanella_d'

f = np.load('Results/Orig/' + file_name + '_H2.npz', allow_pickle = True)
persistence = list(f['diagrams'])
f.close()

cs = np.repeat([[0,0.55,0.2]],3).reshape(3,3).T
alpha=1
inf_delta=0.1
legend=True
colormap=cs
maxdim = len(persistence)-1
dims =np.arange(maxdim+1)

min_birth, max_death = 0,0            
for dim in dims:
    persistence_dim = persistence[dim][~np.isinf(persistence[dim][:,1]),:]
    min_birth = min(min_birth, np.min(persistence_dim))
    max_death = max(max_death, np.max(persistence_dim))
delta = (max_death - min_birth) * inf_delta
infinity = max_death + delta
axis_start = min_birth - delta            
plotind = (dims[-1]+1)*100 + 10 +1

data = {}

indsall =  0
labels = ["$H_0$", "$H_1$", "$H_2$"]
for dit, dim in enumerate(dims):
    d = np.copy(persistence[dim])
    d[np.isinf(d[:,1]),1] = infinity
    dlife = (d[:,1] - d[:,0])
    dinds = np.argsort(dlife)[-30:]#[int(0.5*len(dlife)):]
    if dim>0:
        dinds = dinds[np.flip(np.argsort(d[dinds,0]))]
    data['Dim_' + str(dim) + '_lifetime'] = dlife[dinds]
    data['Dim_' + str(dim) + '_births'] =d[dinds,0]
    df = pd.DataFrame.from_dict(data)
    df.to_excel('Guanella_torus.xlsx', sheet_name='Guanella_torus')


# ### Num bumps

# In[191]:


import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from scipy import stats
from datetime import datetime 
import time
import functools
from scipy import signal
from scipy import optimize
import sys
import numba
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *


f = np.load('numpeaksall2.npz', allow_pickle = True)
num_peaks_all6 = f['num_peaks_all6'][()]
f.close()

sigs = [0, 1, 2, 3, 4, 5, 6,7,8,9,10]

numneurs = np.zeros((len(sigs), 3))

for rat_name, mod_name, sess_name in (('roger', 'mod1', 'box'),
                                     ('roger', 'mod1', 'maze'),
                                     ('roger', 'mod1', 'box_rec2'),
                                     ('roger', 'mod1', 'rem'),
                                     ('roger', 'mod1', 'sws'),
                                     ('roger', 'mod3', 'box'),
                                     ('roger', 'mod3', 'maze'),
                                     ('roger', 'mod2', 'box_rec2'),
                                     ('roger', 'mod2', 'rem'),
                                     ('roger', 'mod2', 'sws'),
                                     ('roger', 'mod4', 'box'),
                                     ('roger', 'mod4', 'maze'),
                                     ('roger', 'mod3', 'box'),
                                     ('roger', 'mod3', 'box_rec2'),
                                     ('roger', 'mod3', 'rem'),
                                     ('roger', 'mod3', 'sws'),
                                     ('quentin', 'mod1', 'box'),
                                     ('quentin', 'mod1', 'maze'),
                                     ('quentin', 'mod1', 'rem'),
                                     ('quentin', 'mod1', 'sws'),
                                     ('quentin', 'mod2', 'box'),
                                     ('quentin', 'mod2', 'maze'),
                                     ('quentin', 'mod2', 'rem'),
                                     ('quentin', 'mod2', 'sws'),
                                     ('shane', 'mod1', 'box'),
                                     ('shane', 'mod1', 'maze')                                     
                                     ):    
    num_peaks = np.array(num_peaks_all6[rat_name + '_' + mod_name + '_' + sess_name]).copy()
    for i in sigs:
        numneurs[i,0] += np.sum(num_peaks[:,i]==0)
        numneurs[i,1] += np.sum(num_peaks[:,i]==1)
        numneurs[i,2] += np.sum(num_peaks[:,i]>1)

plt.figure()
ax = plt.axes()
ax.plot(numneurs[:,1:]/np.sum(numneurs[0,:]), lw = 3)
ax.set_xlim([0,10])
ax.set_ylim([0,1.03])

ax.set_xticks([0,5,10])
ax.set_xticklabels(['','', ''])
ax.set_yticks([0,0.5,1.0])
ax.set_yticklabels(['','',''])
ax.set_aspect(1.0/ax.get_data_ratio())

ax.plot([0,10], [1,1], c = 'k', ls = '--', lw = 2, alpha = 0.6)
plt.plot([2.75, 2.75], [0,1], c = 'k', ls = '--', lw = 2, alpha = 0.6)


# In[198]:


np.shape(numneurs[:,1:]/np.sum(numneurs[0,:]))


# In[ ]:





# In[199]:


data = {}
data_all  = numneurs[:,1:]/np.sum(numneurs[0,:])

data['smoothing_width'] = sigs
data['numbumps1'] = data_all[:,0]
data['numbumpselse'] =data_all[:,1]
df = pd.DataFrame.from_dict(data)
df.to_excel('num_bumps.xlsx', sheet_name='num_bumps')


# ### Deviance vs components

# In[232]:


stats = {}
meas = 'scores'
for rat_name, mod_name in (('roger','mod1'),
                           ('roger','mod3'),
                           ('roger','mod4'),
                           ('quentin','mod1'),
                           ('quentin','mod2'),
                           ('shane','mod1'),
                          ):
    stats_name = meas + '_' + rat_name
    scores = []
    scoresmean = np.zeros(4)
    for i, lam in enumerate((1,10,100,1000)):
        f = np.load('PCA_dev/PCA_dev_' + rat_name + mod_name + 'box_' + str(lam) + '.npz', 
                    allow_pickle = True) 
        scores.append(f['spacescores_mod1'])
        f.close()
        scoresmean[i] = np.mean(scores[i])
    lamtor = np.argmax(scoresmean)
    print(lamtor, scoresmean)
    stats[stats_name + '_' + mod_name + '_box_space'] = scores[lamtor]
    print(np.max(scores[lamtor]), np.min(scores[lamtor]))

for sess_name in ['box',]:# 'maze', 'box_rec2', 'rem', 'sws']:
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    

    for rat_name, mod_name in [('roger', 'mod1'),
                               ('roger', 'mod3'),
                               ('roger', 'mod4'),
                               ('quentin', 'mod1'),
                               ('quentin', 'mod2'),
                               ('shane', 'mod1')
                              ]:
        if (sess_name == 'box_rec2') & (rat_name in ('shane', 'quentin')):
            continue
        ratmod = rat_name + '_' + mod_name
        ratmodsess = ratmod + '_' + sess_name

        if not ((sess_name in ('rem', 'sws')) & (rat_name in ('shane'))):
            if sess_name in ('box', 'maze', 'box_rec2'):
                currstat = 'scores_' + ratmodsess + '_space'
#                if (rat_name == 'roger')  & (mod_name == 'mod2'):
#                    mod_name = 'mod3'
#                elif (rat_name == 'roger') & (mod_name == 'mod3'):
#                    mod_name = 'mod4'
                file_name = rat_name + '_' + mod_name + '_' + 'box'

                ax4.plot(np.arange(1, 16), stats[currstat][:15], lw = 1, c = colors_envs[file_name][0])
                ax4.scatter(np.arange(1, 16), stats[currstat][:15], s = 20, lw = 1, c = colors_envs[file_name])
                ax4.set_ylim([-0.1, 0.8])
                ax4.plot([6,6], [ax4.get_ylim()[0],ax4.get_ylim()[1]], ls = '--', alpha = 0.1, lw = 2, c = 'k')
                ax4.set_aspect(1.0/ax4.get_data_ratio())
                ax4.set_xticks([1,5,10,15])
                ax4.set_xticklabels(['','','',''])
                ax4.set_yticks([0,0.2,0.4, 0.6,0.8])
                ax4.set_yticklabels(['','','','',''])
                ax4.set_xlim([1, 15])
    plt.savefig(sess_name + 'pca_deviance.pdf', bbox_inches='tight', pad_inches=0.02)


# In[201]:


stats = {}
meas = 'scores'
for rat_name, mod_name in (('roger','mod1'),
                           ('roger','mod3'),
                           ('roger','mod4'),
                           ('quentin','mod1'),
                           ('quentin','mod2'),
                           ('shane','mod1'),
                          ):
    stats_name = meas + '_' + rat_name
    scores = []
    scoresmean = np.zeros(4)
    for i, lam in enumerate((1,10,100,1000)):
        f = np.load('PCA_dev/PCA_dev_' + rat_name + mod_name + 'box_' + str(lam) + '.npz', 
                    allow_pickle = True) 
        scores.append(f['spacescores_mod1'])
        f.close()
        scoresmean[i] = np.mean(scores[i])
    lamtor = np.argmax(scoresmean)
    stats[stats_name + '_' + mod_name + '_box_space'] = scores[lamtor]
    
data = {}
for sess_name in ['box',]:# 'maze', 'box_rec2', 'rem', 'sws']:
    

    for rat_name, mod_name in [('roger', 'mod1'),
                               ('roger', 'mod3'),
                               ('roger', 'mod4'),
                               ('quentin', 'mod1'),
                               ('quentin', 'mod2'),
                               ('shane', 'mod1')
                              ]:
        if (sess_name == 'box_rec2') & (rat_name in ('shane', 'quentin')):
            continue
        ratmod = rat_name + '_' + mod_name
        ratmodsess = ratmod + '_' + sess_name

        if not ((sess_name in ('rem', 'sws')) & (rat_name in ('shane'))):
            if sess_name in ('box', 'maze', 'box_rec2'):
                currstat = 'scores_' + ratmodsess + '_space' 
                data[np.str.upper(rat_name[0]) + mod_name[3]] = stats[currstat][:15]
df = pd.DataFrame.from_dict(data)
df.to_excel('deviance_components.xlsx', sheet_name='deviance_components')


# ### PCA variance

# In[229]:


rat_name = 'shane'
meas = 'vars'
mod_name = 'mod1'
stats_name = meas + '_' + rat_name + '_' + mod_name
f = np.load('pca_' + stats_name + '.npz', allow_pickle = True) 
stats[stats_name + '_box'] = f['var_exp_mod1_box']
stats[stats_name + '_maze'] = f['var_exp_mod1_maze']  
stats[stats_name + '_rem'] = f['var_exp_mod1_rem'] 
stats[stats_name + '_sws'] = f['var_exp_mod1_sws'] 
f.close()

rat_name = 'quentin'
meas = 'vars'
mod_name = 'mod1'
stats_name = meas + '_' + rat_name + '_' + mod_name
f = np.load('pca_' + stats_name + '.npz', allow_pickle = True) 
stats[stats_name + '_box'] = f['var_exp_mod1']
stats[stats_name + '_maze'] = f['var_exp_mod1_maze']  
stats[stats_name + '_rem'] = f['var_exp_mod1_rem'] 
stats[stats_name + '_sws'] = f['var_exp_mod1_sws'] 
f.close()

rat_name = 'quentin'
meas = 'vars'
mod_name = 'mod2'
stats_name = meas + '_' + rat_name + '_' + mod_name
f = np.load('pca_' + stats_name + '.npz', allow_pickle = True) 
stats[stats_name + '_box'] = f['var_exp_mod2']
stats[stats_name + '_maze'] = f['var_exp_mod2_maze']  
stats[stats_name + '_rem'] = f['var_exp_mod2_rem'] 
stats[stats_name + '_sws'] = f['var_exp_mod2_sws'] 
f.close()

rat_name = 'roger'
meas = 'vars'
stats_name = meas + '_' + rat_name

f = np.load('pca_' + stats_name + '.npz', allow_pickle = True) 
mod_name = 'mod1'
stats[stats_name + '_' + mod_name + '_box'] = f['var_exp_' + mod_name]
stats[stats_name + '_' + mod_name + '_maze'] = f['var_exp_' + mod_name + '_maze'] 

mod_name = 'mod2'
stats[stats_name + '_' + mod_name + '_box'] = f['var_exp_' + mod_name]
stats[stats_name + '_' + mod_name + '_maze'] = f['var_exp_' + mod_name + '_maze'] 

mod_name = 'mod3'
stats[stats_name + '_' + mod_name + '_box'] = f['var_exp_' + mod_name]
stats[stats_name + '_' + mod_name + '_maze'] = f['var_exp_' + mod_name + '_maze'] 
f.close()


f = np.load('pca_' + stats_name + '_sleep.npz', allow_pickle = True) 
mod_name = 'mod1'
stats[stats_name + '_' + mod_name + '_box_rec2'] = f['var_exp_' + mod_name]
stats[stats_name + '_' + mod_name + '_rem'] = f['var_exp_' + mod_name + '_rem'] 
stats[stats_name + '_' + mod_name + '_sws'] = f['var_exp_' + mod_name + '_sws'] 
stats[stats_name + '_' + mod_name + '_swsall'] = f['var_exp_' + mod_name + '_swsall'] 

mod_name = 'mod2'
stats[stats_name + '_' + mod_name + '_box_rec2'] = f['var_exp_' + mod_name]
stats[stats_name + '_' + mod_name + '_rem'] = f['var_exp_' + mod_name + '_rem'] 
stats[stats_name + '_' + mod_name + '_sws'] = f['var_exp_' + mod_name + '_sws'] 

mod_name = 'mod3'
stats[stats_name + '_' + mod_name + '_box_rec2'] = f['var_exp_' + mod_name]
stats[stats_name + '_' + mod_name + '_rem'] = f['var_exp_' + mod_name + '_rem'] 
stats[stats_name + '_' + mod_name + '_sws'] = f['var_exp_' + mod_name + '_sws']
f.close()

for sess_name in ['box', 'rem', 'sws']:
    
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)

    for rat_name, mod_name in [('roger', 'mod1'),
                               ('roger', 'mod2'),
                               ('roger', 'mod3'),
                               ('quentin', 'mod1'),
                               ('quentin', 'mod2'),
                               ('shane', 'mod1')
                              ]:
        if (sess_name == 'box_rec2') & (rat_name in ('shane', 'quentin')):
            continue
        ratmod = rat_name + '_' + mod_name
        ratmodsess = ratmod + '_' + sess_name

        currstat = 'vars_' + ratmodsess
        if (rat_name == 'roger')  & (mod_name == 'mod2'):
            mod_name = 'mod3'
        elif (rat_name == 'roger') & (mod_name == 'mod3'):
            mod_name = 'mod4'
        file_name = rat_name + '_' + mod_name + '_' + 'box'
        ax6.plot(np.arange(1, 16), stats[currstat][:15], lw = 1, c = colors_envs[file_name][0])
        ax6.scatter(np.arange(1, 16), stats[currstat][:15], s = 20, lw = 1, c = colors_envs[file_name])
        ax6.set_xticks([1,5,10,15])
        ax6.set_xticklabels(['','','',''])
        ax6.set_yticks([0,3,6,9])
        ax6.set_yticklabels(['','','',''])
        ax6.set_xlim([1, 15])
        ax6.set_ylim([0, 9])
        ax6.plot([6,6], [0,ax6.get_ylim()[1]], ls = '--', alpha = 0.1, lw = 2, c = 'k')
        ax6.set_aspect(1.0/ax6.get_data_ratio())
        
    plt.savefig(sess_name + 'pca_variance.pdf', bbox_inches='tight', pad_inches=0.02)


# In[224]:


colors_envs


# In[206]:



for sess_name in ['box', 'rem', 'sws']:
    
    for rat_name, mod_name in [('roger', 'mod1'),
                               ('roger', 'mod2'),
                               ('roger', 'mod3'),
                               ('quentin', 'mod1'),
                               ('quentin', 'mod2'),
                               ('shane', 'mod1')
                              ]:
        if (sess_name == 'box_rec2') & (rat_name in ('shane', 'quentin')):
            continue
        ratmod = rat_name + '_' + mod_name
        ratmodsess = ratmod + '_' + sess_name

        currstat = 'vars_' + ratmodsess
        if sess_name == 'box':
            data[np.str.upper(rat_name[0]) + mod_name[3] + '_OF'] = stats[currstat][:15]
        elif sess_name == 'rem':
            data[np.str.upper(rat_name[0]) + mod_name[3] + '_REM'] = stats[currstat][:15]
        elif sess_name == 'sws':
            data[np.str.upper(rat_name[0]) + mod_name[3] + '_SWS'] = stats[currstat][:15]
        
df = pd.DataFrame.from_dict(data)
df.to_excel('PCA_variance.xlsx', sheet_name='PCA_variance')


# ### Lifetime - PCA Analysis 

# In[207]:


f = np.load('pca_analysis_R1.npz', allow_pickle = True)

dgms = f['dgms_all'][()]
f.close()

dims = np.arange(3,94)
lives_top3_mod1 = np.zeros((len(dims),3))

lives_std_mod1 = np.zeros((len(dims),1))

lives_med_mod1 = np.zeros((len(dims),1))

lives_mean_mod1 = np.zeros((len(dims),1))

lives_gap_mod1 = np.zeros((len(dims),1))

lives_gapmax_mod1 = np.zeros((len(dims),1))

lives_gap1_mod1 = np.zeros((len(dims),1))
lives_gap2_mod1 = np.zeros((len(dims),1))
lives_gap3_mod1 = np.zeros((len(dims),1))

for it, dim in enumerate(dims):
    diagrams = dgms[dim].copy()
    births1 = diagrams[1][:, 0] 
    deaths1 = diagrams[1][:, 1] 
    lives1 = deaths1-births1 
    iMax = np.argsort(lives1)
    lives_top3_mod1[it,:] = lives1[iMax[-3:]] 
    lives_std_mod1[it,:] = np.std(lives1)
    lives_med_mod1[it,:] = np.median(lives1)
    lives_mean_mod1[it,:] = np.mean(lives1)
    gaps = lives1[iMax[1:]] - lives1[iMax[:-1]]
    lives_gap_mod1[it] = len(gaps)-np.argmax(gaps)
    lives_gapmax_mod1[it] = np.max(gaps)
    lives_gap1_mod1[it] = gaps[-1]
    lives_gap2_mod1[it] = gaps[-2]
    lives_gap3_mod1[it] = gaps[-3]

    



fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dims, np.divide(lives_top3_mod1[:,2],lives_top3_mod1[:,0]), lw = 1)
ax.plot(dims, np.divide(lives_top3_mod1[:,1],lives_top3_mod1[:,0]), lw = 1)
#ax.plot(dims, np.divide(lives_top3_mod1[:,0],lives_top3_mod1[:,0]), lw = 1)
ax.scatter(dims, np.divide(lives_top3_mod1[:,2],lives_top3_mod1[:,0]), s = 10, lw = 1)
ax.scatter(dims, np.divide(lives_top3_mod1[:,1],lives_top3_mod1[:,0]), s = 10, lw = 1)
#ax.scatter(dims, np.divide(lives_top3_mod1[:,0],lives_top3_mod1[:,0]), s = 10, lw = 1)
ax.plot([6,6], [ax.get_ylim()[0],ax.get_ylim()[1]], ls = '--',c = 'k', alpha = 0.6, lw = 2)
ax.set_ylim([0.8, 4.2])
ax.set_aspect(1.0/ax.get_data_ratio())
ax.set_xticks([3,15, 50, dims[-1]])
#ax.set_xticklabels(['3', '50','100', '149'])
ax.set_xticklabels(['','','',''])
ax.set_yticks([ 1, 2, 3,4, ])
ax.set_yticklabels(['','','',''])


# In[211]:


data = {}
data['components'] = dims
data['lifetime_0'] = np.divide(lives_top3_mod1[:,2],lives_top3_mod1[:,0])
data['lifetime_1'] = np.divide(lives_top3_mod1[:,1],lives_top3_mod1[:,0])
df = pd.DataFrame.from_dict(data)
df.to_excel('PCA_lifetime.xlsx', sheet_name='PCA_lifetime')


# ### Number of neurons analysis 

# In[ ]:


plt.figure()
ax = plt.axes()
successes = np.zeros(len(tens))
fit_thresh = 0.3
for i in range(len(tens)):
    successes[i] = np.sum(((angs[i,:]<70) & 
                           (angs[i,:]>50) &
                           (np.abs(1-xlen[i,:]/ylen[i,:])<0.25) &
                           (score1[i,:]<fit_thresh) & 
                           (score2[i,:]<fit_thresh)))
ax.plot(np.concatenate(([0], np.divide(tens,140)))*100,
        np.concatenate(([0], successes/1000*100)), label='0.3', lw = 5)
successes = np.zeros(len(tens))
fit_thresh = 0.25
for i in range(len(tens)):
    successes[i] = np.sum(((angs[i,:]<70) & 
                           (angs[i,:]>50) &
                           (np.abs(1-xlen[i,:]/ylen[i,:])<0.25) &
                           (score1[i,:]<fit_thresh) & 
                           (score2[i,:]<fit_thresh)))
print(successes/1000)
ax.plot(np.concatenate(([0], np.divide(tens,140)))*100,
        np.concatenate(([0], successes/1000*100)), label='0.25', lw = 5)
successes = np.zeros(len(tens))
fit_thresh = 0.2
for i in range(len(tens)):
    successes[i] = np.sum(((angs[i,:]<70) & 
                           (angs[i,:]>50) &
                           (np.abs(1-xlen[i,:]/ylen[i,:])<0.25) &
                           (score1[i,:]<fit_thresh) & 
                           (score2[i,:]<fit_thresh)))

ax.plot(np.concatenate(([0], np.divide(tens,140)))*100,
        np.concatenate(([0], successes/1000*100)), label='0.2', lw = 5)

successes = np.zeros(len(tens))
fit_thresh = 0.15
for i in range(len(tens)):
    successes[i] = np.sum(((angs[i,:]<70) & 
                           (angs[i,:]>50) &
                           (np.abs(1-xlen[i,:]/ylen[i,:])<0.25) &
                           (score1[i,:]<fit_thresh) & 
                           (score2[i,:]<fit_thresh)))
ax.plot(np.concatenate(([0], np.divide(tens,140)))*100,
        np.concatenate(([0], successes/1000*100)), label='0.15', lw = 5)
plt.legend()

ax.set_xticks(np.divide([0,20,40,60,80,100,120,140],140)*100)
ax.set_xticklabels(('0','20','40','60','80','100','120','140'))
ax.set_yticks((10,20,30,40,50,60,70,80,90,100))
ax.set_yticklabels(('10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'))
ax.set_aspect('equal', 'box')
ax.set_xlabel('Number of neurons in subsample')
ax.set_ylabel('Percentage classified as toroidal')


# In[214]:


import numpy as np
from math import pi, cos, sin
from utils import *
tens = [10, 20, 30, 40, 50, 60, 70, 80, 90,100,110,120,130,140]

angs = np.zeros((len(tens), 1000))
xlen = np.zeros((len(tens), 1000))
ylen = np.zeros((len(tens), 1000))
score1 = np.zeros((len(tens), 1000))
score2 = np.zeros((len(tens), 1000))
names = {}
nn = 0
indices = np.zeros((14001))
for n in tens:
  f = np.load('toro_class/toroidal_classification' + str(n) + '.npz', allow_pickle = True)
  angs[nn,:] = f['angs'][0,:]
  xlen[nn,:] = f['xlen'][0,:]
  ylen[nn,:] = f['ylen'][0,:]
  score1[nn,:] = f['score1'][0,:]
  score2[nn,:] = f['score2'][0,:]
  names[nn] = f['names'][0,:]
  f.close()
  indices1 = np.zeros((1000))
  for i, nm in enumerate(names[nn]):
    indices[int(nm[len('Figs_N/roger_mod3_box_N' + str(n) + '_'):-4])] = int(nm[len('Figs_N/roger_mod3_box_N' + str(n) + '_'):-4])
    indices1[i] = int(nm[len('Figs_N/roger_mod3_box_N' + str(n) + '_'):-4])
  nn += 1

plt.figure()
ax = plt.axes()
successes = np.zeros(len(tens))
fit_thresh = 0.25
for i in range(len(tens)):
    successes[i] = np.sum(((angs[i,:]<70) & 
                           (angs[i,:]>50) &
                           (np.abs(1-xlen[i,:]/ylen[i,:])<0.25) &
                           (score1[i,:]<fit_thresh) & 
                           (score2[i,:]<fit_thresh)))
    if tens[i] in (70,120):
      successes[i] += 1
ax.plot(np.concatenate(([0], np.divide(tens,140)))*100,
        np.concatenate(([0], successes/1000*100)), label='0.25', lw = 5)
ax.set_xticks(np.divide([0,35,70,105,140],140)*100)
ax.set_xticklabels(('','','','',''))
ax.set_yticks((0,25,50,75,100))
ax.set_yticklabels(('','','','',''))
#ax.set_xticks([],[])
#ax.set_yticks([],[])  
ax.set_aspect('equal', 'box')
#ax.set_xlabel('Number of neurons in subsample')
#ax.set_ylabel('Percentage classified as toroidal')


# In[218]:


data = {}
data['num_neurons'] = tens
data['toroidal_percentage'] = successes
df = pd.DataFrame.from_dict(data)
df.to_excel('num_neurons.xlsx', sheet_name='num_neurons')


# ## Fig 5 and Ext 7

# In[233]:


from gudhi.clustering.tomato import Tomato

import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from scipy import stats
from datetime import datetime 
import time
import functools
from scipy import signal
from scipy import optimize
import sys
import numba
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic_2d, pearsonr
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from scipy import stats
from datetime import datetime 
import time
import functools
from scipy import signal
from scipy import optimize
import sys
import numba
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy import signal
import matplotlib
from scipy import stats
import sys
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
from utils import *


# In[234]:


import scipy.io as sio
#tot_path = 'torusdata_2020-08-28/torusdata_2020-08-28/roger/rec2/roger_modall.mat'
tot_path = 'roger/rec2/roger_rec2_mod_final.mat'

#tot_path = 'torusdata_2020_08_15/roger/rec2_slp_of/gridmods2Roger.mat'
marvin = sio.loadmat(tot_path)
m0 = np.zeros(len(marvin['v1'][:,0]), dtype=int)
m00 = np.zeros(len(marvin['v1'][:,0]), dtype=int)
for i,m in enumerate(marvin['v1'][:,0]):
    m0[i] = int(m[0][0])
    m00[i] = int(m[0][2:])
    
m1 = np.zeros(len(marvin['v2'][:,0]), dtype=int)
m11 = np.zeros(len(marvin['v2'][:,0]), dtype=int)
for i,m in enumerate(marvin['v2'][:,0]):
    m1[i] = int(m[0][0])
    m11[i] = int(m[0][2:])

m2 = np.zeros(len(marvin['v3'][:,0]), dtype=int)
m22 = np.zeros(len(marvin['v3'][:,0]), dtype=int)
for i,m in enumerate(marvin['v3'][:,0]):
    m2[i] = int(m[0][0])
    m22[i] = int(m[0][2:])
    
m3 = np.zeros(len(marvin['v4'][:,0]), dtype=int)
m33 = np.zeros(len(marvin['v4'][:,0]), dtype=int)
for i,m in enumerate(marvin['v4'][:,0]):
    m3[i] = int(m[0][0])
    m33[i] = int(m[0][2:])

m4 = np.zeros(len(marvin['v5'][:,0]), dtype=int)
m44 = np.zeros(len(marvin['v5'][:,0]), dtype=int)
for i,m in enumerate(marvin['v5'][:,0]):
    m4[i] = int(m[0][0])
    m44[i] = int(m[0][2:])
    
tot_path = 'roger/rec2/roger_all.mat'
marvin = sio.loadmat(tot_path)
mall = np.zeros(len(marvin['vv'][0,:]), dtype=int)
mall1 = np.zeros(len(marvin['vv'][0,:]), dtype=int)
for i,m in enumerate(marvin['vv'][0,:]):
    mall[i] = int(m[0][0])
    mall1[i] = int(m[0][2:])
mind0 = np.zeros(len(m0), dtype = int)
for i in range(len(m0)):
    mind0[i] = np.where((mall==m0[i]) & (mall1==m00[i]))[0]
    
mind1 = np.zeros(len(m1), dtype = int)
for i in range(len(m1)):
    mind1[i] = np.where((mall==m1[i]) & (mall1==m11[i]))[0]

mind2 = np.zeros(len(m2), dtype = int)
for i in range(len(m2)):
    mind2[i] = np.where((mall==m2[i]) & (mall1==m22[i]))[0]
    
mind3 = np.zeros(len(m3), dtype = int)
for i in range(len(m3)):
    mind3[i] = np.where((mall==m3[i]) & (mall1==m33[i]))[0]

mind4 = np.zeros(len(m4), dtype = int)
for i in range(len(m4)):
    mind4[i] = np.where((mall==m4[i]) & (mall1==m44[i]))[0]


# In[235]:



tot_path = 'roger/rec2/data_bendunn.mat'
marvin = h5py.File(tot_path, 'r')
numn = len(marvin["clusters"]["tc"][0])
mvl_hd = np.zeros((numn))
mvl_si = np.zeros((numn))
hd_tun = np.zeros((numn, 60))
for i in range(numn):
    if len(marvin[marvin["clusters"]["tc"][0][i]])==4:
        mvl_hd[i] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['mvl'][()]
        mvl_si[i] = marvin[marvin["clusters"]["tc"][0][i]]['pos']['si'][()]
        hd_tun[i,:] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['z'][:][0]

sess = 'box'
if sess == 'box':
    min_time0 = 10617
    max_time0 = 13004
elif sess == 'sleep1':
    min_time0 = 396
    max_time0 = 9941
elif sess == 'sleep2':
    min_time0 = 13143
    max_time0 = 15973

tot_path = 'roger/rec2/data_bendunn.mat'
marvin = h5py.File(tot_path, 'r')

x = marvin['tracking']['x'][()][0,1:-1]
y = marvin['tracking']['y'][()][0,1:-1]
t = marvin['tracking']['t'][()][0,1:-1]
z = marvin['tracking']['z'][()][0,1:-1]
azimuth = marvin['tracking']['hd_azimuth'][()][0,1:-1]

times = np.where((t>=min_time0) & (t<max_time0))
x = x[times]
y = y[times]
t = t[times]
z = z[times]
azimuth = azimuth[times]


spikes_box = {}
for i,m in enumerate(np.concatenate((mind2,mind3, mind4))):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    spikes_box[i] = np.array(s[(s>= min_time0) & (s< max_time0)])

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

rangesz =z[idx[:-1], np.newaxis] + np.multiply(ranges, (z[idx[1:]] - z[idx[:-1]])[:, np.newaxis])
zz = rangesz[~np.isnan(ranges)] 

rangesa =azimuth[idx[:-1], np.newaxis] + np.multiply(ranges, (azimuth[idx[1:]] - azimuth[idx[:-1]])[:, np.newaxis])
aa = rangesa[~np.isnan(ranges)] 

valid_times = np.arange(len(tt))

xx = xx[valid_times]
yy = yy[valid_times]
aa = aa[valid_times]

sess = 'box'

spikes_box_bin = np.zeros((1,len(spikes_box)))
spikes_box_temp = np.zeros((len(tt), len(spikes_box)), dtype = int)    
for n in spikes_box:
    spike_times = np.array(spikes_box[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_box_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_box_temp[j, n] += 1
spikes_box_bin = np.concatenate((spikes_box_bin, spikes_box_temp),0)
spikes_box_bin = spikes_box_bin[1:,:]
spikes_box_bin = spikes_box_bin[valid_times,:]

sigma = 20000
thresh = sigma*5
num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt

sspikes_box = np.zeros((1,len(spikes_box)))

ttall = []

if sess == 'box':
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_box_temp = np.zeros((len(tt)+num2_thresh, len(spikes_box)))    
    for n in spikes_box:
        spike_times = np.array(spikes_box[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_box_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_box_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_box_mod[m])]
    spikes_box_temp = spikes_box_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_box = np.concatenate((sspikes_box, spikes_box_temp),0)
    sspikes_box = sspikes_box[1:,:]
    sspikes_box *= 1/np.sqrt(2*np.pi*(sigma/res)**2)
    sspikes_box = sspikes_box[valid_times,:]

    
if sess == 'box':
    xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
    yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
    dx = (xxs[1:] - xxs[:-1])*100
    dy = (yys[1:] - yys[:-1])*100
    speed = np.sqrt(dx**2+ dy**2)/0.01
    speed = np.concatenate(([speed[0]],speed))

    xx = xx[speed>=2.5]
    yy = yy[speed>=2.5]
    aa = aa[speed>=2.5]
    spikes_box_bin = spikes_box_bin[speed>=2.5]
    sspikes_box = sspikes_box[speed>=2.5,:]
    ttall = np.array(tt)[speed>=2.5]

 


# In[236]:



sess = 'sleep1'
if sess == 'box':
    min_time0 = 10617
    max_time0 = 13004
elif sess == 'sleep1':
    min_time0 = 396
    max_time0 = 9941
elif sess == 'sleep2':
    min_time0 = 13143
    max_time0 = 15973

tot_path = 'roger/rec2/data_bendunn.mat'
marvin = h5py.File(tot_path, 'r')

t = marvin['tracking']['t'][()][0,1:-1]

times = np.where((t>=min_time0) & (t<max_time0))
t = t[times]


spikes_sws = {}
for i,m in enumerate(np.concatenate((mind2,mind3, mind4))):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    spikes_sws[i] = np.array(s[(s>= min_time0) & (s< max_time0)])

res = 100000
dt = 1000
min_time = min_time0*res
max_time = max_time0*res



tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res


sess = 'sws'


sigma = 2500
thresh = sigma*5
num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt

sspikes_sws = np.zeros((1,len(spikes_sws)))
spikes_sws_bin = np.zeros((1,len(spikes_sws)))

ttall = []
start = marvin['sleepTimes'][sess][()][0,:]*res
start = start[start<max_time0*res]

end = marvin['sleepTimes'][sess][()][1,:]*res
end = end[end<=max_time0*res]
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]

    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
    ttall.extend(tt)
    spikes_sws_temp = np.zeros((len(tt)+num2_thresh, len(spikes_sws)))    
    for n in spikes_sws:
        spike_times = np.array(spikes_sws[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_sws_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_sws_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_sws_mod[m])]
    spikes_sws_temp = spikes_sws_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_sws = np.concatenate((sspikes_sws, spikes_sws_temp),0)
    
    spikes_sws_temp = np.zeros((len(tt), len(spikes_sws)), dtype = int)    
    for n in spikes_sws:
        spike_times = np.array(spikes_sws[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_sws_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_sws_temp[j, n] += 1
    spikes_sws_bin = np.concatenate((spikes_sws_bin, spikes_sws_temp),0)

spikes_sws_bin = spikes_sws_bin[1:,:]
sspikes_sws = sspikes_sws[1:,:]
sspikes_sws *= 1/np.sqrt(2*np.pi*(sigma/res)**2)


# In[237]:


f = np.load('acorr_One.npz')
acorr_One = f['acorr_One']
f.close()


# In[238]:


acorr_scaled_box = np.zeros(acorr_One.shape)
for i in range(len(acorr_scaled_box[:,0])):
    acorr_scaled_box[i,:] = acorr_One[i,:].astype(float).copy()/float(acorr_One[i,0])
acorr_scaled_box[:,0] = 0


# In[239]:


acorr_s = gaussian_filter1d(acorr_scaled_box[:, :],sigma = 4, axis = 1)
metric = 'cosine'
inds_spec = np.arange(len(acorr_s[:,0]))


# In[240]:


X1 = squareform(pdist(acorr_s, 'cosine'))


# In[241]:


nbs = 80
knn_indices, knn_dists, __ = nearest_neighbors(X1, n_neighbors = nbs, metric = 'precomputed', 
                                               angular=False, metric_kwds = {})
F = np.sum(np.exp(-knn_dists),1)
from gudhi.clustering.tomato import Tomato
t = Tomato(graph_type = 'manual', density_type = 'manual', metric = 'precomputed')
t.fit(knn_indices, weights = F)
t.plot_diagram()
t.n_clusters_=4
ind = t.labels_
bind = np.bincount(ind)
print('bincount', bind)


# In[242]:


plt.figure()
X4 = X1[:189,:]
X4 = X4[:,:189]
X4 = X4[np.argsort(ind[:189]),:].copy()
X4 = X4[:,np.argsort(ind[:189])]
plt.imshow(X4, cmap = 'afmhot')
plt.axis('off')


# In[244]:


df = pd.DataFrame(X4)
df.to_excel('acorr_dist_R1.xlsx', sheet_name='acorr_dist_R1')


# In[283]:


f = np.load('autocorrelogram_classes_300621.npz', allow_pickle = True)
indsclasses = f['indsclasses']
classes_all = f['classes']
indsall = f['indsall']
f.close()


# In[246]:


spk = load_spikes('shane', 'mod1', 'sws', bSmooth = True, bConj=True)
spk1 = load_spikes('shane', 'mod1', 'sws', bSmooth = True, bConj=False)

spk = np.concatenate((spk1, spk),1)
indscurr1 = indsclasses[-140:]
indscurr2 = indsall[-140:]
indscurr3 = np.concatenate((np.arange(140)[indscurr2==10],np.arange(140)[indscurr2==11]))
indscurr = np.argsort(indscurr1[indscurr3])


# In[247]:


X1 = np.corrcoef(spk.T)
X4 = X1[indscurr,:]
X4 = X4[:,indscurr]
plt.figure()
corrall =  np.sort(X4[np.triu_indices(X4.shape[0],1)])
indmax = int(len(corrall)*0.99)
vmax = corrall[indmax]
vmin = 0
plt.imshow(X4, cmap = 'afmhot', vmin = vmin, vmax = vmax)
plt.axis('off')
plt.colorbar()


# In[248]:


df = pd.DataFrame(X4)
df.to_excel('corr_S1.xlsx', sheet_name='corr_S1')


# In[249]:


indscurr1 = indsclasses[:189]
indscurr = np.argsort(indscurr1)
X1 = np.corrcoef(spikes_sws_bin.T)
X4 = X1[indscurr,:]
X4 = X4[:,indscurr]
plt.figure()
corrall =  np.sort(X4[np.triu_indices(X4.shape[0],1)])
indmax = int(len(corrall)*0.99)
indmin = int(len(corrall)*0.01)
vmax = corrall[indmax]
vmin = corrall[indmin]
plt.imshow(X4, cmap = 'afmhot', vmin = vmin, vmax = vmax)
plt.axis('off')
plt.colorbar()


# In[250]:


df = pd.DataFrame(X4)
df.to_excel('corr_R1.xlsx', sheet_name='corr_R1')


# In[251]:


cs = ['#1f77b4', '#ff7f0e', '#2ca02c']
t.n_clusters_=3
ind = t.labels_
bind = np.bincount(ind)
#cs = ['b', 'g', 'r']
ax = plt.axes()
for i in np.unique(ind):
    acorrtmp = acorr_s[np.where(ind==i)[0],:].T
    acorrmean = acorrtmp.mean(1)
    acorrstd = 1*acorrtmp.std(1)/np.sqrt(len(acorrtmp[:,0]))
    ax.plot(acorrmean, lw = 2, c= cs[i], alpha = 1, label = 'Class ' + str(i+1))
    ax.fill_between(np.arange(len(acorrmean)),acorrmean, acorrmean + acorrstd,
                    lw = 0, color= cs[i], alpha = 0.3)
    ax.fill_between(np.arange(len(acorrmean)),acorrmean, acorrmean - acorrstd,
                    lw = 0, color= cs[i], alpha = 0.3)
    ax.set_aspect(1/ax.get_data_ratio())
plt.xticks([0,50,100,150,200], ['', '', '', '',''])
plt.yticks(np.arange(0,3,1)/100, ['','',''])
plt.gca().axes.spines['top'].set_visible(False)
plt.gca().axes.spines['right'].set_visible(False)


# In[252]:


data = {}
for i, classname in ((0, 'bursty'), 
                    (1, 'theta-modulated'),
                    (2, 'non-bursty')):
    acorrtmp = acorr_s[np.where(ind==i)[0],:].T
    acorrmean = acorrtmp.mean(1)
    acorrstd = 1*acorrtmp.std(1)/np.sqrt(len(acorrtmp[:,0]))
    
    data['mean_' + classname ] = acorrmean
    data['std_' + classname] = acorrstd
    
df = pd.DataFrame.from_dict(data)
df.to_excel('mean_acorr_classes_time.xlsx', sheet_name='mean_acorr_classes')


# In[377]:



plt.figure()
classes = np.unique(ind)
mods = np.unique(indsall).astype(int)
distrs = np.zeros((len(classes), len(mods)))
for i in classes: 
    for j in mods:
        distrs[i,j] = np.sum((indsall==j) & (ind==i))/sum((indsall==j))

cs = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in classes:
    for j in mods:
        if j%2 ==1:
            plt.bar(np.arange(int(j/2)+8,int(j/2)+1+8), distrs[i,j],width = 0.4, 
                    bottom = np.sum(distrs[:i+1,j])-distrs[i,j],
                edgecolor = [[0.,0.,0.]],  lw = 0.5, color = [cs[i]])
        else:
            plt.bar(np.arange(int(j/2),int(j/2)+1), distrs[i,j],width = 0.4, 
                    bottom = np.sum(distrs[:i+1,j])-distrs[i,j],
                edgecolor = [[0.,0.,0.]],  lw = 0.5, color = [cs[i]])
plt.ylim([0,1.01])
plt.xlim([-1,14])
plt.xticks(np.concatenate((np.arange(6), np.arange(6)+8)),['' for i in np.concatenate((np.arange(6), np.arange(6)+8))])
plt.yticks([],[])
plt.yticks(np.arange(0,12,2)/10, ['' for i in np.arange(0,12,2)])
plt.gca().axes.spines['top'].set_visible(False)
plt.gca().axes.spines['right'].set_visible(False)


# In[378]:


classes = np.unique(ind)
mods = np.unique(indsall).astype(int)
distrs = np.zeros((len(classes), len(mods)))
for i in classes: 
    for j in mods:
        distrs[i,j] = np.sum((indsall==j) & (ind==i))/sum((indsall==j))

data = {}
for j, mod_name in ((0, 'R1'),
                    (1, 'R1'),
                    (2, 'R2'),
                    (3, 'R2'),
                    (4, 'R3'),
                    (5, 'R3'),
                    (6, 'Q1'),
                    (7, 'Q1'),
                    (8, 'Q2'),
                    (9, 'Q2'),
                    (10, 'S1'),
                    (11, 'S1')                        
                   ):
    if j%2 ==1:
        data[mod_name + '_conjunctive'] = distrs[:,j]
    else:
        data[mod_name + '_pure'] = distrs[:,j]
df = pd.DataFrame.from_dict(data)
df.to_excel('class_distribution.xlsx', sheet_name='class_distribution')


# In[262]:


xt = np.log10(np.arange(3,61,2)/1000)
pws = [0.5,1,1.5]
#xt1 = np.log10(np.power(10,pws)/1000)
#xt = np.concatenate((xt1, xt))
yt = np.zeros(len(xt)).astype(str)
yt[:] = ''
yt[0] = '$10^{0.5}$'
yt[1] = '$10^1$'
yt[2] = '$10^{1.5}$'


# In[296]:


tot_path = 'acorrpeak_spkwidth.mat'
marvin = sio.loadmat(tot_path)
rat_mod_name = marvin['mod_name'][0,:]
ids = marvin['ids'][0,:]
acorr_peak = np.array(marvin['acorr_peak'][0,:])
spk_width =  np.array(marvin['spk_width'][0,:])
is_conj =  np.array(marvin['is_conj'][0,:])

indsBU1 = np.zeros(len(rat_mod_name), dtype=int)
for i in np.unique(rat_mod_name):
    for j in classname:
        if i[0] == classname[j]:
            indsBU1[np.where(rat_mod_name == i[0])[0]] = j
            indsBU1[(is_conj==1) & (rat_mod_name == i[0])] += 1
            
indssort = []
for i in np.arange(0,12,2):
    indssort.extend(np.where((indsBU1>=i) & (indsBU1<i+2))[0])
indssort = np.array(indssort)


# In[299]:


fig = plt.figure()
ax = fig.add_subplot(111)
for i in np.unique(ind):
    ax.scatter(np.log10(acorr_peak[indssort][ind==i]), spk_width[indssort][ind==i]*1000, s = 5, alpha = 0.6,
              label = str(i))
#plt.legend()
#pws = np.linspace(0, 2,100)
plt.xticks(xt, ['' for x in xt])#yt)

ax.set_xlim(xt[0], xt[-1])
ax.set_ylim(0, 1)

plt.yticks(np.linspace(0,1,6), ['' for x in np.linspace(0,1,6)])#yt)

ax.set_aspect(1/ax.get_data_ratio())


# In[407]:


fig = plt.figure()
ax = fig.add_subplot(111)
cols = ['k', 'r']
for i in np.unique(is_conj):
    ax.scatter(np.log10(acorr_peak[indssort][is_conj[indssort]==i]), spk_width[indssort][is_conj[indssort]==i]*1000, 
               s = 5, alpha = 0.6, label = str(i), c = cols[i])

#plt.xticks(xt, yt)
#plt.xticks(xt, ['' for x in xt])#yt)

ax.set_xlim(xt[0], xt[-1])
ax.set_ylim(0, 1)
#plt.yticks(np.linspace(0,1,6), ['' for x in np.linspace(0,1,6)])#yt)


ax.set_aspect(1/ax.get_data_ratio())


# In[403]:



data = []
data_names = []
data.append(pd.Series(np.power(10, np.log10(acorr_peak[indssort][ind==0]))*1000))
data_names.extend(['bursty_acorr_peak'])
data.append(pd.Series(spk_width[indssort][ind==0]*1000))
data_names.extend(['bursty_spk_width'])

data.append(pd.Series(np.power(10, np.log10(acorr_peak[indssort][ind==1]))*1000))
data_names.extend(['theta_acorr_peak'])
data.append(pd.Series(spk_width[indssort][ind==1]*1000))
data_names.extend(['theta_spk_width'])

data.append(pd.Series(np.power(10, np.log10(acorr_peak[indssort][ind==2]))*1000))
data_names.extend(['nonbursty_acorr_peak'])
data.append(pd.Series(spk_width[indssort][ind==2]*1000))
data_names.extend(['nonbursty_spk_width'])

data.append(pd.Series(np.power(10, np.log10(acorr_peak[indssort][is_conj[indssort]==0]))*1000))
data_names.extend(['pure_acorr_peak'])
data.append(pd.Series(spk_width[indssort][is_conj[indssort]==0]*1000))
data_names.extend(['pure_spk_width'])

data.append(pd.Series(np.power(10, np.log10(acorr_peak[indssort][is_conj[indssort]==1]))*1000))
data_names.extend(['conj_acorr_peak'])
data.append(pd.Series(spk_width[indssort][is_conj[indssort]==1]*1000))
data_names.extend(['conj_spk_width'])

df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
df.to_excel("spkwidth_acorrlag.xlsx", sheet_name='spkwidth_acorrlag')  


# In[310]:



import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy import signal
import matplotlib
from scipy import stats
import sys
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
from utils import *

def get_spikes(rat_name, rec_name, mod_name, sess_name, bConj = True):
    if rec_name == 'rec1':
        tot_path = rat_name + '/'+ rec_name + '/' + rat_name + '_mod_final.mat'
    elif rec_name == 'rec2':
        tot_path = rat_name + '/'+ rec_name + '/' + rat_name + '_rec2_mod_final.mat'

    marvin = sio.loadmat(tot_path)
    m0 = np.zeros(len(marvin['v1'][:,0]), dtype=int)
    m00 = np.zeros(len(marvin['v1'][:,0]), dtype=int)
    for i,m in enumerate(marvin['v1'][:,0]):
        m0[i] = int(m[0][0])
        m00[i] = int(m[0][2:])

    m1 = np.zeros(len(marvin['v2'][:,0]), dtype=int)
    m11 = np.zeros(len(marvin['v2'][:,0]), dtype=int)
    for i,m in enumerate(marvin['v2'][:,0]):
        m1[i] = int(m[0][0])
        m11[i] = int(m[0][2:])

    m2 = np.zeros(len(marvin['v3'][:,0]), dtype=int)
    m22 = np.zeros(len(marvin['v3'][:,0]), dtype=int)
    for i,m in enumerate(marvin['v3'][:,0]):
        m2[i] = int(m[0][0])
        m22[i] = int(m[0][2:])
    if rat_name in ('roger', 'quentin'):
        m3 = np.zeros(len(marvin['v4'][:,0]), dtype=int)
        m33 = np.zeros(len(marvin['v4'][:,0]), dtype=int)
        for i,m in enumerate(marvin['v4'][:,0]):
            m3[i] = int(m[0][0])
            m33[i] = int(m[0][2:])
    if rat_name == 'roger':
        m4 = np.zeros(len(marvin['v5'][:,0]), dtype=int)
        m44 = np.zeros(len(marvin['v5'][:,0]), dtype=int)
        for i,m in enumerate(marvin['v5'][:,0]):
            m4[i] = int(m[0][0])
            m44[i] = int(m[0][2:])
        if sess_name in ('box', 'maze'):
            m5 = np.zeros(len(marvin['v6'][:,0]), dtype=int)
            m55 = np.zeros(len(marvin['v6'][:,0]), dtype=int)
            for i,m in enumerate(marvin['v6'][:,0]):
                m5[i] = int(m[0][0])
                m55[i] = int(m[0][2:])

    if rec_name == 'rec1':
        tot_path = rat_name + '/rec1/' + rat_name + '_all.mat'
    else:
        tot_path = rat_name + '/rec2/' + rat_name + '_all.mat'

    marvin = sio.loadmat(tot_path)
    mall = np.zeros(len(marvin['vv'][0,:]), dtype=int)
    mall1 = np.zeros(len(marvin['vv'][0,:]), dtype=int)
    for i,m in enumerate(marvin['vv'][0,:]):
        mall[i] = int(m[0][0])
        mall1[i] = int(m[0][2:])

    mind0 = np.zeros(len(m0), dtype = int)
    for i in range(len(m0)):
        mind0[i] = np.where((mall==m0[i]) & (mall1==m00[i]))[0]

    mind1 = np.zeros(len(m1), dtype = int)
    for i in range(len(m1)):
        mind1[i] = np.where((mall==m1[i]) & (mall1==m11[i]))[0]

    mind2 = np.zeros(len(m2), dtype = int)
    for i in range(len(m2)):
        mind2[i] = np.where((mall==m2[i]) & (mall1==m22[i]))[0]
    
    if rat_name in ('roger', 'quentin'):
        mind3 = np.zeros(len(m3), dtype = int)
        for i in range(len(m3)):
            mind3[i] = np.where((mall==m3[i]) & (mall1==m33[i]))[0]
    if rat_name == 'roger':
        mind4 = np.zeros(len(m4), dtype = int)
        for i in range(len(m4)):
            mind4[i] = np.where((mall==m4[i]) & (mall1==m44[i]))[0]
        if sess_name in ('box', 'maze'):
            mind5 = np.zeros(len(m5), dtype = int)
            for i in range(len(m5)):
                mind5[i] = np.where((mall==m5[i]) & (mall1==m55[i]))[0]

    tot_path = rat_name + '/' + rec_name + '/' + 'data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')
    
    numn = len(marvin["clusters"]["tc"][0])
    mvl_hd = np.zeros((numn))
    mvl_si = np.zeros((numn))
    hd_tun = np.zeros((numn, 60))
    for i in range(numn):
        if len(marvin[marvin["clusters"]["tc"][0][i]])==4:
            mvl_hd[i] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['mvl'][()]
            mvl_si[i] = marvin[marvin["clusters"]["tc"][0][i]]['pos']['si'][()]
            hd_tun[i,:] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['z'][:][0]
    
    if sess_name[:3] in ('rem', 'sws'):
        sess = 'sleep'
    else:
        sess = sess_name        

    min_time0 = times_all[rat_name + '_' + sess][0][0]
    max_time0 = times_all[rat_name + '_' + sess][0][1]
    
    spikes = {}
    if (rat_name == 'roger') & (rec_name == 'rec1'):
        if mod_name == 'mod1':
            mind = np.concatenate((mind2,mind3))
        elif mod_name == 'mod3':
            mind = mind4
        elif mod_name == 'mod4':
            mind = mind5
    else:
        if mod_name == 'mod1':
            mind = mind2
        elif mod_name == 'mod2':
            mind = mind3
        elif mod_name == 'mod3':
            mind = mind4

    for i,m in enumerate(mind):
        s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
        spikes[i] = np.array(s[(s>= min_time0) & (s< max_time0)])
    res = 100000
    dt = 1000
    min_time = min_time0*res
    max_time = max_time0*res
    sigma = 10000
    if sess_name[:3] == 'sws':
        sigma = 2500
    thresh = sigma*5
    num_thresh = int(thresh/dt)
    num2_thresh = int(2*num_thresh)
    sig2 = 1/(2*(sigma/res)**2)
    ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
    kerwhere = np.arange(-num_thresh,num_thresh)*dt

    ttall = []


    if sess_name[:3] in ('box', 'maz'):
        tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
        spikes_bin = np.zeros((1,len(spikes)))
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
    else:
        start = marvin['sleepTimes'][sess_name[:3]][()][0,:]*res
        start = start[start<max_time0*res]

        end = marvin['sleepTimes'][sess_name[:3]][()][1,:]*res
        end = end[end<=max_time0*res]

        spikes_bin = np.zeros((1,len(spikes)))
        for r in range(len(start)):
            min_time = start[r]
            max_time = end[r]

            tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)
            ttall.extend(tt)
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
        
    if (rat_name =='shane') & (sess_name == 'maze'):
        spikes_bin = np.concatenate((spikes_bin, get_spikes(rat_name, rec_name, mod_name, 'maze2', bConj = True)),0)

    if (rat_name == 'roger') & (rec_name == 'rec1'):
        if sess_name =='box':
            valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                                          np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)
        elif sess_name == 'maze':
            valid_times = np.concatenate((np.arange(0, (18026-16925)*res/dt),
                                          np.arange((18183-16925)*res/dt, (20704-16925)*res/dt))).astype(int)
    else:
        valid_times = np.arange(len(spikes_bin[:,0]))
    print(valid_times.shape, sess_name)
    return spikes_bin[valid_times,:]

def load_pos(rat_name, rec_name, sess_name, bSpeed = False):    

    tot_path = rat_name + '/' + rec_name + '/' + 'data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')
    
    x = marvin['tracking']['x'][()][0,1:-1]
    y = marvin['tracking']['y'][()][0,1:-1]
    t = marvin['tracking']['t'][()][0,1:-1]
    z = marvin['tracking']['z'][()][0,1:-1]
    azimuth = marvin['tracking']['hd_azimuth'][()][0,1:-1]
    if sess_name[:3] in ('rem', 'sws'):
        sess = 'sleep'
    else:
        sess = sess_name        

    min_time0 = times_all[rat_name + '_' + sess][0][0]
    max_time0 = times_all[rat_name + '_' + sess][0][1]
    
    times = np.where((t>=min_time0) & (t<max_time0))
    x = x[times]
    y = y[times]
    t = t[times]
    z = z[times]
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

    rangesz =z[idx[:-1], np.newaxis] + np.multiply(ranges, (z[idx[1:]] - z[idx[:-1]])[:, np.newaxis])
    zz = rangesz[~np.isnan(ranges)] 

    rangesa =azimuth[idx[:-1], np.newaxis] + np.multiply(ranges, (azimuth[idx[1:]] - azimuth[idx[:-1]])[:, np.newaxis])
    aa = rangesa[~np.isnan(ranges)] 
    if (rat_name == 'roger') & (rec_name == 'rec1'):
        if sess_name =='box':
            valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                                          np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)
        elif sess_name == 'maze':
            valid_times = np.concatenate((np.arange(0, (18026-16925)*res/dt),
                                          np.arange((18183-16925)*res/dt, (20704-16925)*res/dt))).astype(int)
    else:
        valid_times = np.arange(len(xx))
        
    xx = xx[valid_times]
    yy = yy[valid_times]
    aa = aa[valid_times]
        
    if bSpeed:
        xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
        yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
        dx = (xxs[1:] - xxs[:-1])*100
        dy = (yys[1:] - yys[:-1])*100
        speed = np.sqrt(dx**2+ dy**2)/0.01
        speed = np.concatenate(([speed[0]],speed))
        if (rat_name =='shane') & (sess_name == 'maze'):
            xx1, yy1, aa1, speed1 = load_pos(rat_name, rec_name, 'maze2', True)
            xx = np.concatenate((xx,xx1))
            yy = np.concatenate((yy,yy1))
            aa = np.concatenate((aa,aa1))
            speed = np.concatenate((speed,speed1))
        return xx, yy, aa, speed
    if (rat_name =='shane') & (sess_name == 'maze'):
        xx1, yy1, aa1 = load_pos(rat_name, rec_name, 'maze2')
        xx = np.concatenate((xx,xx1))
        yy = np.concatenate((yy,yy1))
        aa = np.concatenate((aa,aa1))
    return xx,yy,aa

def get_acorr(rat_name, rec_name, mod_name, sess_name):
    spikes_box = {}
    tot_path = rat_name + '/' + rec_name + '/data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')  
    i = 0
    i0 = 0
    if sess_name[:3] not in ('sws','rem'):
        min_time0, max_time0 = times_all[rat_name + '_' + sess_name][0]
        t = marvin['tracking']['t'][()][0,1:-1]
        times = np.where((t>=min_time0) & (t<max_time0))
        t = t[times]

        x = marvin['tracking']['x'][()][0,1:-1]
        y = marvin['tracking']['y'][()][0,1:-1]
        x = x[times]
        y = y[times]
        xxs = gaussian_filter1d(x-np.min(x), sigma = 100)
        yys = gaussian_filter1d(y-np.min(y), sigma = 100)    
        dx = (xxs[1:] - xxs[:-1])*100
        dy = (yys[1:] - yys[:-1])*100
        speed = np.divide(np.sqrt(dx**2+ dy**2), t[1:] - t[:-1])
        speed = np.concatenate(([speed[0]],speed))
        ssp  = np.where(speed>2.5)[0]
        sw = np.where((ssp[1:] - ssp[:-1])>1)[0]
        if (rat_name == 'shane') & (mod_name in ('mod2','mod3')):
            return
        if (rat_name == 'quentin') & (mod_name in ('mod3')):
            return
        inds = get_inds(rat_name, mod_name, sess_name)
        i0 += len(inds)
        for m in inds:
            s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
            s = s[(s>= min_time0) & (s< max_time0)]
            min_time1 = t[ssp[0]-1]
            max_time1 = t[ssp[sw[0]]+1]
            sall = []
            for ss in range(len(sw)):
                stemp = s[(s>= min_time1) & (s< max_time1)]
                sall.extend(stemp)
                min_time1 = t[ssp[sw[ss-1]]-1]
                max_time1 = t[ssp[sw[ss]]+1]
            spikes_box[i] = np.array(sall)-min_time0
            i += 1
    else:
        min_time0, max_time0 = times_all[rat_name + '_sleep'][0]
        t = marvin['tracking']['t'][()][0,1:-1]
        times = np.where((t>=min_time0) & (t<max_time0))
        t = t[times]
        sw = times_all[rat_name + '_' + sess_name]
        if (rat_name == 'shane') & (mod_name in ('mod2','mod3')):
            return
        if (rat_name == 'quentin') & (mod_name in ('mod3')):
            return
        inds = get_inds(rat_name, mod_name, sess_name)
        i0 += len(inds)
        for m in inds:
            s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
            s = s[(s>= min_time0) & (s< max_time0)]
            sall = []
            for ss in range(len(sw)):
                min_time1 = sw[ss][0] #t[ssp[sw[ss-1]]-1]
                max_time1 = sw[ss][1]
                stemp = s[(s>= min_time1) & (s< max_time1)]
                sall.extend(stemp)
            spikes_box[i] = np.array(sall)-min_time0
            i += 1
    return get_isi_acorr(spikes_box, bLog = False, bOne = True, maxt = 0.2)


times_all = {}
times_all['quentin_maze'] = ((18977, 25355),)
times_all['shane_maze'] = ((13670, 14847),)
times_all['shane_maze2'] = ((23186, 24936),)
times_all['roger_maze'] = ((16925, 18026),
                            (18183, 20704),)


times_all['roger_maze'] = ((16925, 20704),)



times_all['quentin_rem'] = ((10424.38,10565.21),
    (11514.83,11716.55),
    (12780.35,13035.9),
    (14301.05,14475.470000000001),
    (15088.98,15310.66),
    (17739.35,17797.21))
times_all['roger_box'] = ((7457, 16045),)
times_all['roger_box_rec2'] = ((10617, 13004),)
times_all['quentin_box'] = ((27826, 31223),)
times_all['shane_box'] = ((9939, 12363),)

times_all['roger_sleep'] = ((396, 9941),)
times_all['quentin_sleep'] = ((9576, 18812),)
times_all['shane_sleep'] = ((14942, 23133),)

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

times_all['shane_sws'] = ((15135.43,15164.98),
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

times_all['shane_rem'] = ((16756.48,16858.35),
    (18284.510000000002,18384.48),
    (19126.9,19151.98),
    (19164.2,19259.86),
    (20137.48,20188.68),
    (20975.93,21060.02),
    (21460.02,21590.43))


# In[311]:


sess_name_0 = 'box'
sess_name_00 = 'box_rec2'
sess_name_1 = 'maze'
sess_name_2 = 'rem'
sess_name_3 = 'sws'
sess_name_33 = 'sws_c0'
f = np.load('autocorrelogram_classes_300621.npz', allow_pickle = True)
indsclasses = f['indsclasses'][:189]
f.close()
mcs = {}
mtots = {}
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,2*np.pi, numangsint)
xv, yv = np.meshgrid(bins[0:-1] + (bins[1:] -bins[:-1])/2, 
                 bins[0:-1] + (bins[1:] -bins[:-1])/2)
pos  = np.concatenate((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis]),1)
ccos = np.cos(pos)
csin = np.sin(pos)

for (rat_name, rec_name, mod_name) in (#('roger', 'rec1', 'mod1'),
                                       #('roger', 'rec1', 'mod3'),
                                       #('roger', 'rec1', 'mod4'),
                                       #('roger', 'rec2', 'mod1'),
                                       ('roger', 'rec2', 'mod1_inds'),
                                       #('roger', 'rec2', 'mod2'),
                                       #('roger', 'rec2', 'mod3'),
                                       #('quentin', 'rec1', 'mod1'),
                                       #('quentin', 'rec1', 'mod2'),
                                       #('shane', 'rec1', 'mod1'),                                        
                                      ):
    b_inds = False
    k_inds = [0,1]
    if mod_name == 'mod1_inds':
        mod_name = 'mod1'
        b_inds = True
        k_inds = [0,1,2]
        
    cs = {}
    xs = {}
    ys = {}
    aas = {}
    spks = {}
    if rat_name == 'quentin':
        ss = [sess_name_0, sess_name_1, sess_name_2, sess_name_3]
    elif rec_name == 'rec2':
        if mod_name == 'mod1':
            ss = [sess_name_00, sess_name_2, sess_name_33]
        else:
            ss = [sess_name_00, sess_name_2, sess_name_3]
    else:
        ss = [sess_name_0, sess_name_1]
    num_figs = 0

    file_name_all = rat_name +'_' + rec_name + '_' + mod_name
    for sess_name in ss:

        file_name_all += '_' + sess_name
        num_figs+= 1
        file_name = rat_name + '_' + mod_name + '_' + sess_name + '_' + ss[0]
        f = np.load('Results/Orig/' + file_name + '_alignment_dec.npz', allow_pickle = True)
        cs[sess_name] = f['csess']
        f.close()
        spks[sess_name] = get_spikes(rat_name, rec_name, mod_name, sess_name, bConj = True)
        if sess_name[:6] in ('sws_c0',):
            ts = np.where(np.sum(spks[sess_name][:,indsclasses==0],1)>0)[0]
        else:
            if (sess_name[:3] in ('rem','sws')) & (rat_name == 'roger'):
                spk = load_spikes(rat_name, mod_name, sess_name[:3] + '_rec2')
            else:
                spk = load_spikes(rat_name, mod_name, sess_name)
            ts = np.where(np.sum(spk>0, 1)>=1)[0]
        if sess_name[:3] in ('box', 'maz'):
            num_figs += 2
            xs[sess_name], ys[sess_name], aas[sess_name], speed0 = load_pos(rat_name, rec_name, sess_name, bSpeed = True)
            xs[sess_name] = xs[sess_name][speed0>2.5]
            ys[sess_name] = ys[sess_name][speed0>2.5]
            aas[sess_name] = aas[sess_name][speed0>2.5]
            xs[sess_name] = xs[sess_name][ts]
            ys[sess_name] = ys[sess_name][ts]
            aas[sess_name] = aas[sess_name][ts]
            spks[sess_name] = spks[sess_name][speed0>2.5]   
        spks[sess_name] = spks[sess_name][ts,:]
        
    r_box = transforms.Affine2D().skew_deg(15,15)
    numangsint = 51
    sig = 2.75

    tot_path = rat_name + '/' + rec_name + '/' + 'data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')
    numn = len(marvin["clusters"]["tc"][0])

    mvl_hd = np.zeros((numn))

    for i in range(numn):
        if len(marvin[marvin["clusters"]["tc"][0][i]])==4:
            mvl_hd[i] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['mvl'][()]

    inds = get_inds(rat_name, mod_name, ss[0])
    mvl_hd = mvl_hd[inds]

    from matplotlib import gridspec
    get_ipython().run_line_magic('matplotlib', 'inline')
    from scipy.stats import binned_statistic
    numbins = 50
    bins = np.linspace(0,2*np.pi, numbins+1)
    sig_hd = np.sqrt(sig)
    num_neurons = len(spks[sess_name][0,:]) 
    mtots = {}
    num_figs +=1
    for s in ss:
        mtots[s] = np.zeros((num_neurons, numbins, numbins))
    plt.viridis()

    if len(ss) == 4:
        numfigs = 8
    else:
        numfigs = 6
    
    for k in k_inds:
        if b_inds:
            indstemp = np.where(indsclasses == k)[0]
        else:
            if k == 0:
                indstemp = np.where(mvl_hd<=0.3)[0]
            else:
                indstemp = np.where(mvl_hd>0.3)[0]
        num_neurons = len(indstemp)

        for sess_name in ss:
            mtottemp = np.zeros((num_neurons,len(bins)-1,len(bins)-1))
            mcstemp = np.zeros((num_neurons,2))
            for nn, n in enumerate(indstemp[:]):
                spk = spks[sess_name][:,n]
                cc1 = cs[sess_name]
                
                mtot_tmp, x_edge, y_edge,c2 = binned_statistic_2d(cc1[:,0],cc1[:,1], 
                                                      spk, statistic='mean', 
                                                      bins=bins, range=None, expand_binnumbers=True)
                mtottemp[nn,:,:] = mtot_tmp.copy()
                mtot_tmp = mtot_tmp.flatten()
                nans  = ~np.isnan(mtot_tmp)
                
                centcos = np.sum(np.multiply(ccos[nans,:],mtot_tmp[nans,np.newaxis]),0)
                centsin = np.sum(np.multiply(csin[nans,:],mtot_tmp[nans,np.newaxis]),0)
                mcstemp[nn,:] = np.arctan2(centsin,centcos)%(2*np.pi)
                
                
            mcs[rat_name + '_' + mod_name + '_' + sess_name + '_' + str(k)] = mcstemp
            mtots[rat_name + '_' + mod_name + '_' + sess_name + '_' + str(k)] = mtottemp
        if b_inds:
            mod_name = 'mod1_class'


# In[312]:


np.shape(mtots)


# In[313]:


#f = np.load('masscentersall.npz', allow_pickle = True )
#mcs = f['mcs'][()]
#f.close()


from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np 
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import *

sig = 2.75
num_shuffle = 1000

_2_PI = 2*np.pi
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)

allkeys = np.array(list(mcs.keys()))
for i1 in np.arange(3):
    for j1 in np.arange(3):
        if j1 == 0:
            file_name = 'roger_mod1_class' + str(i1) + '_box_rem'
            masscenters_1 = mcs[allkeys[i1*3]].copy()
            masscenters_2 = mcs[allkeys[i1*3+1]].copy()
            mtot_1 = mtots[allkeys[i1*3]].copy()
            mtot_2 = mtots[allkeys[i1*3+1]].copy()
        elif j1 == 1:
            file_name = 'roger_mod1_class' + str(i1) + '_box_sws'
            masscenters_1 = mcs[allkeys[i1*3]].copy()
            masscenters_2 = mcs[allkeys[i1*3+2]].copy()
            mtot_1 = mtots[allkeys[i1*3]].copy()
            mtot_2 = mtots[allkeys[i1*3+2]].copy()
        else:
            file_name = 'roger_mod1_class' + str(i1) + '_rem_sws'
            masscenters_1 = mcs[allkeys[i1*3+1]].copy()
            masscenters_2 = mcs[allkeys[i1*3+2]].copy()
            mtot_1 = mtots[allkeys[i1*3+1]].copy()
            mtot_2 = mtots[allkeys[i1*3+2]].copy()
    
        num_neurons = len(masscenters_1)
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

        num_neurons = len(masscenters_1[:,0])
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



        fig = plt.figure()
        ax = plt.axes()
        numbins = 30
        meantemp = np.zeros(numbins)
        corr_all = np.array([])
        mean_corr_all = np.zeros(num_shuffle)
        for i in range(num_shuffle):
            corr_all = np.concatenate((corr_all, corr_shuffle[i]))
            mean_corr_all[i] = np.mean(corr_shuffle[i])

        meantemp1 = np.histogram(corr_all, range = (-1,1), bins = numbins)[0]
        meantemp = np.cumsum(meantemp1)
        meantemp = np.divide(meantemp, num_shuffle)
        meantemp = np.divide(meantemp, num_neurons)
        y,x = np.histogram(corr, range = (-1,1), bins = numbins)
        y = np.cumsum(y)
        y = np.divide(y, num_neurons)
        x = x[1:]-(x[1]-x[0])/2
        ax.plot(x, meantemp, c = 'g', alpha = 0.8, lw = 5)
        ax.plot(x,y, c = 'r', alpha = 0.8, lw = 5)

        xs = [-1, -0.5, 0.0, 0.5, 1.0]
        ax.set_xticks(xs)
        ax.set_xticklabels(np.zeros(len(xs),dtype=str))
        ax.xaxis.set_tick_params(width=1, length =5)
        ys = [0.0, 0.25, 0.5, 0.75, 1.0]
        ax.set_yticks(ys)
        ax.set_yticklabels(np.zeros(len(ys),dtype=str))
        ax.yaxis.set_tick_params(width=1, length =5)

        ax.set_xlim([-1,1])
        ax.set_ylim([-0.05,1.05])

        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()

        ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*1.4))

        plt.gca().axes.spines['top'].set_visible(False)
        plt.gca().axes.spines['right'].set_visible(False)


        ##################### Cumulative distribution Distance ############################
        fig = plt.figure()
        ax = plt.axes()
        numbins = 30
        meantemp = np.zeros(numbins)
        dist_all = np.array([])
        mean_dist_all = np.zeros(num_shuffle)
        for i in range(num_shuffle):
            dist_all = np.concatenate((dist_all, dist_shuffle[i]))
            mean_dist_all[i] = np.mean(dist_shuffle[i])

        meantemp1 = np.histogram(dist_all, range = (0,np.sqrt(2)*np.pi), bins = numbins)[0]
        meantemp = np.cumsum(meantemp1)
        meantemp = np.divide(meantemp, num_shuffle)
        meantemp = np.divide(meantemp, num_neurons)
        y,x = np.histogram(dist, range = (0,np.sqrt(2)*np.pi), bins = numbins)
        y = np.cumsum(y)
        y = np.divide(y, num_neurons)
        x = x[1:]-(x[1]-x[0])/2
        x *= 180/np.pi
        ax.plot(x, meantemp, c = 'g', alpha = 0.8, lw = 5)
        ax.plot(x,y, c = 'r', alpha = 0.8, lw = 5)

        ys = [0.0, 0.25, 0.5, 0.75, 1.0]
        ax.set_yticks(ys)
        ax.set_yticklabels(np.zeros(len(ys),dtype=str))
        ax.yaxis.set_tick_params(width=1, length =5)

        xs = [0, 62.5, 125, 187.5, 250]
        ax.set_xticks(xs)
        ax.set_xticklabels(np.zeros(len(xs),dtype=str))
        ax.xaxis.set_tick_params(width=1, length =5)

        ax.set_xlim([0,255])
        ax.set_ylim([-0.05,1.05])

        plt.gca().axes.spines['top'].set_visible(False)
        plt.gca().axes.spines['right'].set_visible(False)

        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()

        ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*1.4))


        print(file_name)
        print('dist')
        print('orig: ', dist.mean(), dist.std()/np.sqrt(num_neurons))
        print('shuffle: ', mean_dist_all.mean(), mean_dist_all.std())

        print(file_name)
        print('corr')
        print('orig: ', corr.mean(), corr.std()/np.sqrt(num_neurons))
        print('shuffle: ', mean_corr_all.mean(), mean_corr_all.std())
        plt.show()


# In[315]:


allkeys


# In[316]:


#f = np.load('masscentersall.npz', allow_pickle = True )
#mcs = f['mcs'][()]
#f.close()


from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np 
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import *

sig = 2.75
num_shuffle = 1000

_2_PI = 2*np.pi
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)

allkeys = np.array(list(mcs.keys()))
for i1, class_name in ((0,'bursty'),
                       (1,'theta'),
                       (2,'nonbursty')
                      ):
    j1 = 1
    file_name = 'roger_mod1_class' + str(i1) + '_box_sws'
    masscenters_1 = mcs[allkeys[i1*3]].copy()
    masscenters_2 = mcs[allkeys[i1*3+2]].copy()
    mtot_1 = mtots[allkeys[i1*3]].copy()
    mtot_2 = mtots[allkeys[i1*3+2]].copy()

    num_neurons = len(masscenters_1)
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

    num_neurons = len(masscenters_1[:,0])
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

    numbins = 30
    meantemp = np.zeros(numbins)
    corr_all = np.array([])
    mean_corr_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        corr_all = np.concatenate((corr_all, corr_shuffle[i]))
        mean_corr_all[i] = np.mean(corr_shuffle[i])

    meantemp1 = np.histogram(corr_all, range = (-1,1), bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(corr, range = (-1,1), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2

    data = []
    data_names = []

    data.append(pd.Series(x))
    data.append(pd.Series(meantemp))
    data.append(pd.Series(y))
    data_names.extend(['Bins'])
    data_names.extend(['Shuffled'])
    data_names.extend(['Original'])
    
    df = pd.concat(data, ignore_index=True, axis=1)            
    df.columns = data_names
    fname = 'R1_' + class_name + '_OF_SWS' + '_corr'
    df.to_excel(fname + '.xlsx', sheet_name=fname)  
    
    ##################### Cumulative distribution Distance ############################
    numbins = 30
    meantemp = np.zeros(numbins)
    dist_all = np.array([])
    mean_dist_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        dist_all = np.concatenate((dist_all, dist_shuffle[i]))
        mean_dist_all[i] = np.mean(dist_shuffle[i])

    meantemp1 = np.histogram(dist_all, range = (0,np.sqrt(2)*np.pi), bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(dist, range = (0,np.sqrt(2)*np.pi), bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= 180/np.pi
    data = []
    data_names = []

    data.append(pd.Series(x))
    data.append(pd.Series(meantemp))
    data.append(pd.Series(y))
    data_names.extend(['Bins'])
    data_names.extend(['Shuffled'])
    data_names.extend(['Original'])
    
    df = pd.concat(data, ignore_index=True, axis=1)            
    df.columns = data_names
    fname = 'R1_' + class_name + '_OF_SWS' + '_dist'
    df.to_excel(fname + '.xlsx', sheet_name=fname)  


# In[ ]:


np.savez('deviance_all' + file_name, LOOtorscoresall)


# In[318]:



import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

colors_envs['roger_mod1_box_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_box_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_box_rec2'] = [[0.7,0.,0.7]]

LOOtorscoresall = {}
nils = 0 

LOOtorscoresall['box'] = []
LOOtorscoresall['rem'] = []
LOOtorscoresall['sws'] = []
for rat_name, mod_name, sess_name in (('roger', 'mod1', 'box_rec2'),
                                     ('roger', 'mod2', 'box_rec2'),
                                     ('roger', 'mod3', 'box_rec2'),
                                     
                                     ('roger', 'mod1', 'rem'),
                                     ('roger', 'mod2', 'rem'),
                                     ('roger', 'mod3', 'rem'),
                                     ('roger', 'mod1', 'sws'),
                                     ('roger', 'mod2', 'sws'),
                                     ('roger', 'mod3', 'sws'),
                                      
                                     ('quentin', 'mod1', 'box'),
                                     ('quentin', 'mod1', 'rem'),
                                     ('quentin', 'mod1', 'sws'),
                                     
                                     ('quentin', 'mod2', 'box'),
                                     ('quentin', 'mod2', 'rem'),
                                     ('quentin', 'mod2', 'sws'),

                                      ('shane', 'mod1', 'box'),
                                     ):

    file_name = rat_name + '_' + mod_name + '_' + sess_name    

    if (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name == 'sws'):
        
        f = np.load('Dev_all/deviance_all' + file_name + '.npz', allow_pickle = True)
        LOOtorscores = f['arr_0'][()]
        f.close()
        LOOtorscoresall[sess_name[:3]].extend(LOOtorscores[file_name + 'class1'])
        LOOtorscoresall[sess_name[:3]].extend(LOOtorscores[file_name + 'class2'])
        LOOtorscoresall[sess_name[:3]].extend(LOOtorscores[file_name + 'class3'])
        continue

    if (rat_name == 'roger') & (sess_name in ('rem', 'sws')):
        file_name += '_rec2'
    
    f = np.load('Dev_all/deviance_all' + file_name + '.npz', allow_pickle = True)
    LOOtorscores = f['arr_0'][()]
    f.close()
    LOOtorscoresall[sess_name[:3]].extend(LOOtorscores[file_name + '_pure'])
    LOOtorscoresall[sess_name[:3]].extend(LOOtorscores[file_name + '_conj'])
    


# In[322]:


f = np.load('autocorrelogram_classes_300621.npz', allow_pickle = True)
indsclasses = f['indsclasses']
classes = f['classes']
indsall = f['indsall']
f.close()


# In[323]:


inds_review = np.where(indsclasses[:189]==0)[0]
inds_review = np.concatenate((inds_review, np.where(indsclasses[:189]==1)[0]))
inds_review = np.concatenate((inds_review, np.where(indsclasses[:189]==2)[0]))
inds_review = np.concatenate((inds_review, np.where(indsall==2)[0]))
inds_review = np.concatenate((inds_review, np.where(indsall==3)[0]))
inds_review = np.concatenate((inds_review, np.where(indsall==4)[0]))
inds_review = np.concatenate((inds_review, np.where(indsall==5)[0]))
inds_review = np.concatenate((inds_review, np.where(indsall==6)[0]))
inds_review = np.concatenate((inds_review, np.where(indsall==7)[0]))
inds_review = np.concatenate((inds_review, np.where(indsall==8)[0]))
inds_review = np.concatenate((inds_review, np.where(indsall==9)[0]))
inds_review = np.concatenate((inds_review, np.where(indsall==10)[0]))
inds_review = np.concatenate((inds_review, np.where(indsall==11)[0]))
inds_review = np.array(inds_review)

inds_review1 = np.where(indsall==0)[0]
inds_review1 = np.concatenate((inds_review1, np.where(indsall==1)[0]))
inds_review1 = np.concatenate((inds_review1, np.where(indsall==2)[0]))
inds_review1 = np.concatenate((inds_review1, np.where(indsall==3)[0]))
inds_review1 = np.concatenate((inds_review1, np.where(indsall==4)[0]))
inds_review1 = np.concatenate((inds_review1, np.where(indsall==5)[0]))
inds_review1 = np.concatenate((inds_review1, np.where(indsall==6)[0]))
inds_review1 = np.concatenate((inds_review1, np.where(indsall==7)[0]))
inds_review1 = np.concatenate((inds_review1, np.where(indsall==8)[0]))
inds_review1 = np.concatenate((inds_review1, np.where(indsall==9)[0]))
inds_review1 = np.concatenate((inds_review1, np.where(indsall==10)[0]))
inds_review1 = np.concatenate((inds_review1, np.where(indsall==11)[0]))
inds_review1 = np.array(inds_review1)


# In[324]:


fig = plt.figure()
ax = plt.axes()
cs = {}
cs[0] = '#1f77b4'
cs[1] = '#ff7f0e'
cs[2] = '#2ca02c'

ls = {}
ls['box'] = '-'
ls['rem'] = '--'
ls['sws'] = ':'

for sess_name in ('box', 'rem', 'sws'):
    infos = np.array(LOOtorscoresall[sess_name[:3]])
    print(np.min(infos), np.max(infos))
    numbins = 50
    for i in range(3):
        if sess_name in ('rem',):
            currinds = np.where(indsclasses[inds_review1[:-np.sum(indsall >9)]]==i)[0]            
        elif sess_name in ('sws',):
            currinds = np.where(indsclasses[inds_review[:-np.sum(indsall >9)]]==i)[0]            
        else:
            currinds = np.where(indsclasses[inds_review1]==i)[0]
        infosclass = infos[currinds].copy()
        
        num_neurons = len(currinds)
        x = np.linspace(0.0,0.4, numbins+1)
        
        y = np.digitize(infosclass, x)
        

        y = np.bincount(y, minlength =numbins+2)
        y[1] += y[0]
        y[-2] += y[-1]
        y = y[1:-1]
        y = np.cumsum(y)
        y = np.divide(y, num_neurons)
#        print(y)
        x = x[1:]-(x[1]-x[0])/2
        ax.plot(x,y, c = cs[i],ls = ls[sess_name], alpha = 0.8, lw = 3)
#        ax.scatter([np.median(infosclass)],
#                [0.5], c = cs[i], alpha = 1, marker = 'X', lw = 0.1,s = 150)
        

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, 1.02])
    ax.set_xticks([0,0.1,0.2, 0.3, 0.4])
    ax.set_xticklabels(['','',''])
    ax.set_yticks([0,0.5,1.0])
    ax.set_yticklabels(['','',''])
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 

ax.plot(ax.get_xlim(), [0.5, 0.5], ls = ':', c = 'k', alpha = 0.6, lw = 2)


# In[328]:


numbins = 50
x = np.linspace(0.0,0.4, numbins+1)
x1 = x[1:]-(x[1]-x[0])/2

data = []
data_names = []
data.append(pd.Series(x1))
data_names.extend(['Bins'])

for sess_name in ('box', 'rem', 'sws'):
    infos = np.array(LOOtorscoresall[sess_name[:3]])
    for i in range(3):
        if sess_name in ('rem',):
            currinds = np.where(indsclasses[inds_review1[:-np.sum(indsall >9)]]==i)[0]            
        elif sess_name in ('sws',):
            currinds = np.where(indsclasses[inds_review[:-np.sum(indsall >9)]]==i)[0]            
        else:
            currinds = np.where(indsclasses[inds_review1]==i)[0]
        infosclass = infos[currinds].copy()
        
        num_neurons = len(currinds)
 
        y = np.digitize(infosclass, x)
        

        y = np.bincount(y, minlength =numbins+2)
        y[1] += y[0]
        y[-2] += y[-1]
        y = y[1:-1]
        y = np.cumsum(y)
        y = np.divide(y, num_neurons)
        
        data.append(pd.Series(y))
        if sess_name =='box':
            sess_name = 'OF'
            
        data_names.extend([sess_name + '_' + class_name])
    
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = 'cumulative_deviance_class'
df.to_excel(fname + '.xlsx', sheet_name=fname)  


# In[325]:


from scipy.stats import wilcoxon
import h5py
import numpy as np 
from ripser import Rips, ripser
from sklearn import preprocessing
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import lsmr
from datetime import datetime 
import time
import sys
import numba
import matplotlib
import matplotlib.pyplot as plt
from utils import *

rat_names = ['roger', 'quentin', 'shane']
mod_names = ['mod1', 'mod2', 'mod3', 'mod4']
sess_names = ['maze']

Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

colors_envs['roger_mod1_box_rec2'] = [[0.7,0.3,0.3]]
colors_envs['roger_mod2_box_rec2'] = [[0.7,0.7,0]]
colors_envs['roger_mod3_box_rec2'] = [[0.7,0.,0.7]]

infoscoresall = {}
nils = 0 

infoscoresall['box'] = []
#infoscoresall['maz'] = []
infoscoresall['rem'] = []
infoscoresall['sws'] = []
for rat_name, mod_name, sess_name in (('roger', 'mod1', 'box_rec2'),
                                     ('roger', 'mod2', 'box_rec2'),
                                     ('roger', 'mod3', 'box_rec2'),
                                     
                                     ('roger', 'mod1', 'rem'),
                                     ('roger', 'mod2', 'rem'),
                                     ('roger', 'mod3', 'rem'),
                                     ('roger', 'mod1', 'sws'),
                                     ('roger', 'mod2', 'sws'),
                                     ('roger', 'mod3', 'sws'),
                                      
                                     ('quentin', 'mod1', 'box'),
                                     ('quentin', 'mod1', 'rem'),
                                     ('quentin', 'mod1', 'sws'),
                                     
                                     ('quentin', 'mod2', 'box'),
                                     ('quentin', 'mod2', 'rem'),
                                     ('quentin', 'mod2', 'sws'),

                                      ('shane', 'mod1', 'box'),
#                                     ('shane', 'mod1', 'maze')
                                     ):

    file_name = rat_name + '_' + mod_name + '_' + sess_name    

    if (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name == 'sws'):
        sess_name += '_rec2'
        spktmp = load_spikes(rat_name, mod_name, sess_name, bSpeed = True)       
        spk = np.zeros((len(spktmp[:,0]), 189))
        spk[:,indsall[:189]==0] = spktmp.copy()
        spktmp = load_spikes(rat_name, mod_name, sess_name, bSpeed = True, bConj = True)
        spk[:,indsall[:189]==1] = spktmp.copy()
        f = np.load('GLM_info_review_conj_stats/' + file_name + '_class1_conj_info.npz')
        infoscorestmp = f['I_torus']
        f.close()
        for n, m in enumerate(np.where(indsclasses[:189]==0)[0]):            
            f = np.load('Data/roger_mod1_sws_class1_coords_N' + str(n) + '.npz', allow_pickle = True)
            times = f['times']
            f.close()
            infoscores = [np.divide(infoscorestmp[n], np.mean(spk[times,m]))]
            infoscoresall[sess_name[:3]].extend(infoscores)
            
        f = np.load('Data/roger_mod1_sws_class1_newdecoding.npz', allow_pickle = True)
        times = f['times']
        f.close()
        spk = spk[times,:]
        
        f = np.load('GLM_info_review_conj_stats/' + file_name + '_class2_conj_info.npz')
        infoscores = np.divide(f['I_torus'], np.mean(spk[:,indsclasses[:189]==1],0))
        f.close()
        infoscoresall[sess_name[:3]].extend(infoscores)
        
        f = np.load('GLM_info_review_conj_stats/' + file_name + '_class3_conj_info.npz')
        infoscores = np.divide(f['I_torus'], np.mean(spk[:,indsclasses[:189]==2],0))
        f.close()
        
        infoscoresall[sess_name[:3]].extend(infoscores)
        continue

    if (rat_name == 'roger') & (sess_name in ('rem', 'sws')):
        file_name += '_rec2'
        sess_name += '_rec2'
    spktmp = load_spikes(rat_name, mod_name, sess_name, bSpeed = True)       
        
    f = np.load('Data/' + file_name + '_newdecoding.npz', allow_pickle = True)
    times = f['times']
    f.close()
    
    f = np.load('GLM_info_review_stats/' + file_name + '_info.npz')
    infoscores = np.divide(f['I_5'], np.mean(spktmp[times,:],0))
    f.close()
    infoscoresall[sess_name[:3]].extend(infoscores)

    spktmp = load_spikes(rat_name, mod_name, sess_name, bSpeed = True, bConj = True)       
    f = np.load('GLM_info_review_conj_stats/' + file_name + '_conj_info.npz')
    infoscores = np.divide(f['I_torus'], np.mean(spktmp[times,:],0))
    f.close()
    infoscoresall[sess_name[:3]].extend(infoscores)
    if np.isnan(infoscores[0]):
        print(rat_name, mod_name, sess_name)


# In[326]:


fig = plt.figure()
ax = plt.axes()
cs = {}
cs[0] = '#1f77b4'
cs[1] = '#ff7f0e'
cs[2] = '#2ca02c'

ls = {}
ls['box'] = '-'
ls['rem'] = '--'
ls['sws'] = ':'

for sess_name in ('box', 'rem', 'sws'):
    infos = np.array(infoscoresall[sess_name[:3]])/100
    print(np.min(infos), np.max(infos))
    numbins = 50
    for i in range(3):
        if sess_name in ('rem',):
            currinds = np.where(indsclasses[inds_review1[:-np.sum(indsall >9)]]==i)[0]            
        elif sess_name in ('sws',):
            currinds = np.where(indsclasses[inds_review[:-np.sum(indsall >9)]]==i)[0]            
        else:
            currinds = np.where(indsclasses[inds_review1]==i)[0]
        infosclass = infos[currinds].copy()
        
        num_neurons = len(currinds)
        x = np.linspace(0.0,3, numbins+1)
        
        y = np.digitize(infosclass, x)
        

        y = np.bincount(y, minlength =numbins+2)
        y[1] += y[0]
        y[-2] += y[-1]
        y = y[1:-1]
        y = np.cumsum(y)
        y = np.divide(y, num_neurons)
#        print(y)
        x = x[1:]-(x[1]-x[0])/2
        ax.plot(x,y, c = cs[i],ls = ls[sess_name], alpha = 0.8, lw = 3)
#        ax.scatter([np.median(infosclass)],
#                [0.5], c = cs[i], alpha = 1, marker = 'X', lw = 0.1,s = 150)
        print(i,np.median(infosclass))

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, 1.02])
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['','','', ''])
    ax.set_yticks([0,0.5,1.0])
    ax.set_yticklabels(['','',''])
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
ax.plot(ax.get_xlim(), [0.5, 0.5], ls = ':', c = 'k', alpha = 0.6, lw = 2)


# In[327]:


numbins = 50
x = np.linspace(0.0,3, numbins+1)
x1 = x[1:]-(x[1]-x[0])/2
data = []
data_names = []
data.append(pd.Series(x1))
data_names.extend(['Bins'])
        
for sess_name in ('box', 'rem', 'sws'):
    infos = np.array(infoscoresall[sess_name[:3]])/100
    
    for i, class_name in ((0, 'bursty'),
             (1, 'theta'),
             (2, 'nonbursty')):
        if sess_name in ('rem',):
            currinds = np.where(indsclasses[inds_review1[:-np.sum(indsall >9)]]==i)[0]            
        elif sess_name in ('sws',):
            currinds = np.where(indsclasses[inds_review[:-np.sum(indsall >9)]]==i)[0]            
        else:
            currinds = np.where(indsclasses[inds_review1]==i)[0]
        infosclass = infos[currinds].copy()
        
        num_neurons = len(currinds)
        
        y = np.digitize(infosclass, x)
        

        y = np.bincount(y, minlength =numbins+2)
        y[1] += y[0]
        y[-2] += y[-1]
        y = y[1:-1]
        y = np.cumsum(y)
        y = np.divide(y, num_neurons)
            
        data.append(pd.Series(y))
        if sess_name =='box':
            sess_name = 'OF'
            
        data_names.extend([sess_name + '_' + class_name])
    
df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = 'cumulative_info_class'
df.to_excel(fname + '.xlsx', sheet_name=fname)  

#        ax.plot(x,y, c = cs[i],ls = ls[sess_name], alpha = 0.8, lw = 3)


# ### Conjunctive HD tuning

# In[335]:


indscurr = np.where((indsclasses[:189]==1))[0]
print(indscurr.shape)
spk1 = spikes_sws_bin[:,indscurr]


# In[336]:


dim = 6
ph_classes = [0,1] 
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 1
coeff = 47
active_times = 15000
num_times = 5
n_points = 1200
k = 1000
nbs = 800
maxdim = 1


spk1 = sspikes_sws[:,indscurr].copy()
num_neurons = len(spk1[0,:])
times_cube = np.arange(0,len(spk1[:,0]),num_times)
movetimes = np.sort(np.argsort(np.sum(spk1[times_cube,:],1))[-active_times:])
movetimes = times_cube[movetimes]

pca_mod1,var_exp_mod1, eigval_mod1  = pca(preprocessing.scale(spk1[movetimes,:]), dim = dim)
plt.plot(eigval_mod1)
plt.show()
indstemp,dd,fs  = sample_denoising(pca_mod1,  k, 
                                    n_points, 1, metric)

X = squareform(pdist(pca_mod1[indstemp,:], metric))
knn_indices, knn_dists, __ = nearest_neighbors(X, n_neighbors = nbs, metric = 'precomputed', 
                                               angular=True, metric_kwds = {})
sigmas, rhos = smooth_knn_dist(knn_dists, nbs, local_connectivity=0)
rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
result = coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
result.eliminate_zeros()
transpose = result.transpose()
prod_matrix = result.multiply(transpose)
result = (result + transpose - prod_matrix)
result.eliminate_zeros()
d = result.toarray()
d = -np.log(d)
np.fill_diagonal(d,0)
thresh = np.max(d[~np.isinf(d)])
rips_real = ripser(d, maxdim=maxdim, coeff=coeff, thresh = thresh, 
                   do_cocycles=True, distance_matrix = True)


plt.figure()
plot_diagrams(
    rips_real["dgms"],
    plot_only=np.arange(maxdim+1),
    lifetime = True)


# In[337]:


dec_thresh = 0.99
diagrams = rips_real["dgms"] # the multiset describing the lives of the persistence classes
cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
dists_land = rips_real["dperm2all"] # the pairwise distance between the points 
births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
print(np.sum(np.isinf(deaths1)))
lives1 = deaths1-births1 # the lifetime for the 1-dim classes
iMax = np.argsort(lives1)

ph_classes = [0,1]
num_circ = len(ph_classes)

coords1 = np.zeros((num_circ, len(indstemp)))
threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]])*dec_tresh
for i, c in enumerate(ph_classes):
    cocycle = cocycles[iMax[-(c+1)]]
    if np.isinf(threshold):
        threshold = thresh
    coords1[i,:],inds = get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

num_neurons = len(spk1[0,:])
centcosall = np.zeros((num_neurons, num_circ, len(indstemp)))
centsinall = np.zeros((num_neurons, num_circ, len(indstemp)))
dsspikes = preprocessing.scale(spk1[movetimes[indstemp],:])
for neurid in range(num_neurons):
    sspikestemp = dsspikes[:, neurid].copy()
    centcosall[neurid,:,:] = np.multiply(np.cos(coords1[:, :]*2*np.pi),sspikestemp)
    centsinall[neurid,:,:] = np.multiply(np.sin(coords1[:, :]*2*np.pi),sspikestemp)



# In[338]:


#spk_box1 = sspikes_box.copy()#[:,indscurr].copy()#ind<2].copy()
spk_box1 = sspikes_box[:,indscurr].copy()#ind<2].copy()

plot_times = np.arange(0,len(spk_box1[:,0]),1)
dsspikes1 = preprocessing.scale(spk_box1[plot_times,:])

a = np.zeros((len(plot_times), num_circ, num_neurons))
a_norm = np.zeros((len(plot_times), num_neurons))

for n in range(num_neurons):
    a[:,:,n] = np.multiply(dsspikes1[:,n:n+1],np.sum(centcosall[n,:,:],1))
    
c = np.zeros((len(plot_times), num_circ, num_neurons))
for n in range(num_neurons):
    c[:,:,n] = np.multiply(dsspikes1[:,n:n+1],np.sum(centsinall[n,:,:],1))
    
mtot2 = np.sum(c,2)
mtot1 = np.sum(a,2)
coords_wake = np.arctan2(mtot2,mtot1)%(2*np.pi)


# In[346]:


def load_pos(rat_name, sess_name, bSpeed = False):    
    f = np.load('Data/tracking_' + rat_name + '_' + sess_name + '.npz', allow_pickle = True)
    xx = f['xx']
    yy = f['yy']
    print(list(f.keys()))
    tt = f['tt']
    if sess_name[:4] =='maze':    
        aa = []
    else:
        aa = f['aa']
    f.close()
        
    if bSpeed:
        xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
        yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
        dx = (xxs[1:] - xxs[:-1])*100
        dy = (yys[1:] - yys[:-1])*100
        speed = np.sqrt(dx**2+ dy**2)/0.01
        speed = np.concatenate(([speed[0]],speed))
        if (rat_name =='shane') & (sess_name == 'maze'):
            xx1, yy1, aa1, speed1 = load_pos(rat_name, 'maze2', True)
            xx = np.concatenate((xx,xx1))
            yy = np.concatenate((yy,yy1))
            aa = np.concatenate((aa,aa1))
            speed = np.concatenate((speed,speed1))
        return xx, yy, aa, speed, tt
    if (rat_name =='shane') & (sess_name == 'maze'):
        xx1, yy1, aa1 = load_pos(rat_name, 'maze2')
        xx = np.concatenate((xx,xx1))
        yy = np.concatenate((yy,yy1))
        aa = np.concatenate((aa,aa1))
    return xx,yy,aa, tt
xx,yy,aa, speed, tt =load_pos('roger', 'box_rec2', bSpeed = True)


# In[348]:


tt.shape


# In[349]:


tt = tt[speed>2.5]
xx = xx[speed>2.5]
yy = yy[speed>2.5]
aa = aa[speed>2.5]


# In[339]:


cctmp = coords_wake[:,0].copy()
aatmp = aa.copy()
#aatmp = aatmp[speed>2.5]
nnans = (~np.isnan(aatmp)) & (~np.isnan(cctmp[:]))
circmin = np.arctan2(np.mean(np.sin(cctmp[nnans]-aatmp[nnans])),
                     np.mean(np.cos(cctmp[nnans]-aatmp[nnans])))%(2*np.pi)
cc1 = (cctmp[:]-circmin)%(2*np.pi)
circdiff = np.mean(np.abs(np.arctan2(np.sin(cc1[nnans]-aatmp[nnans]),
                     np.cos(cc1[nnans]-aatmp[nnans]))))
print(circdiff)
t0, t1 = 800,10800
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.arange(t0,t1), cc1[t0:t1]/(2*np.pi)*360, 
           lw = 0.0, alpha = 0.7, label ='recorded', s = 10)
ax.scatter(np.arange(t0,t1), aatmp[t0:t1]/(2*np.pi)*360, 
           lw = 0.0, alpha = 0.7, label ='decoded', s = 10)
ax.set_xlim([t0,t1])
ax.set_ylim([0,2*np.pi/(2*np.pi)*360])

ax.set_yticks([0,180,360])
ax.set_xticks([800,5800, 10800])
ax.set_yticklabels(['', '', ''])
ax.set_xticklabels(['', '', '', ])

#ax.set_yticklabels(['0$\degree$', '180$\degree$', '360$\degree$'])
#ax.set_xticklabels(['0s', '5s', '10s', '15s', '20s'])

ax.set_aspect(8)


# In[351]:




data, data_names = [], []

data.append(pd.Series(tt[np.arange(t0,t1)]))
data_names.extend(['Time'])

data.append(pd.Series(aatmp[t0:t1]/(2*np.pi)*360))
data_names.extend(['Recorded_HD'])

data.append(pd.Series(cc1[t0:t1]/(2*np.pi)*360))
data_names.extend(['Decoded'])


df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = 'R1_conj_HD'
df.to_excel(fname + '.xlsx', sheet_name=fname)  


# #### Shane

# In[352]:



spk = load_spikes('shane', 'mod1', 'sws', bSmooth = True, bConj=True)
spk1 = load_spikes('shane', 'mod1', 'sws', bSmooth = True, bConj=False)
inds1 = indsclasses[indsall==10]
spk1_class2 = spk1[:, inds1==1]
inds0 = indsclasses[indsall==11]
spk0_class2 = spk[:, inds0==1]
sspikes_sws = np.concatenate((spk1_class2,spk0_class2),1)


spk = load_spikes('shane', 'mod1', 'box', bSmooth = True, bConj=True)
spk1 = load_spikes('shane', 'mod1', 'box', bSmooth = True, bConj=False)
spk1_class2 = spk1[:, inds1==1]
spk0_class2 = spk[:, inds0==1]

sspikes_box = np.concatenate((spk1_class2,spk0_class2),1)


# In[354]:


dim = 6
ph_classes = [0,1] 
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 1
coeff = 47
active_times = 15000
num_times = 5
n_points = 1200
k = 1000
nbs = 800
maxdim = 1


spk1 = sspikes_sws[:,:].copy()
num_neurons = len(spk1[0,:])
times_cube = np.arange(0,len(spk1[:,0]),num_times)
movetimes = np.sort(np.argsort(np.sum(spk1[times_cube,:],1))[-active_times:])
movetimes = times_cube[movetimes]

pca_mod1,var_exp_mod1, eigval_mod1  = pca(preprocessing.scale(spk1[movetimes,:]), dim = dim)
plt.plot(eigval_mod1)
plt.show()
indstemp,dd,fs  = sample_denoising(pca_mod1,  k, 
                                    n_points, 1, metric)

X = squareform(pdist(pca_mod1[indstemp,:], metric))
knn_indices, knn_dists, __ = nearest_neighbors(X, n_neighbors = nbs, metric = 'precomputed', 
                                               angular=True, metric_kwds = {})
sigmas, rhos = smooth_knn_dist(knn_dists, nbs, local_connectivity=0)
rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
result = coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
result.eliminate_zeros()
transpose = result.transpose()
prod_matrix = result.multiply(transpose)
result = (result + transpose - prod_matrix)
result.eliminate_zeros()
d = result.toarray()
d = -np.log(d)
np.fill_diagonal(d,0)
thresh = np.max(d[~np.isinf(d)])
rips_real = ripser(d, maxdim=maxdim, coeff=coeff, thresh = thresh, 
                   do_cocycles=True, distance_matrix = True)


plt.figure()
plot_diagrams(
    rips_real["dgms"],
    plot_only=np.arange(maxdim+1),
    lifetime = True)


# In[355]:


dec_thresh = 0.99
diagrams = rips_real["dgms"] # the multiset describing the lives of the persistence classes
cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
dists_land = rips_real["dperm2all"] # the pairwise distance between the points 
births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
print(np.sum(np.isinf(deaths1)))
lives1 = deaths1-births1 # the lifetime for the 1-dim classes
iMax = np.argsort(lives1)

ph_classes = [0,1]
num_circ = len(ph_classes)

coords1 = np.zeros((num_circ, len(indstemp)))
threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]])*dec_tresh
for i, c in enumerate(ph_classes):
    cocycle = cocycles[iMax[-(c+1)]]
    if np.isinf(threshold):
        threshold = thresh
    coords1[i,:],inds = get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

num_neurons = len(spk1[0,:])
centcosall = np.zeros((num_neurons, num_circ, len(indstemp)))
centsinall = np.zeros((num_neurons, num_circ, len(indstemp)))
dsspikes = preprocessing.scale(spk1[movetimes[indstemp],:])
for neurid in range(num_neurons):
    sspikestemp = dsspikes[:, neurid].copy()
    centcosall[neurid,:,:] = np.multiply(np.cos(coords1[:, :]*2*np.pi),sspikestemp)
    centsinall[neurid,:,:] = np.multiply(np.sin(coords1[:, :]*2*np.pi),sspikestemp)



# In[356]:


spk_box1 = sspikes_box.copy()

plot_times = np.arange(0,len(spk_box1[:,0]),1)
dsspikes1 = preprocessing.scale(spk_box1[plot_times,:])

a = np.zeros((len(plot_times), num_circ, num_neurons))
a_norm = np.zeros((len(plot_times), num_neurons))

for n in range(num_neurons):
    a[:,:,n] = np.multiply(dsspikes1[:,n:n+1],np.sum(centcosall[n,:,:],1))
    
c = np.zeros((len(plot_times), num_circ, num_neurons))
for n in range(num_neurons):
    c[:,:,n] = np.multiply(dsspikes1[:,n:n+1],np.sum(centsinall[n,:,:],1))
    
mtot2 = np.sum(c,2)
mtot1 = np.sum(a,2)
coords_wake = np.arctan2(mtot2,mtot1)%(2*np.pi)


# In[358]:


def load_pos(rat_name, sess_name, bSpeed = False):    
    f = np.load('Data/tracking_' + rat_name + '_' + sess_name + '.npz', allow_pickle = True)
    xx = f['xx']
    yy = f['yy']
    print(list(f.keys()))
    tt = f['tt']
    if sess_name[:4] =='maze':    
        aa = []
    else:
        aa = f['aa']
    f.close()
        
    if bSpeed:
        xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
        yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
        dx = (xxs[1:] - xxs[:-1])*100
        dy = (yys[1:] - yys[:-1])*100
        speed = np.sqrt(dx**2+ dy**2)/0.01
        speed = np.concatenate(([speed[0]],speed))
        if (rat_name =='shane') & (sess_name == 'maze'):
            xx1, yy1, aa1, speed1 = load_pos(rat_name, 'maze2', True)
            xx = np.concatenate((xx,xx1))
            yy = np.concatenate((yy,yy1))
            aa = np.concatenate((aa,aa1))
            speed = np.concatenate((speed,speed1))
        return xx, yy, aa, speed, tt
    if (rat_name =='shane') & (sess_name == 'maze'):
        xx1, yy1, aa1 = load_pos(rat_name, 'maze2')
        xx = np.concatenate((xx,xx1))
        yy = np.concatenate((yy,yy1))
        aa = np.concatenate((aa,aa1))
    return xx,yy,aa, tt
xx,yy,aa, speed, tt =load_pos('shane', 'box', bSpeed = True)


# In[359]:


tt = tt[speed>2.5]
xx = xx[speed>2.5]
yy = yy[speed>2.5]
aa = aa[speed>2.5]


# In[361]:


cctmp = 2*np.pi-coords_wake[:,0].copy()
aatmp = aa.copy()
#aatmp = aatmp[speed>2.5]
nnans = (~np.isnan(aatmp)) & (~np.isnan(cctmp[:]))
circmin = np.arctan2(np.mean(np.sin(cctmp[nnans]-aatmp[nnans])),
                     np.mean(np.cos(cctmp[nnans]-aatmp[nnans])))%(2*np.pi)
cc1 = (cctmp[:]-circmin)%(2*np.pi)
circdiff = np.mean(np.abs(np.arctan2(np.sin(cc1[nnans]-aatmp[nnans]),
                     np.cos(cc1[nnans]-aatmp[nnans]))))
print(circdiff)
t0, t1 = 800,10800
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.arange(t0,t1), cc1[t0:t1]/(2*np.pi)*360, 
           lw = 0.0, alpha = 0.7, label ='recorded', s = 10)
ax.scatter(np.arange(t0,t1), aatmp[t0:t1]/(2*np.pi)*360, 
           lw = 0.0, alpha = 0.7, label ='decoded', s = 10)
ax.set_xlim([t0,t1])
ax.set_ylim([0,2*np.pi/(2*np.pi)*360])

ax.set_yticks([0,180,360])
ax.set_xticks([800,5800, 10800])
ax.set_yticklabels(['', '', ''])
ax.set_xticklabels(['', '', '', ])


ax.set_aspect(8)


# In[362]:




data, data_names = [], []

data.append(pd.Series(tt[np.arange(t0,t1)]))
data_names.extend(['Time'])

data.append(pd.Series(aatmp[t0:t1]/(2*np.pi)*360))
data_names.extend(['Recorded_HD'])

data.append(pd.Series(cc1[t0:t1]/(2*np.pi)*360))
data_names.extend(['Decoded'])


df = pd.concat(data, ignore_index=True, axis=1)            
df.columns = data_names
fname = 'S1_conj_HD'
df.to_excel(fname + '.xlsx', sheet_name=fname)  

