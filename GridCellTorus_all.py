#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


def glm2d(xxss, ys, num_bins, periodicprior,  LAM, GoGaussian, nF):
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
        fg = fg==1
        X_space = preprocess_dataX2(xxss[fg,:], num_bins)
#        print(ys[fg], X_space, LAM, periodicprior, GoGaussian)
        P[:, i] = fitmodel(ys[fg], X_space, LAM, periodicprior, GoGaussian)

        xt = xxss#[~fg,:]
        X_test = X_space#preprocess_dataX2(xt, num_bins)

        if(GoGaussian):
            yt[fg] = np.dot(P[:, i], X_test)        
            Lvals[fg] = -np.sum( (ys-yt)**2 )             
        else:
            H = np.dot(P[:, i], X_test)
            expH = np.exp(H)
            yt[fg] = expH
            finthechat = (np.ravel(np.log(factorial(ys[fg]))))
            Lvals[fg] = (np.ravel(ys[fg]*H - expH)) - finthechat

    leastsq = np.sum( (ys-yt)**2 )
    #print('LEAST SQ', leastsq)
    ym = np.mean(ys)
    #return (np.sum((yt-ym)**2) / np.sum((ys-ym)**2))
    return yt, (1. - leastsq/np.sum((ys-ym)**2)), P, Lvals

def load_pos(rat_name, sess_name, bSpeed = False):    
    f = np.load('Data/tracking_' + rat_name + '_' + sess_name + '.npz', allow_pickle = True)
    xx = f['xx']
    yy = f['yy']
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
        return xx, yy, aa, speed
    if (rat_name =='shane') & (sess_name == 'maze'):
        xx1, yy1, aa1 = load_pos(rat_name, 'maze2')
        xx = np.concatenate((xx,xx1))
        yy = np.concatenate((yy,yy1))
        aa = np.concatenate((aa,aa1))
    return xx,yy,aa

def get_inds(rat_name, mod_name, sess_name, bOrig = False):
    if rat_name == 'shane':
        tot_path_all = 'shane/rec1/shane_all.mat'
        tot_path_mod = 'shane/rec1/shane_mod_final.mat'
        if mod_name == 'mod1':
            v = 'v3'
    elif rat_name == 'quentin':
        tot_path_all = 'quentin/rec1/quentin_all.mat'
        tot_path_mod = 'quentin/rec1/quentin_mod_final.mat'
        if mod_name == 'mod1':
            v = 'v3'
        elif mod_name == 'mod2':
            v = 'v4'        
    elif (rat_name == 'roger') & (sess_name in ('box_rec2', 'sws_rec2', 'rem_rec2')):
        tot_path_all = 'roger/rec2/roger_all.mat'
        tot_path_mod = 'roger/rec2/roger_rec2_mod_final.mat'
        if mod_name == 'mod1':
            v = 'v3'
        elif mod_name == 'mod2':
            v = 'v4'
        elif mod_name == 'mod3':
            v = 'v5'
    elif (rat_name == 'roger') & (sess_name in ('box', 'maze')):
        tot_path_all = 'roger/rec1/roger_all.mat'
        tot_path_mod = 'roger/rec1/roger_mod_final.mat'
        if mod_name == 'mod1':
            v = 'v3'
        elif mod_name == 'mod2':
            v = 'v4'
        elif mod_name == 'mod3':
            v = 'v5'
        elif mod_name == 'mod4':
            v = 'v6' 
    marvin = sio.loadmat(tot_path_all)
    mall = np.zeros(len(marvin['vv'][0,:]), dtype=int)
    mall1 = np.zeros(len(marvin['vv'][0,:]), dtype=int)
    for i,m in enumerate(marvin['vv'][0,:]):
        mall[i] = int(m[0][0])
        mall1[i] = int(m[0][2:])

    marvin = sio.loadmat(tot_path_mod)
    m2 = np.zeros(len(marvin[v][:,0]), dtype=int)
    m22 = np.zeros(len(marvin[v][:,0]), dtype=int)
    for i,m in enumerate(marvin[v][:,0]):
        m2[i] = int(m[0][0])
        m22[i] = int(m[0][2:])            

    inds = np.zeros(len(m2), dtype = int)
    for i in range(len(m2)):
        inds[i] = np.where((mall==m2[i]) & (mall1==m22[i]))[0]

    if (sess_name[-4:] == 'rec2'):
        rec_name = 'rec2'
    else:
        rec_name = 'rec1' 
    if (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name in ('box', 'maze')):
        inds1 = get_inds(rat_name, 'mod2', sess_name)
        inds = np.concatenate((inds, inds1))
    if bOrig:
        return inds, m2, m22
        
    return inds

def get_mvl(rat_name, sess_name):
    if (sess_name[-4:] == 'rec2'):
        rec_name = 'rec2'
    else:
        rec_name = 'rec1' 
    tot_path = rat_name + '/' + rec_name + '/' + 'data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')
    numn = len(marvin["clusters"]["tc"][0])
    mvl_hd = np.zeros((numn))
    mvl_si = np.zeros((numn))
    mvl_theta = np.zeros((numn))
    mu_theta = np.zeros((numn))
    hd_tun = np.zeros((numn, 60))
    for i in range(numn):
        if len(marvin[marvin["clusters"]["tc"][0][i]])==4:
            mvl_hd[i] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['mvl'][()]
            mvl_si[i] = marvin[marvin["clusters"]["tc"][0][i]]['pos']['si'][()]
            mu_theta[i] = marvin[marvin["clusters"]["tc"][0][i]]['thetaPhase']['mu'][()]
            mvl_theta[i] = marvin[marvin["clusters"]["tc"][0][i]]['thetaPhase']['mvl'][()]
            hd_tun[i,:] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['z'][()][0,:]
    return mvl_hd, mvl_si, mu_theta, mvl_theta,hd_tun

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


# In[ ]:


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


# In[ ]:



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
spikes_box = {}

i = 0
i0 = 0
for rat_name, rec_name, sess_name in (#('roger', 'rec1'),
                           ('roger', 'rec2', 'box_rec2'),
                           ('quentin', 'rec1', 'box'),
                           ('shane', 'rec1', 'box'),
                          ):
    
    tot_path = rat_name + '/' + rec_name + '/data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')    
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
        for mod_name in ('mod1', 'mod2', 'mod3'):
            if (rat_name == 'shane') & (mod_name in ('mod2','mod3')):
                continue
            if (rat_name == 'quentin') & (mod_name in ('mod3')):
                continue
            inds = get_inds(rat_name, mod_name, sess_name)
            i0 += len(inds)
            print(len(inds),i0)
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
        for mod_name in ('mod1', 'mod2', 'mod3'):
            if (rat_name == 'shane') & (mod_name in ('mod2','mod3')):
                continue
            if (rat_name == 'quentin') & (mod_name in ('mod3')):
                continue
            inds = get_inds(rat_name, mod_name, sess_name)
            i0 += len(inds)
            print(len(inds),i0)
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
spikes_sws = {}

i = 0
i0 = 0
for rat_name, rec_name, sess_name in (#('roger', 'rec1'),
                           ('roger', 'rec2', 'sws_rec2'),
                           ('quentin', 'rec1', 'sws'),
                           ('shane', 'rec1', 'sws'),
                          ):
    
    tot_path = rat_name + '/' + rec_name + '/data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')    
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
        for mod_name in ('mod1', 'mod2', 'mod3'):
            if (rat_name == 'shane') & (mod_name in ('mod2','mod3')):
                continue
            if (rat_name == 'quentin') & (mod_name in ('mod3')):
                continue
            inds = get_inds(rat_name, mod_name, sess_name)
            i0 += len(inds)
            print(len(inds),i0)
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
                spikes_sws[i] = np.array(sall)-min_time0
                i += 1
    else:
        min_time0, max_time0 = times_all[rat_name + '_sleep'][0]
        t = marvin['tracking']['t'][()][0,1:-1]
        times = np.where((t>=min_time0) & (t<max_time0))
        t = t[times]
        sw = times_all[rat_name + '_' + sess_name]
        for mod_name in ('mod1', 'mod2', 'mod3'):
            if (rat_name == 'shane') & (mod_name in ('mod2','mod3')):
                continue
            if (rat_name == 'quentin') & (mod_name in ('mod3')):
                continue
            inds = get_inds(rat_name, mod_name, sess_name)
            i0 += len(inds)
            print(len(inds),i0)
            for m in inds:
                s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
                s = s[(s>= min_time0) & (s< max_time0)]
                sall = []
                for ss in range(len(sw)):
                    min_time1 = sw[ss][0] #t[ssp[sw[ss-1]]-1]
                    max_time1 = sw[ss][1]
                    stemp = s[(s>= min_time1) & (s< max_time1)]
                    sall.extend(stemp)
                spikes_sws[i] = np.array(sall)-min_time0
                i += 1


# In[ ]:


acorr_One = get_isi_acorr(spikes_box, bLog = False, bOne = True, maxt = 0.2)
np.savez('acorr_One', acorr_One = acorr_One)


# In[ ]:


indsBU = np.array([])
mvl_hd = np.array([])
mvl_si = np.array([])
mvl_theta = np.array([])
mu_theta = np.array([])
hd_tuning =  np.zeros((1,60))
classname = {}
classcurr = 0
cc = 0
for rat_name, sess_name in (('roger', 'box_rec2'), 
                            ('quentin', 'box'), 
                            ('shane', 'box')):
    mvl_hd_tmp, mvl_si_tmp, mu_theta_tmp, mvl_theta_tmp, hd_tuning_tmp = get_mvl(rat_name, sess_name)
    cc1 = len(mvl_hd_tmp)
    for mod_name in ('mod1', 'mod2', 'mod3'):
        if (rat_name == 'shane') & (mod_name in ('mod2','mod3')):
            continue
        if (rat_name == 'quentin') & (mod_name in ('mod3')):
            continue
        inds = get_inds(rat_name, mod_name, sess_name)
        mvl_hd = np.concatenate((mvl_hd, mvl_hd_tmp[inds]))
        mvl_si = np.concatenate((mvl_si, mvl_si_tmp[inds]))
        mvl_theta = np.concatenate((mvl_theta, mvl_theta_tmp[inds]))
        mu_theta = np.concatenate((mu_theta, mu_theta_tmp[inds]))
        hd_tuning = np.concatenate((hd_tuning, hd_tuning_tmp[inds,:]),0)
        
        indspure = np.where(mvl_hd_tmp[inds]<0.3)[0]
        indsconj = np.where(mvl_hd_tmp[inds]>=0.3)[0]
        indsmod = np.zeros(len(inds))
        indsmod[indspure] = classcurr
        classname[classcurr] = rat_name + '_' + mod_name
        classcurr += 1
        indsmod[indsconj] = classcurr
        classname[classcurr] = rat_name + '_' + mod_name + '_conj'
        indsBU = np.concatenate((indsBU, indsmod))
        classcurr += 1
    cc += cc1
hd_tuning = hd_tuning[1:,:]


# In[ ]:


f = np.load('Data/rog1_inds.npz', allow_pickle = True)
indsmod1 = f['ind']
f.close()

indsBUmod1 = np.zeros_like(indsBU)
indsBUmod1[:] = -1
indsBUmod1[indsBU==0] = indsmod1
cname = {}
cname[0] = 'C1'
cname[1] = 'C2'
cname[2] = 'C3'
acorr_scaled_box = np.zeros(acorr_One.shape)
for i in range(len(acorr_scaled_box[:,0])):
    acorr_scaled_box[i,:] = acorr_One[i,:].astype(float).copy()/float(acorr_One[i,0])
acorr_scaled_box[:,0] = 0
acorr_s = gaussian_filter1d(acorr_scaled_box[:, :],sigma = 4, axis = 1)
metric = 'cosine'
inds_spec = np.arange(len(acorr_s[:,0]))

X1 = squareform(pdist(acorr_s, 'cosine'))

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
for i in np.unique(ind):
    print('')
    print('class', i,': N = ' + str(sum(ind==i)))
    for j in np.unique(indsBU):
        print(classname[j], round(np.sum((indsBU==j) & (ind==i))/sum((indsBU==j)),2))
        if j == 0:
            for k in range(3):
                print('   ',cname[k], 
                      round(np.sum((indsBU==j) & (ind==i) & (indsBUmod1==k))/sum((indsBUmod1==k)),2))
        if (j+1)%2 == 0:
            print('')
    acorrtmp = acorr_scaled_box[np.where(ind==i)[0],:].T
    acorrmean = acorrtmp.mean(1)
    acorrstd = 3*acorrtmp.std(1)
    plt.figure()
    plt.plot(acorrmean, lw = 5, c = 'b')
    plt.plot(acorrmean + acorrstd, ls = '--', lw = 2, c = 'r')
    plt.plot(acorrmean - acorrstd, ls = '--', lw = 2, c = 'r')
    plt.savefig('acorr_class' + str(i), bbox_inches='tight', pad_inches=0.1)
X3 = X1[np.argsort(ind),:].copy()
X3 = X3[:,np.argsort(ind)]
plt.imshow(X3, cmap = 'afmhot')
plt.colorbar()
plt.figure()
X4 = X1[:189,:]
X4 = X4[:,:189]
X4 = X4[np.argsort(ind[:189]),:].copy()
X4 = X4[:,np.argsort(ind[:189])]
plt.imshow(X4, cmap = 'afmhot')
plt.axis('off')
plt.colorbar()
plt.savefig('distance_R1')
plt.figure()
X4 = X1[:189,:]
X4 = X4[:,:189]
X4 = X4[np.argsort(ind[:189]),:].copy()
X4 = X4[:,np.argsort(ind[:189])]
plt.imshow(X4, cmap = 'afmhot')
plt.axis('off')
#plt.colorbar()
plt.savefig('distance_R1')
plt.savefig('distance_R1.pdf')
np.savez('autocorrelogram_classes_300621', classes = indsall_orig, indsall = indsBU, indsclasses = ind)
f = np.load('autocorrelogram_classes_300621.npz', allow_pickle = True)
indsclasses = f['indsclasses']
classes = f['classes']
indsall = f['indsall']
f.close()


# In[ ]:


### Guanella 

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage.filters as filt
arena_size = 50

arenaX = [0,arena_size]
arenaY = [0,arena_size]

## Initial position
Txx = [arenaX[1]/2]
Tyy = [arenaY[1]/2]

def conv(ang):
    x = np.cos(np.radians(ang)) 
    y = np.sin(np.radians(ang)) 
    return x , y

def random_navigation(length):
    thetaList = []

    theta = 90
    counter = 0
    lenght_counter = 0
    for i in range(length):
        lenght_counter += 1

        prevTheta = np.copy(theta)

        if( Txx[-1]<2 ): theta = np.random.randint(-85,85)

        if( Txx[-1]>arena_size-2 ): theta = np.random.randint(95,260)

        if( Tyy[-1]<2 ): theta = np.random.randint(10,170)

        if( Tyy[-1]>arena_size-2 ): theta = np.random.randint(190,350)


        Txx.append( Txx[-1]+conv(theta)[0] + np.random.uniform(-0.5,0.5) )
        Tyy.append( Tyy[-1]+conv(theta)[1] + np.random.uniform(-0.5,0.5)  )

        cx = abs( Txx[-1] - Txx[-2]  )
        cy = abs( Tyy[-1] - Tyy[-2]  )
        h = np.sqrt( cx**2 + cy**2  )
        counter+=h

        if(theta != prevTheta or i == length-1):
            thetaList.append( [prevTheta, conv(prevTheta)[0], conv(prevTheta)[1], counter]  )
            counter = 0
    
    plt.plot(Txx,Tyy, '-')
    plt.show()


random_navigation(5000)

Txx = np.array(Txx)
Tyy = np.array(Tyy)

class Grid():
    def __init__(self, gain):
        
        self.mm = 20
        self.nn = 20
        self.TAO = 0.9
        self.II = 0.3
        self.SIGMA = 0.24
        self.SIGMA2 = self.SIGMA**2
        self.TT = 0.05
        self.grid_gain = gain
        self.grid_layers = len(self.grid_gain)  
        self.grid_activity = np.random.uniform(0,1,(self.mm,self.nn,self.grid_layers))  
        self.distTri = self.buildTopology(self.mm,self.nn)


    def update(self, speedVector):

        self.speedVector = speedVector
        
        grid_ActTemp = []
        for jj in range(0,self.grid_layers):
            rrr = self.grid_gain[jj]*np.exp(1j*0)
            matWeights = self.updateWeight(self.distTri,rrr)
            activityVect = np.ravel(self.grid_activity[:,:,jj])
            activityVect = self.Bfunc(activityVect, matWeights)
            activityTemp = activityVect.reshape(self.mm,self.nn)
            activityTemp += self.TAO *( activityTemp/np.mean(activityTemp) - activityTemp)
            activityTemp[activityTemp<0] = 0

            self.grid_activity[:,:,jj] = (activityTemp-np.min(activityTemp))/(  np.max(activityTemp)-np.min(activityTemp)) * 30  ##Eq 2
                        

    def buildTopology(self,mm,nn):  # Build connectivity matrix     ### Eq 4
        mmm = (np.arange(mm)+(0.5/mm))/mm
        nnn = ((np.arange(nn)+(0.5/nn))/nn)*np.sqrt(3)/2
        xx,yy = np.meshgrid(mmm, nnn)
        posv = xx+1j * yy
        Sdist = [ 0+1j*0, -0.5+1j*np.sqrt(3)/2, -0.5+1j*(-np.sqrt(3)/2), 0.5+1j*np.sqrt(3)/2, 0.5+1j*(-np.sqrt(3)/2), -1+1j*0, 1+1j*0]      
        xx,yy = np.meshgrid( np.ravel(posv) , np.ravel(posv) )
        distmat = xx-yy
        for ii in range(len(Sdist)):
            aaa1 = abs(distmat)
            rrr = xx-yy + Sdist[ii]
            aaa2 = abs(rrr)
            iii = np.where(aaa2<aaa1)
            distmat[iii] = rrr[iii]
        return distmat.transpose()

    def updateWeight(self,topology,rrr): # Slight update on weights based on speed vector.
        matWeights = self.II * np.exp((-abs(topology-rrr*self.speedVector)**2)/self.SIGMA2) - self.TT   ## Eq 3
        return matWeights

    def Bfunc(self,activity, matWeights):  ## Eq 1
        activity += np.dot(activity,matWeights)
        return activity
# this produces grid cells with different scales. Change the list to just one scale for one module
scale = [0.04,0.05,0.06,0.07,0.08]
grid = Grid(scale)

log_grid_cells = []
for i in range(1,Txx.size)

    speedVector = (Txx[i]-Txx[i-1])+1j*(Tyy[i]-Tyy[i-1])

    grid.update(speedVector)
    log_grid_cells.append( grid.grid_activity.flatten()  )
    
log_grid_cells = np.array(log_grid_cells)
xx = np.copy(Txx[1:])
yy = np.copy(Tyy[1:])
dv_levels = [0,5,10]
plt.figure(figsize=(18,2.5))
ax = plt.axes()

dv_levels = 10
dv_start = 25
for cell_num in np.arange(dv_levels):

    celula = log_grid_cells[:,dv_start+cell_num]

    pos_spike_idx = np.where( celula > celula.max()*.9 )[0]

    
    plt.subplot(1,dv_levels,cell_num+1)
    plt.plot(xx,yy)
    plt.plot(   xx[pos_spike_idx] , yy[pos_spike_idx], 'or' )
    #ax.set_aspect('equal', 'box')


# In[ ]:


#### Couey
sess_name = 'roger_box'
f = np.load('Main\\Data\\tracking_' + sess_name + '.npz', allow_pickle = True)
xx = f['xx']
yy = f['yy']
f.close()

#xx,yy = xx[np.arange(0,15000,10)],yy[np.arange(0,15000,10)]
dxx =  xx[:100000]#gaussian_filter1d(xx,10)
dyy =  yy[:100000]#gaussian_filter1d(yy,10)
#dxx,dyy = dxx[np.arange(0,50000,1)],dyy[np.arange(0,50000,1)]
#num_steps = len(dxx)
speeds = np.zeros(len(dxx))
speeds[1:] =np.sqrt(np.square(dxx[1:]-dxx[:-1]) + np.square(dyy[1:]-dyy[:-1]))*1000
speeds[0] = speeds[1]
angs = np.zeros(len(dxx))
angs[1:] = np.arctan2(dyy[1:]-dyy[:-1],dxx[1:]-dxx[:-1])
angs[0] = angs[1]
posx = xx[np.arange(0,180000,2)]
posy = yy[np.arange(0,180000,2)]
post = np.arange(0,180000,2)/100
#xx,yy = xx[np.arange(0,15000,10)],yy[np.arange(0,15000,10)]
dxx =  xx[:100000]#gaussian_filter1d(xx,10)
dyy =  yy[:100000]#gaussian_filter1d(yy,10)
#dxx,dyy = dxx[np.arange(0,50000,1)],dyy[np.arange(0,50000,1)]
#num_steps = len(dxx)
speeds = np.zeros(len(dxx))
speeds[1:] =np.sqrt(np.square(dxx[1:]-dxx[:-1]) + np.square(dyy[1:]-dyy[:-1]))*1000
speeds[0] = speeds[1]
angs = np.zeros(len(dxx))
angs[1:] = np.arctan2(dyy[1:]-dyy[:-1],dxx[1:]-dxx[:-1])
angs[0] = angs[1]
posx = xx[np.arange(0,180000,2)]
posy = yy[np.arange(0,180000,2)]
post = np.arange(0,180000,2)/100
post = post[np.isfinite(posx)]
posy = posy[np.isfinite(posx)]
posx = posx[np.isfinite(posx)]
post = post[np.isfinite(posy)]
posx = posx[np.isfinite(posy)]
posy = posy[np.isfinite(posy)]
post *= 1000

side = max(max(posx)-min(posx), max(posy)-min(posy))
posx *= 1./side
posy *= 1./side
posx -= min(posx)
posy -= min(posy)

tnew = np.arange(0, 599000, 1)
posx = np.interp(tnew, post, posx)
posy = np.interp(tnew, post, posy)
post = tnew

#Get angles and velocities
angs = np.zeros(len(post))
angs[:-1] = np.arctan2(posy[1:]-posy[:-1], posx[1:]-posx[:-1])
angs[-1] = angs[-2]
speeds = 1000.*np.sqrt((posx[1:]-posx[:-1])**2+(posy[1:]-posy[:-1])**2)
nums = len(speeds)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from scipy import *
import scipy.io

numbumps = 4 #4 or 8
#matfile = '15030612+13_pos.mat'
#name = "15030612+13_bumps%02d" % (numbumps)

# parameters of the model
extinp = 1.
alpha = 0.15
ell = 2.
inh = -0.02
R  = 15.
Nx = 28
Ny = 44

if(numbumps==4):
    Nx*=2
if(numbumps==8):
    Nx*=2
    Ny*=2
NG=Nx*Ny 

### MAKE CONNECTIVITY WITH AN OFFSET RELATIVE TO PREFERRED DIRECTION
theta = zeros([Nx,Ny])
theta[0:Nx:2,0:Ny:2] = 0
theta[1:Nx:2,0:Ny:2] = 1
theta[0:Nx:2,1:Ny:2] = 2
theta[1:Nx:2,1:Ny:2] = 3
theta = 0.5*pi*theta
theta = ravel(theta)
xes = zeros([Nx,Ny])
yes = zeros([Nx,Ny])
for x in range(Nx):
    for y in range(Ny):
        xes[x,y] = x
        yes[x,y] = y
xes = ravel(xes)
yes = ravel(yes)
Rsqrd = R**2
W = zeros([NG,NG])
for xi in range(Nx):
    xdiffA = abs(xes-xi-ell*cos(theta))
    xdiffB = Nx-xdiffA
    xdiffA = xdiffA**2
    xdiffB = xdiffB**2
    for y in range(Ny):
        n = xi*Ny+y
        ydiffA = abs(yes-y-ell*sin(theta))
        ydiffB = Ny-ydiffA
        ydiffA = ydiffA**2
        ydiffB = ydiffB**2
        d = xdiffA+ydiffA
        W[d<Rsqrd,n] += inh
        d = xdiffB+ydiffA
        W[d<Rsqrd,n] += inh
        d = xdiffA+ydiffB
        W[d<Rsqrd,n] += inh
        d = xdiffB+ydiffB
        W[d<Rsqrd,n] += inh
xes=0
yes=0
N = NG

S = (rand(Nx*Ny) > 0.1)*1.0
for t in range(2000):
    S = S + 0.1*(-S + maximum((extinp+np.matmul(S,W)),0.))
    S[S<0.00001] = 0.

maxS = max(np.ravel(S))
minx = min([min(posy),min(posx)])
maxx = max([max(posy),max(posx)])
whichn = np.arange(len(S))#random.sample(np.arange(len(S)), 100)
nodes1 = np.zeros([len(whichn), nums])

fig = plt.figure(13)
plt.plot([],[],'-')
#print animation.writers.avail

for t in range(0, nums):
    S = S + 0.1*(-S + np.maximum((extinp+np.matmul(S,W)+alpha*speeds[t]*np.cos(angs[t]-theta)),0.))
    if(np.mod(t,10)==0):
        S[S<0.0001] = 0.
    nodes1[:,t] = S#[]#>0.3*maxS ##some fake spikes
    if(np.mod(t,5000)==0 and t>2):
        print('%2.2f percent done'%(float(t)*100/float(nums)))
from scipy.stats import binned_statistic_2d, pearsonr
num_neurons = len(nodes1[:,0])
inds = np.arange(num_neurons)
np.random.shuffle(inds)
for i in inds[:5]:
    mtot, x_edge, y_edge, circ = binned_statistic_2d(posx[:t],posy[:t], nodes1[i,:t], 
        statistic='mean', bins=50, range=None, expand_binnumbers=True)
    plt.figure()
    plt.imshow(mtot)


# In[ ]:



#### plot torus ###
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage import gaussian_filter1d
numpoints = 10000
rat_name = 'roger'
mod_name = 'mod2'
sess_name = 'box_rec2'

file_name = rat_name + '_' + mod_name + '_' + sess_name
f = np.load('Main/Results/Orig/' + file_name + '_decoding.npz', allow_pickle = True)
c1 = f['c11all']
c2 = f['c12all']
f.close()
sig = 10
c1 = np.arctan2(gaussian_filter1d(np.sin(c1),sig), gaussian_filter1d(np.cos(c1),sig))
c2 = np.arctan2(gaussian_filter1d(np.sin(c2),sig), gaussian_filter1d(np.cos(c2),sig))

numpoints = len(c1)
plt.hsv()
r1, r2 = 1.5, 1
x = np.zeros(numpoints)
y = np.zeros(numpoints)
z = np.zeros(numpoints)
for i in range(numpoints):
        x[i] = (r1 + r2*np.cos(c1[i]))*np.cos(c2[i]) 
        y[i] = (r1 + r2*np.cos(c1[i]))*np.sin(c2[i])  
        z[i] =  r2*np.sin(c1[i])

plt.hsv()
fig = plt.figure()
times = np.arange(0,len(c1), 5)
ax = Axes3D(fig)#c = c12all_orig0[times][ex],
ax.scatter(x[times],y[times],z[times], c = np.cos((c1[times]-0.3)%(2*np.pi)), cmap = 'viridis', alpha = 0.6, s = 1)
ax.axis('off')
ax.view_init(60, 80)
ax.set_zlim(-3,3)
#plt.savefig('torus_c1.svg')
plt.savefig('torus_c1')
plt.savefig('torus_c1.pdf')


# In[ ]:


########### load rat Q ##############
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
import scipy.io as sio
#tot_path = 'torusdata_2020-08-28/torusdata_2020-08-28/quentin/rec1/quentin_modall.mat'
tot_path = 'Main/quentin/rec1/quentin_mod_final.mat'

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

tot_path = 'Main/quentin/rec1/quentin_all.mat'
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

    
tot_path = 'Main/quentin/rec1/data_bendunn.mat'
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
#mind2  = mind2[(mvl_hd[mind2]>=0.2) & (mvl_hd[mind2]<0.3)]
#mind3  = mind3[(mvl_hd[mind3]>=0.2) & (mvl_hd[mind3]<0.3)]
mind22  = mind2[(mvl_hd[mind2]<0.3)]
mind33  = mind3[(mvl_hd[mind3]<0.3)]


tot_path = 'Main/quentin/rec1/data_bendunn.mat'
marvin = h5py.File(tot_path, 'r')
x = marvin['tracking']['x'][()][0,1:-1]
y = marvin['tracking']['y'][()][0,1:-1]
z = marvin['tracking']['z'][()][0,1:-1]
hd_azimuth = marvin['tracking']['hd_azimuth'][()][0,1:-1]
t = marvin['tracking']['t'][()][0,1:-1]


sleep_start1 = 9576
sleep_end1 = 18812
times = np.where((t>=sleep_start1) & (t<sleep_end1))
x_sleep1 = x[times]
y_sleep1 = y[times]
t_sleep1 = t[times]

maze_start1 = 18977
maze_end1 = 25355
times = np.where((t>=maze_start1) & (t<maze_end1))
x_maze1 = x[times]
y_maze1 = y[times]
t_maze1 = t[times]



sleep_start2 = 25403
sleep_end2 = 27007
times = np.where((t>=sleep_start2) & (t<sleep_end2))
x_sleep2 = x[times]
y_sleep2 = y[times]
t_sleep2 = t[times]

box_start1 = 27826
box_end1 = 31223
times = np.where((t>=box_start1) & (t<box_end1))
x_box1 = x[times]
y_box1 = y[times]
t_box1 = t[times]

#3,  sleep_box_2,        start=9576, end=18812   
#4,  linear_track_2,     start=18977, end=25355   
#5,  sleep_box_3,        start=25403, end=27007   
#6,  open_field_1,       start=27826, end=31223   


spikes_mod1_rem1 = {}
spikes_mod1_sws1 = {}
spikes_mod1_maze1 = {}
spikes_mod1_box1 = {}
spikes_mod1_rem2 = {}
spikes_mod1_sws2 = {}

for i,m in enumerate(mind2):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]<sleep_start2)):
        rem_start1 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end1 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod1_rem1[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]<sleep_start2)):
        sws_start1 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end1 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod1_sws1[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]>=sleep_start2)):
        rem_start1 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end1 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod1_rem2[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]>=sleep_start2)):
        sws_start1 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end1 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod1_sws2[i] = np.array(ss)

    spikes_mod1_maze1[i] = np.array(s[(s>= maze_start1) & (s< maze_end1)])
    spikes_mod1_box1[i] = np.array(s[(s>= box_start1) & (s< box_end1)])
    
spikes_mod2_rem1 = {}
spikes_mod2_sws1 = {}
spikes_mod2_maze1 = {}
spikes_mod2_box1 = {}
spikes_mod2_rem2 = {}
spikes_mod2_sws2 = {}

for i,m in enumerate(mind3):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]<sleep_start2)):
        rem_start1 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end1 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod2_rem1[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]<sleep_start2)):
        sws_start1 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end1 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod2_sws1[i] = np.array(ss)

    spikes_mod2_maze1[i] = np.array(s[(s>= maze_start1) & (s< maze_end1)])
    spikes_mod2_box1[i] = np.array(s[(s>= box_start1) & (s< box_end1)])
    
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]>=sleep_start2)):
        rem_start1 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end1 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod2_rem2[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]>=sleep_start2)):
        sws_start1 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end1 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod2_sws2[i] = np.array(ss)    

## mod 1 rem

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000


num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt

start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start<sleep_start2*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[start<sleep_start2*res]

sspikes_mod1_rem1 = np.zeros((1,len(spikes_mod1_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_rem1)))    
    for n in spikes_mod1_rem1:
        spike_times = np.array(spikes_mod1_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod1_rem1 = np.concatenate((sspikes_mod1_rem1, spikes_temp),0)
sspikes_mod1_rem1 = sspikes_mod1_rem1[1:,:]
sspikes_mod1_rem1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_rem1_bin = np.zeros((1,len(spikes_mod1_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod1_rem1)), dtype = int)    
    for n in spikes_mod1_rem1:
        spike_times = np.array(spikes_mod1_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod1_rem1_bin = np.concatenate((spikes_mod1_rem1_bin, spikes_temp),0)
spikes_mod1_rem1_bin = spikes_mod1_rem1_bin[1:,:]

## mod 1 sws


res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start<sleep_start2*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end<sleep_start2*res]

sspikes_mod1_sws1 = np.zeros((1,len(spikes_mod1_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_sws1)))    
    for n in spikes_mod1_sws1:
        spike_times = np.array(spikes_mod1_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod1_sws1 = np.concatenate((sspikes_mod1_sws1, spikes_temp),0)
sspikes_mod1_sws1 = sspikes_mod1_sws1[1:,:]
sspikes_mod1_sws1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_sws1_bin = np.zeros((1,len(spikes_mod1_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod1_sws1)), dtype = int)    
    for n in spikes_mod1_sws1:
        spike_times = np.array(spikes_mod1_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod1_sws1_bin = np.concatenate((spikes_mod1_sws1_bin, spikes_temp),0)
spikes_mod1_sws1_bin = spikes_mod1_sws1_bin[1:,:]

#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)

## mod 1 rem2 

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt

start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start>=sleep_start2*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[start>=sleep_start2*res]

sspikes_mod1_rem2 = np.zeros((1,len(spikes_mod1_rem2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_rem2)))    
    for n in spikes_mod1_rem2:
        spike_times = np.array(spikes_mod1_rem2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod1_rem2 = np.concatenate((sspikes_mod1_rem2, spikes_temp),0)
sspikes_mod1_rem2 = sspikes_mod1_rem2[1:,:]
sspikes_mod1_rem2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_rem2_bin = np.zeros((1,len(spikes_mod1_rem2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod1_rem2)), dtype = int)    
    for n in spikes_mod1_rem2:
        spike_times = np.array(spikes_mod1_rem2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod1_rem2_bin = np.concatenate((spikes_mod1_rem2_bin, spikes_temp),0)
spikes_mod1_rem2_bin = spikes_mod1_rem2_bin[1:,:]

#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)

## mod 1 sws 2

res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start>=sleep_start2*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end>=sleep_start2*res]

sspikes_mod1_sws2 = np.zeros((1,len(spikes_mod1_sws2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_sws2)))    
    for n in spikes_mod1_sws2:
        spike_times = np.array(spikes_mod1_sws2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod1_sws2 = np.concatenate((sspikes_mod1_sws2, spikes_temp),0)
sspikes_mod1_sws2 = sspikes_mod1_sws2[1:,:]
sspikes_mod1_sws2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_sws2_bin = np.zeros((1,len(spikes_mod1_sws2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod1_sws2)), dtype = int)    
    for n in spikes_mod1_sws2:
        spike_times = np.array(spikes_mod1_sws2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod1_sws2_bin = np.concatenate((spikes_mod1_sws2_bin, spikes_temp),0)
spikes_mod1_sws2_bin = spikes_mod1_sws2_bin[1:,:]

#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)

## mod 1 of

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod1_box1 = np.zeros((1,len(spikes_mod1_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_box1)))    
for n in spikes_mod1_box1:
    spike_times = np.array(spikes_mod1_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod1_box1 = np.concatenate((sspikes_mod1_box1, spikes_temp),0)
sspikes_mod1_box1 = sspikes_mod1_box1[1:,:]
sspikes_mod1_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_box1_bin = np.zeros((1,len(spikes_mod1_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod1_box1)), dtype = int)    
for n in spikes_mod1_box1:
    spike_times = np.array(spikes_mod1_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod1_box1_bin = np.concatenate((spikes_mod1_box1_bin, spikes_temp),0)
spikes_mod1_box1_bin = spikes_mod1_box1_bin[1:,:]
#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)

## mod 1 ww

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start1*res
max_time = maze_end1*res

sspikes_mod1_maze1 = np.zeros((1,len(spikes_mod1_maze1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_maze1)))    
for n in spikes_mod1_maze1:
    spike_times = np.array(spikes_mod1_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod1_maze1 = np.concatenate((sspikes_mod1_maze1, spikes_temp),0)
sspikes_mod1_maze1 = sspikes_mod1_maze1[1:,:]
sspikes_mod1_maze1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_maze1_bin = np.zeros((1,len(spikes_mod1_maze1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod1_maze1)), dtype = int)    
for n in spikes_mod1_maze1:
    spike_times = np.array(spikes_mod1_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod1_maze1_bin = np.concatenate((spikes_mod1_maze1_bin, spikes_temp),0)
spikes_mod1_maze1_bin = spikes_mod1_maze1_bin[1:,:]
#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)


## mod 2 rem

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start<sleep_start2*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[end<sleep_start2*res]

sspikes_mod2_rem1 = np.zeros((1,len(spikes_mod2_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod2_rem1)))    
    for n in spikes_mod2_rem1:
        spike_times = np.array(spikes_mod2_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod2_rem1 = np.concatenate((sspikes_mod2_rem1, spikes_temp),0)
sspikes_mod2_rem1 = sspikes_mod2_rem1[1:,:]
sspikes_mod2_rem1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod2_rem1_bin = np.zeros((1,len(spikes_mod2_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod2_rem1)), dtype = int)    
    for n in spikes_mod2_rem1:
        spike_times = np.array(spikes_mod2_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod2_rem1_bin = np.concatenate((spikes_mod2_rem1_bin, spikes_temp),0)
spikes_mod2_rem1_bin = spikes_mod2_rem1_bin[1:,:]

#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)


## mod2 sws
res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000


num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start<sleep_start2*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end<sleep_start2*res]

sspikes_mod2_sws1 = np.zeros((1,len(spikes_mod2_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod2_sws1)))    
    for n in spikes_mod2_sws1:
        spike_times = np.array(spikes_mod2_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod2_sws1 = np.concatenate((sspikes_mod2_sws1, spikes_temp),0)
sspikes_mod2_sws1 = sspikes_mod2_sws1[1:,:]
sspikes_mod2_sws1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod2_sws1_bin = np.zeros((1,len(spikes_mod2_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod2_sws1)), dtype = int)    
    for n in spikes_mod2_sws1:
        spike_times = np.array(spikes_mod2_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod2_sws1_bin = np.concatenate((spikes_mod2_sws1_bin, spikes_temp),0)
spikes_mod2_sws1_bin = spikes_mod2_sws1_bin[1:,:]

#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)

## mod2 rem 2

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt

start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start>=sleep_start2*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[start>=sleep_start2*res]

sspikes_mod2_rem2 = np.zeros((1,len(spikes_mod2_rem2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod2_rem2)))    
    for n in spikes_mod2_rem2:
        spike_times = np.array(spikes_mod2_rem2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod2_rem2 = np.concatenate((sspikes_mod2_rem2, spikes_temp),0)
sspikes_mod2_rem2 = sspikes_mod2_rem2[1:,:]
sspikes_mod2_rem2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod2_rem2_bin = np.zeros((1,len(spikes_mod2_rem2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod2_rem2)), dtype = int)    
    for n in spikes_mod2_rem2:
        spike_times = np.array(spikes_mod2_rem2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod2_rem2_bin = np.concatenate((spikes_mod2_rem2_bin, spikes_temp),0)
spikes_mod2_rem2_bin = spikes_mod2_rem2_bin[1:,:]

#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)

## Mod 2 sws 2

res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start>=sleep_start2*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end>=sleep_start2*res]

sspikes_mod2_sws2 = np.zeros((1,len(spikes_mod2_sws2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod2_sws2)))    
    for n in spikes_mod2_sws2:
        spike_times = np.array(spikes_mod2_sws2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod2_sws2 = np.concatenate((sspikes_mod2_sws2, spikes_temp),0)
sspikes_mod2_sws2 = sspikes_mod2_sws2[1:,:]
sspikes_mod2_sws2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod2_sws2_bin = np.zeros((1,len(spikes_mod2_sws2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod2_sws2)), dtype = int)    
    for n in spikes_mod2_sws2:
        spike_times = np.array(spikes_mod2_sws2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod2_sws2_bin = np.concatenate((spikes_mod2_sws2_bin, spikes_temp),0)
spikes_mod2_sws2_bin = spikes_mod2_sws2_bin[1:,:]

#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)

## MOD 2 OF
f = np.load('quentin_mod2_box_spk_times.npz')
spikes_mod2_box1 = f['spiketimes']
box_start1 = 27826
box_end1 = 31223

res = 100000
sigma = 5000
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod2_box1 = np.zeros((1,len(spikes_mod2_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod2_box1)))    
for n in spikes_mod2_box1:
    spike_times = np.array(spikes_mod2_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod2_box1 = np.concatenate((sspikes_mod2_box1, spikes_temp),0)
sspikes_mod2_box1 = sspikes_mod2_box1[1:,:]
sspikes_mod2_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod2_box1_bin = np.zeros((1,len(spikes_mod2_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod2_box1)), dtype = int)    
for n in spikes_mod2_box1:
    spike_times = np.array(spikes_mod2_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod2_box1_bin = np.concatenate((spikes_mod2_box1_bin, spikes_temp),0)
spikes_mod2_box1_bin = spikes_mod2_box1_bin[1:,:]
#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)

## MOD 2 WW

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start1*res
max_time = maze_end1*res

sspikes_mod2_maze1 = np.zeros((1,len(spikes_mod2_maze1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod2_maze1)))    
for n in spikes_mod2_maze1:
    spike_times = np.array(spikes_mod2_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod2_maze1 = np.concatenate((sspikes_mod2_maze1, spikes_temp),0)
sspikes_mod2_maze1 = sspikes_mod2_maze1[1:,:]
sspikes_mod2_maze1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod2_maze1_bin = np.zeros((1,len(spikes_mod2_maze1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod2_maze1)), dtype = int)    
for n in spikes_mod2_maze1:
    spike_times = np.array(spikes_mod2_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod2_maze1_bin = np.concatenate((spikes_mod2_maze1_bin, spikes_temp),0)
spikes_mod2_maze1_bin = spikes_mod2_maze1_bin[1:,:]
#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)

## TRAJECTORIES
## OF 
min_time = box_start1*res
max_time = box_end1*res
tt_box1 = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res

idt =  np.concatenate((np.digitize(t_box1[:-1], tt_box1[:])-1, [len(tt_box1)+1]))
idtt = np.digitize(np.arange(len(tt_box1)), idt)-1

idx = np.concatenate((np.unique(idtt), [np.max(idtt)+1]))
divisor = np.bincount(idtt)
steps = (1.0/divisor[divisor>0]) 
N = np.max(divisor)
ranges = np.multiply(np.arange(N)[np.newaxis,:], steps[:, np.newaxis])
ranges[ranges>=1] = np.nan

rangesx =x_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (x_box1[idx[1:]] - x_box1[idx[:-1]])[:, np.newaxis])
xx_box1 = rangesx[~np.isnan(ranges)] 

rangesy =y_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (y_box1[idx[1:]] - y_box1[idx[:-1]])[:, np.newaxis])
yy_box1 = rangesy[~np.isnan(ranges)] 


#np.savez('Data/tracking_' + spikes_name, azimuth = azimuth_, xx = xx_, yy = yy_, tt = tt_)

## WW

min_time = maze_start1*res
max_time = maze_end1*res
tt_maze1 = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res

idt =  np.concatenate((np.digitize(t_maze1[:-1], tt_maze1[:])-1, [len(tt_maze1)+1]))
idtt = np.digitize(np.arange(len(tt_maze1)), idt)-1

idx = np.concatenate((np.unique(idtt), [np.max(idtt)+1]))
divisor = np.bincount(idtt)
steps = (1.0/divisor[divisor>0]) 
N = np.max(divisor)
ranges = np.multiply(np.arange(N)[np.newaxis,:], steps[:, np.newaxis])
ranges[ranges>=1] = np.nan

rangesx =x_maze1[idx[:-1], np.newaxis] + np.multiply(ranges, (x_maze1[idx[1:]] - x_maze1[idx[:-1]])[:, np.newaxis])
xx_maze1 = rangesx[~np.isnan(ranges)] 

rangesy =y_maze1[idx[:-1], np.newaxis] + np.multiply(ranges, (y_maze1[idx[1:]] - y_maze1[idx[:-1]])[:, np.newaxis])
yy_maze1 = rangesy[~np.isnan(ranges)] 


np.savez('Main/Data/tracking_quentin_box', xx = xx_box1, yy = yy_box1, tt = tt_box1)
np.savez('Main/Data/tracking_quentin_maze', xx = xx_maze1, yy = yy_maze1, tt = tt_maze1)
np.savez('Main/Data/quentin_mod1_maze_spikes', spikes = spikes_mod1_maze1_bin, sspikes = sspikes_mod1_maze1)
np.savez('Main/Data/quentin_mod1_box_spikes', spikes = spikes_mod1_box1_bin, sspikes = sspikes_mod1_box1)
np.savez('Main/Data/quentin_mod1_rem_spikes', spikes = spikes_mod1_rem1_bin, sspikes = sspikes_mod1_rem1)
np.savez('Main/Data/quentin_mod1_sws_spikes', spikes = spikes_mod1_sws1_bin, sspikes = sspikes_mod1_sws1)
np.savez('Main/Data/quentin_mod1_sws2_spikes', spikes = spikes_mod1_sws2_bin, sspikes = sspikes_mod1_sws2)

np.savez('Main/Data/quentin_mod2_maze_spikes', spikes = spikes_mod2_maze1_bin, sspikes = sspikes_mod2_maze1)
np.savez('Main/Data/quentin_mod2_box_spikes', spikes = spikes_mod2_box1_bin, sspikes = sspikes_mod2_box1)
np.savez('Main/Data/quentin_mod2_rem_spikes', spikes = spikes_mod2_rem1_bin, sspikes = sspikes_mod2_rem1)
np.savez('Main/Data/quentin_mod2_sws_spikes', spikes = spikes_mod2_sws1_bin, sspikes = sspikes_mod2_sws1)
np.savez('Main/Data/quentin_mod2_sws2_spikes', spikes = spikes_mod2_sws2_bin, sspikes = sspikes_mod2_sws2)


# In[ ]:


import scipy.io as sio
tot_path = 'Main/shane/rec1/shane_mod_final.mat'

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

tot_path = 'Main/shane/rec1/shane_all.mat'
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

tot_path = 'Main/shane/rec1/data_bendunn.mat'
marvin = h5py.File(tot_path, 'r')
numn = len(marvin["clusters"]["tc"][0])
mvl_hd = np.zeros((numn))
for i in range(numn):
    if len(marvin[marvin["clusters"]["tc"][0][i]])==4:
        mvl_hd[i] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['mvl'][()]
mind2  = mind2[(mvl_hd[mind2]<0.3)]
mind2.shape
tot_path = 'Main/shane/rec1/data_bendunn.mat'

#tot_path = 'torusdata_2020-08-28/torusdata_2020-08-28/shane/rec1/data_bendunn.mat'
marvin = h5py.File(tot_path, 'r')
x = marvin['tracking']['x'][()][0,1:-1]
y = marvin['tracking']['y'][()][0,1:-1]
t = marvin['tracking']['t'][()][0,1:-1]
azimuth = marvin['tracking']['hd_azimuth'][()][0,1:-1]
box_start1 = 9939
box_end1 = 12363
times = np.where((t>=box_start1) & (t<box_end1))
x_box1 = x[times]
y_box1 = y[times]
t_box1 = t[times]
azimuth_box1 = azimuth[times]

maze_start1 = 13670
maze_end1 = 14847
times = np.where((t>=maze_start1) & (t<maze_end1))
x_maze1 = x[times]
y_maze1 = y[times]
t_maze1 = t[times]


sleep_start1 = 14942
sleep_end1 = 23133
times = np.where((t>=sleep_start1) & (t<sleep_end1))
x_sleep1 = x[times]
y_sleep1 = y[times]
t_sleep1 = t[times]

maze_start2 = 23186
maze_end2 = 24936
times = np.where((t>=maze_start2) & (t<maze_end2))
x_maze2 = x[times]
y_maze2 = y[times]
t_maze2 = t[times]

#3,  sleep_box_2,        start=9576, end=18812   
#4,  linear_track_2,     start=18977, end=25355   
#5,  sleep_box_3,        start=25403, end=27007   
#6,  open_field_1,       start=27826, end=31223   

spikes_mod1_rem1 = {}
spikes_mod1_sws1 = {}
spikes_mod1_maze1 = {}
spikes_mod1_box1 = {}
spikes_mod1_maze2 = {}

for i,m in enumerate(mind2):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]<=sleep_end1)):
        rem_start1 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end1 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod1_rem1[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]<=sleep_end1)):
        sws_start1 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end1 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod1_sws1[i] = np.array(ss)

    spikes_mod1_maze1[i] = np.array(s[(s>= maze_start1) & (s< maze_end1)])
                                           
#    spikes_mod1_maze1[i] = np.concatenate((np.array(s[(s>= maze_start1) & (s< maze_end1)]), 
#                                           np.array(s[(s>= maze_start2) & (s< maze_end2)])))
    spikes_mod1_box1[i] = np.array(s[(s>= box_start1) & (s< box_end1)])
    spikes_mod1_maze2[i] = np.array(s[(s>= maze_start2) & (s< maze_end2)])
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start<=sleep_end1*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[end<=sleep_end1*res]

sspikes_mod1_rem1 = np.zeros((1,len(spikes_mod1_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_rem1)))    
    for n in spikes_mod1_rem1:
        spike_times = np.array(spikes_mod1_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod1_rem1 = np.concatenate((sspikes_mod1_rem1, spikes_temp),0)
sspikes_mod1_rem1 = sspikes_mod1_rem1[1:,:]
sspikes_mod1_rem1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_rem1_bin = np.zeros((1,len(spikes_mod1_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod1_rem1)), dtype = int)    
    for n in spikes_mod1_rem1:
        spike_times = np.array(spikes_mod1_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod1_rem1_bin = np.concatenate((spikes_mod1_rem1_bin, spikes_temp),0)
spikes_mod1_rem1_bin = spikes_mod1_rem1_bin[1:,:]

#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)


res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start<sleep_end1*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end<=sleep_end1*res]

sspikes_mod1_sws1 = np.zeros((1,len(spikes_mod1_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_sws1)))    
    for n in spikes_mod1_sws1:
        spike_times = np.array(spikes_mod1_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod1_sws1 = np.concatenate((sspikes_mod1_sws1, spikes_temp),0)
sspikes_mod1_sws1 = sspikes_mod1_sws1[1:,:]
sspikes_mod1_sws1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_sws1_bin = np.zeros((1,len(spikes_mod1_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod1_sws1)), dtype = int)    
    for n in spikes_mod1_sws1:
        spike_times = np.array(spikes_mod1_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod1_sws1_bin = np.concatenate((spikes_mod1_sws1_bin, spikes_temp),0)
spikes_mod1_sws1_bin = spikes_mod1_sws1_bin[1:,:]

#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)


res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod1_box1 = np.zeros((1,len(spikes_mod1_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_box1)))    
for n in spikes_mod1_box1:
    spike_times = np.array(spikes_mod1_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod1_box1 = np.concatenate((sspikes_mod1_box1, spikes_temp),0)
sspikes_mod1_box1 = sspikes_mod1_box1[1:,:]
sspikes_mod1_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_box1_bin = np.zeros((1,len(spikes_mod1_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod1_box1)), dtype = int)    
for n in spikes_mod1_box1:
    spike_times = np.array(spikes_mod1_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod1_box1_bin = np.concatenate((spikes_mod1_box1_bin, spikes_temp),0)
spikes_mod1_box1_bin = spikes_mod1_box1_bin[1:,:]
#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)


res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start1*res
max_time = maze_end1*res

sspikes_mod1_maze1 = np.zeros((1,len(spikes_mod1_maze1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_maze1)))    
for n in spikes_mod1_maze1:
    spike_times = np.array(spikes_mod1_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod1_maze1 = np.concatenate((sspikes_mod1_maze1, spikes_temp),0)
sspikes_mod1_maze1 = sspikes_mod1_maze1[1:,:]
sspikes_mod1_maze1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_maze1_bin = np.zeros((1,len(spikes_mod1_maze1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod1_maze1)), dtype = int)    
for n in spikes_mod1_maze1:
    spike_times = np.array(spikes_mod1_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod1_maze1_bin = np.concatenate((spikes_mod1_maze1_bin, spikes_temp),0)
spikes_mod1_maze1_bin = spikes_mod1_maze1_bin[1:,:]
#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)


res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start2*res
max_time = maze_end2*res

sspikes_mod1_maze2 = np.zeros((1,len(spikes_mod1_maze2)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_maze2)))    
for n in spikes_mod1_maze2:
    spike_times = np.array(spikes_mod1_maze2[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod1_maze2 = np.concatenate((sspikes_mod1_maze2, spikes_temp),0)
sspikes_mod1_maze2 = sspikes_mod1_maze2[1:,:]
sspikes_mod1_maze2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_maze2_bin = np.zeros((1,len(spikes_mod1_maze2)))
spikes_temp = np.zeros((len(tt), len(spikes_mod1_maze2)), dtype = int)    
for n in spikes_mod1_maze2:
    spike_times = np.array(spikes_mod1_maze2[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod1_maze2_bin = np.concatenate((spikes_mod1_maze2_bin, spikes_temp),0)
spikes_mod1_maze2_bin = spikes_mod1_maze2_bin[1:,:]
#np.savez('Data/spikes_mod' + modnum + '_' + spikes_name, spikes = spikes, sspikes = sspikes)


min_time = maze_start1*res
max_time = maze_end1*res
tt_maze1 = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res

idt =  np.concatenate((np.digitize(t_maze1[:-1], tt_maze1[:])-1, [len(tt_maze1)+1]))
idtt = np.digitize(np.arange(len(tt_maze1)), idt)-1

idx = np.concatenate((np.unique(idtt), [np.max(idtt)+1]))
divisor = np.bincount(idtt)
steps = (1.0/divisor[divisor>0]) 
N = np.max(divisor)
ranges = np.multiply(np.arange(N)[np.newaxis,:], steps[:, np.newaxis])
ranges[ranges>=1] = np.nan

rangesx =x_maze1[idx[:-1], np.newaxis] + np.multiply(ranges, (x_maze1[idx[1:]] - x_maze1[idx[:-1]])[:, np.newaxis])
xx_maze1 = rangesx[~np.isnan(ranges)] 

rangesy =y_maze1[idx[:-1], np.newaxis] + np.multiply(ranges, (y_maze1[idx[1:]] - y_maze1[idx[:-1]])[:, np.newaxis])
yy_maze1 = rangesy[~np.isnan(ranges)] 
min_time = box_start1*res
max_time = box_end1*res
tt_box1 = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res

idt =  np.concatenate(([0], np.digitize(t_box1[1:-1], tt_box1[:])-1, [len(tt_box1)+1]))
idtt = np.digitize(np.arange(len(tt_box1)), idt)-1

idx = np.concatenate((np.unique(idtt), [np.max(idtt)+1]))
divisor = np.bincount(idtt)
steps = (1.0/divisor[divisor>0]) 
N = np.max(divisor)
ranges = np.multiply(np.arange(N)[np.newaxis,:], steps[:, np.newaxis])
ranges[ranges>=1] = np.nan

rangesx =x_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (x_box1[idx[1:]] - x_box1[idx[:-1]])[:, np.newaxis])
xx_box1 = rangesx[~np.isnan(ranges)] 

rangesy =y_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (y_box1[idx[1:]] - y_box1[idx[:-1]])[:, np.newaxis])
yy_box1 = rangesy[~np.isnan(ranges)] 

rangesa =azimuth_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (azimuth_box1[idx[1:]] - azimuth_box1[idx[:-1]])[:, np.newaxis])
aa_box1 = rangesa[~np.isnan(ranges)] 
min_time = maze_start2*res
max_time = maze_end2*res
tt_maze2 = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res

idt =  np.concatenate((np.digitize(t_maze2[:-1], tt_maze2[:])-1, [len(tt_maze2)+1]))
idtt = np.digitize(np.arange(len(tt_maze2)), idt)-1

idx = np.concatenate((np.unique(idtt), [np.max(idtt)+1]))
divisor = np.bincount(idtt)
steps = (1.0/divisor[divisor>0]) 
N = np.max(divisor)
ranges = np.multiply(np.arange(N)[np.newaxis,:], steps[:, np.newaxis])
ranges[ranges>=1] = np.nan

rangesx =x_maze2[idx[:-1], np.newaxis] + np.multiply(ranges, (x_maze2[idx[1:]] - x_maze2[idx[:-1]])[:, np.newaxis])
xx_maze2 = rangesx[~np.isnan(ranges)] 

rangesy =y_maze2[idx[:-1], np.newaxis] + np.multiply(ranges, (y_maze2[idx[1:]] - y_maze2[idx[:-1]])[:, np.newaxis])
yy_maze2 = rangesy[~np.isnan(ranges)] 

np.savez('Main/Data/tracking_shane_box', xx = xx_box1, yy = yy_box1, tt = tt_box1, aa = aa_box1)
np.savez('Data/tracking_shane_maze', xx = xx_maze1, yy = yy_maze1, tt = tt_maze1)
np.savez('Data/tracking_shane_maze2', xx = xx_maze2, yy = yy_maze2, tt = tt_maze2)

np.savez('Main/Data/shane_mod1_box_spikes_conj', spikes = spikes_mod1_box1_bin, sspikes = sspikes_mod1_box1)
np.savez('Main/Data/shane_mod1_maze2_spikes_conj', spikes = spikes_mod1_maze2_bin, sspikes = sspikes_mod1_maze2)
np.savez('Main/Data/shane_mod1_maze_spikes_conj', spikes = spikes_mod1_maze1_bin, sspikes = sspikes_mod1_maze1)
np.savez('Main/Data/shane_mod1_rem_spikes_conj', spikes = spikes_mod1_rem1_bin, sspikes = sspikes_mod1_rem1)
np.savez('Main/Data/shane_mod1_sws_spikes_conj', spikes = spikes_mod1_sws1_bin, sspikes = sspikes_mod1_sws1)


# In[ ]:


import scipy.io as sio
#tot_path = 'torusdata_2020-08-28/torusdata_2020-08-28/roger/rec2/roger_modall2.mat'
tot_path = 'Main/roger/rec2/roger_rec2_mod_final.mat'

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
    
tot_path = 'Main/roger/rec2/roger_all2.mat'
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

tot_path = 'Main/roger/rec2/data_bendunn.mat'
marvin = h5py.File(tot_path, 'r')
numn = len(marvin["clusters"]["tc"][0])
mvl_hd = np.zeros((numn))
for i in range(numn):
    if len(marvin[marvin["clusters"]["tc"][0][i]])==4:
        mvl_hd[i] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['mvl'][()]
mind2  = mind2[(mvl_hd[mind2]<0.3)]
mind3  = mind3[(mvl_hd[mind3]<0.3)]
mind4  = mind4[(mvl_hd[mind4]<0.3)]
print(mind2.shape, mind3.shape, mind4.shape)
tot_path = 'Main/roger/rec2/data_bendunn.mat'

marvin = h5py.File(tot_path, 'r')
x = marvin['tracking']['x'][()][0,1:-1]
y = marvin['tracking']['y'][()][0,1:-1]
t = marvin['tracking']['t'][()][0,1:-1]
z = marvin['tracking']['z'][()][0,1:-1]
azimuth = marvin['tracking']['hd_azimuth'][()][0,1:-1]

box_start1 = 10617
box_end1 = 13004
times = np.where((t>=box_start1) & (t<box_end1))
x_box1 = x[times]
y_box1 = y[times]
t_box1 = t[times]
z_box1 = z[times]
azimuth_box1 = azimuth[times]


sleep_start1 = 396
sleep_end1 = 9941
times = np.where((t>=sleep_start1) & (t<sleep_end1))
x_sleep1 = x[times]
y_sleep1 = y[times]
t_sleep1 = t[times]
z_sleep1 = z[times]
azimuth_sleep1 = azimuth[times]


sleep_start2 = 13143
sleep_end2 = 15973
times = np.where((t>=sleep_start2) & (t<sleep_end2))
x_sleep2 = x[times]
y_sleep2 = y[times]
t_sleep2 = t[times]
z_sleep2 = z[times]
azimuth_sleep2 = azimuth[times]


#1,  open_field_1,       start=7457, end=16045, valid_times=[7457,14778;14890,16045]
#2,  foraging_maze_1,    start=16925, end=20704, valid_times=[16925,18026;18183,20704]
#3,  foraging_maze_2,    start=20895, end=21640
#4,  sleep_box_1,        start=21799, end=23771
spikes_mod1_rem1 = {}
spikes_mod1_sws1 = {}
spikes_mod1_box1 = {}
#spikes_mod1_sleep1 = {}
spikes_mod1_rem2 = {}
spikes_mod1_sws2 = {}

for i,m in enumerate(mind2):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]<sleep_end2)):
        rem_start1 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end1 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod1_rem1[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]<sleep_end2)):
        sws_start1 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end1 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod1_sws1[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]>sleep_end1)):
        rem_start2 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end2 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod1_rem2[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]>sleep_end1)):
        sws_start2 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end2 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod1_sws2[i] = np.array(ss)

    spikes_mod1_box1[i] = np.array(s[(s>= box_start1) & (s< box_end1)])
    #spikes_mod1_sleep1[i] = np.array(s[(s>= sleep_start1) & (s< sleep_end1)])
    

spikes_mod3_rem1 = {}
spikes_mod3_sws1 = {}
spikes_mod3_box1 = {}
#spikes_mod3_sleep1 = {}
spikes_mod3_rem2 = {}
spikes_mod3_sws2 = {}

for i,m in enumerate(mind3):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]<sleep_end2)):
        rem_start1 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end1 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod3_rem1[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]<sleep_end2)):
        sws_start1 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end1 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod3_sws1[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]>sleep_end1)):
        rem_start2 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end2 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod3_rem2[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]>sleep_end1)):
        sws_start2 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end2 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod3_sws2[i] = np.array(ss)

    spikes_mod3_box1[i] = np.array(s[(s>= box_start1) & (s< box_end1)])
    #spikes_mod3_sleep1[i] = np.array(s[(s>= sleep_start1) & (s< sleep_end1)])
    

spikes_mod4_rem1 = {}
spikes_mod4_sws1 = {}
spikes_mod4_box1 = {}
#spikes_mod4_sleep1 = {}
spikes_mod4_rem2 = {}
spikes_mod4_sws2 = {}

for i,m in enumerate(mind4):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]<sleep_end2)):
        rem_start1 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end1 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod4_rem1[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]<sleep_end2)):
        sws_start1 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end1 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod4_sws1[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['rem'][()][0,:]>sleep_end1)):
        rem_start2 = marvin['sleepTimes']['rem'][()][0, r]
        rem_end2 = marvin['sleepTimes']['rem'][()][1, r]
        ss.extend(s[(s>= rem_start1) & (s< rem_end1)])
    spikes_mod4_rem2[i] = np.array(ss)
    
    ss = []
    for r in range(len(marvin['sleepTimes']['sws'][()][0,:]>sleep_end1)):
        sws_start2 = marvin['sleepTimes']['sws'][()][0, r]
        sws_end2 = marvin['sleepTimes']['sws'][()][1, r]
        ss.extend(s[(s>= sws_start1) & (s< sws_end1)])
    spikes_mod4_sws2[i] = np.array(ss)

    spikes_mod4_box1[i] = np.array(s[(s>= box_start1) & (s< box_end1)])
    #spikes_mod4_sleep1[i] = np.array(s[(s>= sleep_start1) & (s< sleep_end1)])
    

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start<sleep_end1*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[end<=sleep_end1*res]

sspikes_mod1_rem1 = np.zeros((1,len(spikes_mod1_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_rem1)))    
    for n in spikes_mod1_rem1:
        spike_times = np.array(spikes_mod1_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod1_rem1 = np.concatenate((sspikes_mod1_rem1, spikes_temp),0)
sspikes_mod1_rem1 = sspikes_mod1_rem1[1:,:]
sspikes_mod1_rem1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_rem1_bin = np.zeros((1,len(spikes_mod1_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod1_rem1)), dtype = int)    
    for n in spikes_mod1_rem1:
        spike_times = np.array(spikes_mod1_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod1_rem1_bin = np.concatenate((spikes_mod1_rem1_bin, spikes_temp),0)
spikes_mod1_rem1_bin = spikes_mod1_rem1_bin[1:,:]

res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start<sleep_end1*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end<=sleep_end1*res]

sspikes_mod1_sws1 = np.zeros((1,len(spikes_mod1_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_sws1)))    
    for n in spikes_mod1_sws1:
        spike_times = np.array(spikes_mod1_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod1_sws1 = np.concatenate((sspikes_mod1_sws1, spikes_temp),0)
sspikes_mod1_sws1 = sspikes_mod1_sws1[1:,:]
sspikes_mod1_sws1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_sws1_bin = np.zeros((1,len(spikes_mod1_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod1_sws1)), dtype = int)    
    for n in spikes_mod1_sws1:
        spike_times = np.array(spikes_mod1_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod1_sws1_bin = np.concatenate((spikes_mod1_sws1_bin, spikes_temp),0)
spikes_mod1_sws1_bin = spikes_mod1_sws1_bin[1:,:]

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start>sleep_end1*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[end>sleep_end1*res]

sspikes_mod1_rem2 = np.zeros((1,len(spikes_mod1_rem2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_rem2)))    
    for n in spikes_mod1_rem2:
        spike_times = np.array(spikes_mod1_rem2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod1_rem2 = np.concatenate((sspikes_mod1_rem2, spikes_temp),0)
sspikes_mod1_rem2 = sspikes_mod1_rem2[1:,:]
sspikes_mod1_rem2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_rem2_bin = np.zeros((1,len(spikes_mod1_rem2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod1_rem2)), dtype = int)    
    for n in spikes_mod1_rem2:
        spike_times = np.array(spikes_mod1_rem2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod1_rem2_bin = np.concatenate((spikes_mod1_rem2_bin, spikes_temp),0)
spikes_mod1_rem2_bin = spikes_mod1_rem2_bin[1:,:]

res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start>sleep_end1*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end>sleep_end1*res]

sspikes_mod1_sws2 = np.zeros((1,len(spikes_mod1_sws2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_sws2)))    
    for n in spikes_mod1_sws2:
        spike_times = np.array(spikes_mod1_sws2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod1_sws2 = np.concatenate((sspikes_mod1_sws2, spikes_temp),0)
sspikes_mod1_sws2 = sspikes_mod1_sws2[1:,:]
sspikes_mod1_sws2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_sws2_bin = np.zeros((1,len(spikes_mod1_sws2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod1_sws2)), dtype = int)    
    for n in spikes_mod1_sws2:
        spike_times = np.array(spikes_mod1_sws2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod1_sws2_bin = np.concatenate((spikes_mod1_sws2_bin, spikes_temp),0)
spikes_mod1_sws2_bin = spikes_mod1_sws2_bin[1:,:]

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod1_box1 = np.zeros((1,len(spikes_mod1_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_box1)))    
for n in spikes_mod1_box1:
    spike_times = np.array(spikes_mod1_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod1_box1 = np.concatenate((sspikes_mod1_box1, spikes_temp),0)
sspikes_mod1_box1 = sspikes_mod1_box1[1:,:]
sspikes_mod1_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_box1_bin = np.zeros((1,len(spikes_mod1_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod1_box1)), dtype = int)    
for n in spikes_mod1_box1:
    spike_times = np.array(spikes_mod1_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod1_box1_bin = np.concatenate((spikes_mod1_box1_bin, spikes_temp),0)
spikes_mod1_box1_bin = spikes_mod1_box1_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start<sleep_end1*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[end<=sleep_end1*res]

sspikes_mod3_rem1 = np.zeros((1,len(spikes_mod3_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod3_rem1)))    
    for n in spikes_mod3_rem1:
        spike_times = np.array(spikes_mod3_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod3_rem1 = np.concatenate((sspikes_mod3_rem1, spikes_temp),0)
sspikes_mod3_rem1 = sspikes_mod3_rem1[1:,:]
sspikes_mod3_rem1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod3_rem1_bin = np.zeros((1,len(spikes_mod3_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod3_rem1)), dtype = int)    
    for n in spikes_mod3_rem1:
        spike_times = np.array(spikes_mod3_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod3_rem1_bin = np.concatenate((spikes_mod3_rem1_bin, spikes_temp),0)
spikes_mod3_rem1_bin = spikes_mod3_rem1_bin[1:,:]

res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start<sleep_end1*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end<=sleep_end1*res]

sspikes_mod3_sws1 = np.zeros((1,len(spikes_mod3_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod3_sws1)))    
    for n in spikes_mod3_sws1:
        spike_times = np.array(spikes_mod3_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod3_sws1 = np.concatenate((sspikes_mod3_sws1, spikes_temp),0)
sspikes_mod3_sws1 = sspikes_mod3_sws1[1:,:]
sspikes_mod3_sws1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod3_sws1_bin = np.zeros((1,len(spikes_mod3_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod3_sws1)), dtype = int)    
    for n in spikes_mod3_sws1:
        spike_times = np.array(spikes_mod3_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod3_sws1_bin = np.concatenate((spikes_mod3_sws1_bin, spikes_temp),0)
spikes_mod3_sws1_bin = spikes_mod3_sws1_bin[1:,:]

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start>sleep_end1*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[end>sleep_end1*res]

sspikes_mod3_rem2 = np.zeros((1,len(spikes_mod3_rem2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod3_rem2)))    
    for n in spikes_mod3_rem2:
        spike_times = np.array(spikes_mod3_rem2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod3_rem2 = np.concatenate((sspikes_mod3_rem2, spikes_temp),0)
sspikes_mod3_rem2 = sspikes_mod3_rem2[1:,:]
sspikes_mod3_rem2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod3_rem2_bin = np.zeros((1,len(spikes_mod3_rem2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod3_rem2)), dtype = int)    
    for n in spikes_mod3_rem2:
        spike_times = np.array(spikes_mod3_rem2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod3_rem2_bin = np.concatenate((spikes_mod3_rem2_bin, spikes_temp),0)
spikes_mod3_rem2_bin = spikes_mod3_rem2_bin[1:,:]

res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start>sleep_end1*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end>sleep_end1*res]

sspikes_mod3_sws2 = np.zeros((1,len(spikes_mod3_sws2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod3_sws2)))    
    for n in spikes_mod3_sws2:
        spike_times = np.array(spikes_mod3_sws2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod3_sws2 = np.concatenate((sspikes_mod3_sws2, spikes_temp),0)
sspikes_mod3_sws2 = sspikes_mod3_sws2[1:,:]
sspikes_mod3_sws2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod3_sws2_bin = np.zeros((1,len(spikes_mod3_sws2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod3_sws2)), dtype = int)    
    for n in spikes_mod3_sws2:
        spike_times = np.array(spikes_mod3_sws2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod3_sws2_bin = np.concatenate((spikes_mod3_sws2_bin, spikes_temp),0)
spikes_mod3_sws2_bin = spikes_mod3_sws2_bin[1:,:]

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod3_box1 = np.zeros((1,len(spikes_mod3_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod3_box1)))    
for n in spikes_mod3_box1:
    spike_times = np.array(spikes_mod3_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod3_box1 = np.concatenate((sspikes_mod3_box1, spikes_temp),0)
sspikes_mod3_box1 = sspikes_mod3_box1[1:,:]
sspikes_mod3_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod3_box1_bin = np.zeros((1,len(spikes_mod3_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod3_box1)), dtype = int)    
for n in spikes_mod3_box1:
    spike_times = np.array(spikes_mod3_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod3_box1_bin = np.concatenate((spikes_mod3_box1_bin, spikes_temp),0)
spikes_mod3_box1_bin = spikes_mod3_box1_bin[1:,:]
res = 100000
sigma = 7000
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start<sleep_end1*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[end<=sleep_end1*res]

sspikes_mod4_rem1 = np.zeros((1,len(spikes_mod4_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod4_rem1)))    
    for n in spikes_mod4_rem1:
        spike_times = np.array(spikes_mod4_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod4_rem1 = np.concatenate((sspikes_mod4_rem1, spikes_temp),0)
sspikes_mod4_rem1 = sspikes_mod4_rem1[1:,:]
sspikes_mod4_rem1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod4_rem1_bin = np.zeros((1,len(spikes_mod4_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod4_rem1)), dtype = int)    
    for n in spikes_mod4_rem1:
        spike_times = np.array(spikes_mod4_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod4_rem1_bin = np.concatenate((spikes_mod4_rem1_bin, spikes_temp),0)
spikes_mod4_rem1_bin = spikes_mod4_rem1_bin[1:,:]

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start<sleep_end1*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[end<=sleep_end1*res]

sspikes_mod4_rem1 = np.zeros((1,len(spikes_mod4_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod4_rem1)))    
    for n in spikes_mod4_rem1:
        spike_times = np.array(spikes_mod4_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod4_rem1 = np.concatenate((sspikes_mod4_rem1, spikes_temp),0)
sspikes_mod4_rem1 = sspikes_mod4_rem1[1:,:]
sspikes_mod4_rem1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod4_rem1_bin = np.zeros((1,len(spikes_mod4_rem1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod4_rem1)), dtype = int)    
    for n in spikes_mod4_rem1:
        spike_times = np.array(spikes_mod4_rem1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod4_rem1_bin = np.concatenate((spikes_mod4_rem1_bin, spikes_temp),0)
spikes_mod4_rem1_bin = spikes_mod4_rem1_bin[1:,:]

res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start<sleep_end1*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end<=sleep_end1*res]

sspikes_mod4_sws1 = np.zeros((1,len(spikes_mod4_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod4_sws1)))    
    for n in spikes_mod4_sws1:
        spike_times = np.array(spikes_mod4_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod4_sws1 = np.concatenate((sspikes_mod4_sws1, spikes_temp),0)
sspikes_mod4_sws1 = sspikes_mod4_sws1[1:,:]
sspikes_mod4_sws1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod4_sws1_bin = np.zeros((1,len(spikes_mod4_sws1)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod4_sws1)), dtype = int)    
    for n in spikes_mod4_sws1:
        spike_times = np.array(spikes_mod4_sws1[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod4_sws1_bin = np.concatenate((spikes_mod4_sws1_bin, spikes_temp),0)
spikes_mod4_sws1_bin = spikes_mod4_sws1_bin[1:,:]

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['rem'][()][0,:]*res
start = start[start>sleep_end1*res]

end = marvin['sleepTimes']['rem'][()][1,:]*res
end = end[end>sleep_end1*res]

sspikes_mod4_rem2 = np.zeros((1,len(spikes_mod4_rem2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod4_rem2)))    
    for n in spikes_mod4_rem2:
        spike_times = np.array(spikes_mod4_rem2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod4_rem2 = np.concatenate((sspikes_mod4_rem2, spikes_temp),0)
sspikes_mod4_rem2 = sspikes_mod4_rem2[1:,:]
sspikes_mod4_rem2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod4_rem2_bin = np.zeros((1,len(spikes_mod4_rem2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod4_rem2)), dtype = int)    
    for n in spikes_mod4_rem2:
        spike_times = np.array(spikes_mod4_rem2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod4_rem2_bin = np.concatenate((spikes_mod4_rem2_bin, spikes_temp),0)
spikes_mod4_rem2_bin = spikes_mod4_rem2_bin[1:,:]

res = 100000
sigma = 2500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


start = marvin['sleepTimes']['sws'][()][0,:]*res
start = start[start>sleep_end1*res]

end = marvin['sleepTimes']['sws'][()][1,:]*res
end = end[end>sleep_end1*res]

sspikes_mod4_sws2 = np.zeros((1,len(spikes_mod4_sws2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod4_sws2)))    
    for n in spikes_mod4_sws2:
        spike_times = np.array(spikes_mod4_sws2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
    sspikes_mod4_sws2 = np.concatenate((sspikes_mod4_sws2, spikes_temp),0)
sspikes_mod4_sws2 = sspikes_mod4_sws2[1:,:]
sspikes_mod4_sws2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod4_sws2_bin = np.zeros((1,len(spikes_mod4_sws2)))
for r in range(len(start)):
    min_time = start[r]
    max_time = end[r]
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

    spikes_temp = np.zeros((len(tt), len(spikes_mod4_sws2)), dtype = int)    
    for n in spikes_mod4_sws2:
        spike_times = np.array(spikes_mod4_sws2[n]*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes_temp[j, n] += 1
    spikes_mod4_sws2_bin = np.concatenate((spikes_mod4_sws2_bin, spikes_temp),0)
spikes_mod4_sws2_bin = spikes_mod4_sws2_bin[1:,:]

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod4_box1 = np.zeros((1,len(spikes_mod4_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod4_box1)))    
for n in spikes_mod4_box1:
    spike_times = np.array(spikes_mod4_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod4_box1 = np.concatenate((sspikes_mod4_box1, spikes_temp),0)
sspikes_mod4_box1 = sspikes_mod4_box1[1:,:]
sspikes_mod4_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod4_box1_bin = np.zeros((1,len(spikes_mod4_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod4_box1)), dtype = int)    
for n in spikes_mod4_box1:
    spike_times = np.array(spikes_mod4_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod4_box1_bin = np.concatenate((spikes_mod4_box1_bin, spikes_temp),0)
spikes_mod4_box1_bin = spikes_mod4_box1_bin[1:,:]
min_time = box_start1*res
max_time = box_end1*res
tt_box1 = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res

idt =  np.concatenate(([0], np.digitize(t_box1[1:-1], tt_box1[:])-1, [len(tt_box1)+1]))
idtt = np.digitize(np.arange(len(tt_box1)), idt)-1

idx = np.concatenate((np.unique(idtt), [np.max(idtt)+1]))
divisor = np.bincount(idtt)
steps = (1.0/divisor[divisor>0]) 
N = np.max(divisor)
ranges = np.multiply(np.arange(N)[np.newaxis,:], steps[:, np.newaxis])
ranges[ranges>=1] = np.nan

rangesx =x_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (x_box1[idx[1:]] - x_box1[idx[:-1]])[:, np.newaxis])
xx_box1 = rangesx[~np.isnan(ranges)] 

rangesy =y_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (y_box1[idx[1:]] - y_box1[idx[:-1]])[:, np.newaxis])
yy_box1 = rangesy[~np.isnan(ranges)] 

rangesz =z_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (z_box1[idx[1:]] - z_box1[idx[:-1]])[:, np.newaxis])
zz_box1 = rangesz[~np.isnan(ranges)] 

rangesa =azimuth_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (azimuth_box1[idx[1:]] - azimuth_box1[idx[:-1]])[:, np.newaxis])
aa_box1 = rangesa[~np.isnan(ranges)] 
np.savez('Main/Data/tracking_roger_box_rec2', xx = xx_box1, yy = yy_box1, tt = tt_box1, zz = zz_box1, aa = aa_box1)
np.savez('Main/Data/spikes_mod1_roger_box_rec2_7', spikes = spikes_mod1_box1_bin, sspikes = sspikes_mod1_box1)
np.savez('Main/Data/spikes_mod1_roger_rem_rec2_7', spikes = spikes_mod1_rem1_bin, sspikes = sspikes_mod1_rem1)
np.savez('Main/Data/spikes_mod1_roger_sws_rec2_2', spikes = spikes_mod1_sws1_bin, sspikes = sspikes_mod1_sws1)
np.savez('Main/Data/spikes_mod2_roger_box_rec2_7', spikes = spikes_mod3_box1_bin, sspikes = sspikes_mod3_box1)
np.savez('Main/Data/spikes_mod2_roger_rem_rec2_7', spikes = spikes_mod3_rem1_bin, sspikes = sspikes_mod3_rem1)
np.savez('Main/Data/spikes_mod2_roger_sws_rec2_2', spikes = spikes_mod3_sws1_bin, sspikes = sspikes_mod3_sws1)
np.savez('Main/Data/spikes_mod3_roger_box_rec2_7', spikes = spikes_mod4_box1_bin, sspikes = sspikes_mod4_box1)
np.savez('Main/Data/spikes_mod3_roger_rem_rec2_7', spikes = spikes_mod4_rem1_bin, sspikes = sspikes_mod4_rem1)


# In[ ]:


import scipy.io as sio

#tot_path = 'torusdata_2020-08-28/torusdata_2020-08-28/roger/rec1/roger_modall1.mat'
tot_path = 'Main/roger/rec1/roger_mod_final.mat'

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

    
m5 = np.zeros(len(marvin['v6'][:,0]), dtype=int)
m55 = np.zeros(len(marvin['v6'][:,0]), dtype=int)
for i,m in enumerate(marvin['v6'][:,0]):
    m5[i] = int(m[0][0])
    m55[i] = int(m[0][2:])

    
tot_path = 'Main/roger/rec1/roger_all1.mat'
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

mind5 = np.zeros(len(m5), dtype = int)
for i in range(len(m5)):
    mind5[i] = np.where((mall==m5[i]) & (mall1==m55[i]))[0]
    
tot_path = 'Main/roger/rec1/data_bendunn.mat'
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
#mind2  = mind2[(mvl_hd[mind2]>=0.2) & (mvl_hd[mind2]<0.3)]
#mind3  = mind3[(mvl_hd[mind3]>=0.2) & (mvl_hd[mind3]<0.3)]
mind2  = mind2[(mvl_hd[mind2]<=0.3)]
mind3  = mind3[(mvl_hd[mind3]<=0.3)]
mind4  = mind4[(mvl_hd[mind4]<=0.3)]
mind5  = mind5[(mvl_hd[mind5]<=0.3)]


tot_path = 'Main/roger/rec1/data_bendunn.mat'
marvin = h5py.File(tot_path, 'r')

x = marvin['tracking']['x'][()][0,1:-1]
y = marvin['tracking']['y'][()][0,1:-1]
t = marvin['tracking']['t'][()][0,1:-1]
z = marvin['tracking']['z'][()][0,1:-1]
azimuth = marvin['tracking']['hd_azimuth'][()][0,1:-1]

box_start1 = 7457
box_end1 = 16045
times = np.where((t>=box_start1) & (t<box_end1))
x_box1 = x[times]
y_box1 = y[times]
t_box1 = t[times]
z_box1 = z[times]
azimuth_box1 = azimuth[times]


maze_start1 = 16925
maze_end1 = 20704
times = np.where((t>=maze_start1) & (t<maze_end1))
x_maze1 = x[times]
y_maze1 = y[times]
t_maze1 = t[times]
z_maze1 = z[times]
azimuth_maze1 = azimuth[times]

maze_start2 = 20895
maze_end2 = 21640
times = np.where((t>=maze_start2) & (t<maze_end2))
x_maze2 = x[times]
y_maze2 = y[times]
t_maze2 = t[times]
z_maze2 = z[times]
azimuth_maze2 = azimuth[times]


#1,  open_field_1,       start=7457, end=16045, valid_times=[7457,14778;14890,16045]
#2,  foraging_maze_1,    start=16925, end=20704, valid_times=[16925,18026;18183,20704]
#3,  foraging_maze_2,    start=20895, end=21640
#4,  sleep_box_1,        start=21799, end=23771
spikes_mod1_maze1 = {}
spikes_mod1_maze2 = {}
spikes_mod1_box1 = {}

for i,m in enumerate(np.concatenate((mind2,mind3))):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    spikes_mod1_maze1[i] = np.array(s[(s>= maze_start1) & (s< maze_end1)])
    spikes_mod1_maze2[i] = np.array(s[(s>= maze_start2) & (s< maze_end2)])
    spikes_mod1_box1[i] = np.array(s[(s>= box_start1) & (s< box_end1)])
    

spikes_mod3_maze1 = {}
spikes_mod3_maze2 = {}
spikes_mod3_box1 = {}

for i,m in enumerate(mind3):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    spikes_mod3_maze1[i] = np.array(s[(s>= maze_start1) & (s< maze_end1)])
    spikes_mod3_maze2[i] = np.array(s[(s>= maze_start2) & (s< maze_end2)])
    spikes_mod3_box1[i] = np.array(s[(s>= box_start1) & (s< box_end1)])
    

spikes_mod4_maze1 = {}
spikes_mod4_maze2 = {}
spikes_mod4_box1 = {}

for i,m in enumerate(mind4):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    spikes_mod4_maze1[i] = np.array(s[(s>= maze_start1) & (s< maze_end1)])
    spikes_mod4_maze2[i] = np.array(s[(s>= maze_start2) & (s< maze_end2)])
    spikes_mod4_box1[i] = np.array(s[(s>= box_start1) & (s< box_end1)])
    

spikes_mod5_maze1 = {}
spikes_mod5_maze2 = {}
spikes_mod5_box1 = {}

for i,m in enumerate(mind5):
    s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
    spikes_mod5_maze1[i] = np.array(s[(s>= maze_start1) & (s< maze_end1)])
    spikes_mod5_maze2[i] = np.array(s[(s>= maze_start2) & (s< maze_end2)])
    spikes_mod5_box1[i] = np.array(s[(s>= box_start1) & (s< box_end1)])
    

res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod1_box1 = np.zeros((1,len(spikes_mod1_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_box1)))    
for n in spikes_mod1_box1:
    spike_times = np.array(spikes_mod1_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod1_box1 = np.concatenate((sspikes_mod1_box1, spikes_temp),0)
sspikes_mod1_box1 = sspikes_mod1_box1[1:,:]
sspikes_mod1_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_box1_bin = np.zeros((1,len(spikes_mod1_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod1_box1)), dtype = int)    
for n in spikes_mod1_box1:
    spike_times = np.array(spikes_mod1_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod1_box1_bin = np.concatenate((spikes_mod1_box1_bin, spikes_temp),0)
spikes_mod1_box1_bin = spikes_mod1_box1_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start1*res
max_time = maze_end1*res

sspikes_mod1_maze1 = np.zeros((1,len(spikes_mod1_maze1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_maze1)))    
for n in spikes_mod1_maze1:
    spike_times = np.array(spikes_mod1_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod1_maze1 = np.concatenate((sspikes_mod1_maze1, spikes_temp),0)
sspikes_mod1_maze1 = sspikes_mod1_maze1[1:,:]
sspikes_mod1_maze1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_maze1_bin = np.zeros((1,len(spikes_mod1_maze1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod1_maze1)), dtype = int)    
for n in spikes_mod1_maze1:
    spike_times = np.array(spikes_mod1_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod1_maze1_bin = np.concatenate((spikes_mod1_maze1_bin, spikes_temp),0)
spikes_mod1_maze1_bin = spikes_mod1_maze1_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start2*res
max_time = maze_end2*res

sspikes_mod1_maze2 = np.zeros((1,len(spikes_mod1_maze2)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod1_maze2)))    
for n in spikes_mod1_maze2:
    spike_times = np.array(spikes_mod1_maze2[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod1_maze2 = np.concatenate((sspikes_mod1_maze2, spikes_temp),0)
sspikes_mod1_maze2 = sspikes_mod1_maze2[1:,:]
sspikes_mod1_maze2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod1_maze2_bin = np.zeros((1,len(spikes_mod1_maze2)))
spikes_temp = np.zeros((len(tt), len(spikes_mod1_maze2)), dtype = int)    
for n in spikes_mod1_maze2:
    spike_times = np.array(spikes_mod1_maze2[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod1_maze2_bin = np.concatenate((spikes_mod1_maze2_bin, spikes_temp),0)
spikes_mod1_maze2_bin = spikes_mod1_maze2_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod3_box1 = np.zeros((1,len(spikes_mod3_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod3_box1)))    
for n in spikes_mod3_box1:
    spike_times = np.array(spikes_mod3_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod3_box1 = np.concatenate((sspikes_mod3_box1, spikes_temp),0)
sspikes_mod3_box1 = sspikes_mod3_box1[1:,:]
sspikes_mod3_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod3_box1_bin = np.zeros((1,len(spikes_mod3_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod3_box1)), dtype = int)    
for n in spikes_mod3_box1:
    spike_times = np.array(spikes_mod3_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod3_box1_bin = np.concatenate((spikes_mod3_box1_bin, spikes_temp),0)
spikes_mod3_box1_bin = spikes_mod3_box1_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start1*res
max_time = maze_end1*res

sspikes_mod3_maze1 = np.zeros((1,len(spikes_mod3_maze1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod3_maze1)))    
for n in spikes_mod3_maze1:
    spike_times = np.array(spikes_mod3_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod3_maze1 = np.concatenate((sspikes_mod3_maze1, spikes_temp),0)
sspikes_mod3_maze1 = sspikes_mod3_maze1[1:,:]
sspikes_mod3_maze1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod3_maze1_bin = np.zeros((1,len(spikes_mod3_maze1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod3_maze1)), dtype = int)    
for n in spikes_mod3_maze1:
    spike_times = np.array(spikes_mod3_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod3_maze1_bin = np.concatenate((spikes_mod3_maze1_bin, spikes_temp),0)
spikes_mod3_maze1_bin = spikes_mod3_maze1_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start2*res
max_time = maze_end2*res

sspikes_mod3_maze2 = np.zeros((1,len(spikes_mod3_maze2)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod3_maze2)))    
for n in spikes_mod3_maze2:
    spike_times = np.array(spikes_mod3_maze2[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod3_maze2 = np.concatenate((sspikes_mod3_maze2, spikes_temp),0)
sspikes_mod3_maze2 = sspikes_mod3_maze2[1:,:]
sspikes_mod3_maze2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod3_maze2_bin = np.zeros((1,len(spikes_mod3_maze2)))
spikes_temp = np.zeros((len(tt), len(spikes_mod3_maze2)), dtype = int)    
for n in spikes_mod3_maze2:
    spike_times = np.array(spikes_mod3_maze2[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod3_maze2_bin = np.concatenate((spikes_mod3_maze2_bin, spikes_temp),0)
spikes_mod3_maze2_bin = spikes_mod3_maze2_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod4_box1 = np.zeros((1,len(spikes_mod4_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod4_box1)))    
for n in spikes_mod4_box1:
    spike_times = np.array(spikes_mod4_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod4_box1 = np.concatenate((sspikes_mod4_box1, spikes_temp),0)
sspikes_mod4_box1 = sspikes_mod4_box1[1:,:]
sspikes_mod4_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod4_box1_bin = np.zeros((1,len(spikes_mod4_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod4_box1)), dtype = int)    
for n in spikes_mod4_box1:
    spike_times = np.array(spikes_mod4_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod4_box1_bin = np.concatenate((spikes_mod4_box1_bin, spikes_temp),0)
spikes_mod4_box1_bin = spikes_mod4_box1_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start1*res
max_time = maze_end1*res

sspikes_mod4_maze1 = np.zeros((1,len(spikes_mod4_maze1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod4_maze1)))    
for n in spikes_mod4_maze1:
    spike_times = np.array(spikes_mod4_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod4_maze1 = np.concatenate((sspikes_mod4_maze1, spikes_temp),0)
sspikes_mod4_maze1 = sspikes_mod4_maze1[1:,:]
sspikes_mod4_maze1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod4_maze1_bin = np.zeros((1,len(spikes_mod4_maze1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod4_maze1)), dtype = int)    
for n in spikes_mod4_maze1:
    spike_times = np.array(spikes_mod4_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod4_maze1_bin = np.concatenate((spikes_mod4_maze1_bin, spikes_temp),0)
spikes_mod4_maze1_bin = spikes_mod4_maze1_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start2*res
max_time = maze_end2*res

sspikes_mod4_maze2 = np.zeros((1,len(spikes_mod4_maze2)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod4_maze2)))    
for n in spikes_mod4_maze2:
    spike_times = np.array(spikes_mod4_maze2[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod4_maze2 = np.concatenate((sspikes_mod4_maze2, spikes_temp),0)
sspikes_mod4_maze2 = sspikes_mod4_maze2[1:,:]
sspikes_mod4_maze2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod4_maze2_bin = np.zeros((1,len(spikes_mod4_maze2)))
spikes_temp = np.zeros((len(tt), len(spikes_mod4_maze2)), dtype = int)    
for n in spikes_mod4_maze2:
    spike_times = np.array(spikes_mod4_maze2[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod4_maze2_bin = np.concatenate((spikes_mod4_maze2_bin, spikes_temp),0)
spikes_mod4_maze2_bin = spikes_mod4_maze2_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod5_box1 = np.zeros((1,len(spikes_mod5_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod5_box1)))    
for n in spikes_mod5_box1:
    spike_times = np.array(spikes_mod5_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod5_box1 = np.concatenate((sspikes_mod5_box1, spikes_temp),0)
sspikes_mod5_box1 = sspikes_mod5_box1[1:,:]
sspikes_mod5_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod5_box1_bin = np.zeros((1,len(spikes_mod5_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod5_box1)), dtype = int)    
for n in spikes_mod5_box1:
    spike_times = np.array(spikes_mod5_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod5_box1_bin = np.concatenate((spikes_mod5_box1_bin, spikes_temp),0)
spikes_mod5_box1_bin = spikes_mod5_box1_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = box_start1*res
max_time = box_end1*res

sspikes_mod5_box1 = np.zeros((1,len(spikes_mod5_box1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod5_box1)))    
for n in spikes_mod5_box1:
    spike_times = np.array(spikes_mod5_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod5_box1 = np.concatenate((sspikes_mod5_box1, spikes_temp),0)
sspikes_mod5_box1 = sspikes_mod5_box1[1:,:]
sspikes_mod5_box1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod5_box1_bin = np.zeros((1,len(spikes_mod5_box1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod5_box1)), dtype = int)    
for n in spikes_mod5_box1:
    spike_times = np.array(spikes_mod5_box1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod5_box1_bin = np.concatenate((spikes_mod5_box1_bin, spikes_temp),0)
spikes_mod5_box1_bin = spikes_mod5_box1_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start1*res
max_time = maze_end1*res

sspikes_mod5_maze1 = np.zeros((1,len(spikes_mod5_maze1)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod5_maze1)))    
for n in spikes_mod5_maze1:
    spike_times = np.array(spikes_mod5_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod5_maze1 = np.concatenate((sspikes_mod5_maze1, spikes_temp),0)
sspikes_mod5_maze1 = sspikes_mod5_maze1[1:,:]
sspikes_mod5_maze1 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod5_maze1_bin = np.zeros((1,len(spikes_mod5_maze1)))
spikes_temp = np.zeros((len(tt), len(spikes_mod5_maze1)), dtype = int)    
for n in spikes_mod5_maze1:
    spike_times = np.array(spikes_mod5_maze1[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod5_maze1_bin = np.concatenate((spikes_mod5_maze1_bin, spikes_temp),0)
spikes_mod5_maze1_bin = spikes_mod5_maze1_bin[1:,:]
res = 100000
sigma = 7500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt


min_time = maze_start2*res
max_time = maze_end2*res

sspikes_mod5_maze2 = np.zeros((1,len(spikes_mod5_maze2)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes_mod5_maze2)))    
for n in spikes_mod5_maze2:
    spike_times = np.array(spikes_mod5_maze2[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
sspikes_mod5_maze2 = np.concatenate((sspikes_mod5_maze2, spikes_temp),0)
sspikes_mod5_maze2 = sspikes_mod5_maze2[1:,:]
sspikes_mod5_maze2 *= 1/np.sqrt(2*np.pi*(sigma/res)**2)

spikes_mod5_maze2_bin = np.zeros((1,len(spikes_mod5_maze2)))
spikes_temp = np.zeros((len(tt), len(spikes_mod5_maze2)), dtype = int)    
for n in spikes_mod5_maze2:
    spike_times = np.array(spikes_mod5_maze2[n]*res-min_time, dtype = int)
    spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
    spikes_mod = dt-spike_times%dt
    spike_times= np.array(spike_times/dt, int)
    for m, j in enumerate(spike_times):
        spikes_temp[j, n] += 1
spikes_mod5_maze2_bin = np.concatenate((spikes_mod5_maze2_bin, spikes_temp),0)
spikes_mod5_maze2_bin = spikes_mod5_maze2_bin[1:,:]
min_time = maze_start1*res
max_time = maze_end1*res
tt_maze1 = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res

idt =  np.concatenate(([0], np.digitize(t_maze1[1:-1], tt_maze1[:])-1, [len(tt_maze1)+1]))
idtt = np.digitize(np.arange(len(tt_maze1)), idt)-1

idx = np.concatenate((np.unique(idtt), [np.max(idtt)+1]))
divisor = np.bincount(idtt)
steps = (1.0/divisor[divisor>0]) 
N = np.max(divisor)
ranges = np.multiply(np.arange(N)[np.newaxis,:], steps[:, np.newaxis])
ranges[ranges>=1] = np.nan

rangesx =x_maze1[idx[:-1], np.newaxis] + np.multiply(ranges, (x_maze1[idx[1:]] - x_maze1[idx[:-1]])[:, np.newaxis])
xx_maze1 = rangesx[~np.isnan(ranges)] 

rangesy =y_maze1[idx[:-1], np.newaxis] + np.multiply(ranges, (y_maze1[idx[1:]] - y_maze1[idx[:-1]])[:, np.newaxis])
yy_maze1 = rangesy[~np.isnan(ranges)] 

rangesz =z_maze1[idx[:-1], np.newaxis] + np.multiply(ranges, (z_maze1[idx[1:]] - z_maze1[idx[:-1]])[:, np.newaxis])
zz_maze1 = rangesz[~np.isnan(ranges)] 

rangesa =azimuth_maze1[idx[:-1], np.newaxis] + np.multiply(ranges, (azimuth_maze1[idx[1:]] - azimuth_maze1[idx[:-1]])[:, np.newaxis])
aa_maze1 = rangesa[~np.isnan(ranges)] 
min_time = maze_start2*res
max_time = maze_end2*res
tt_maze2 = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res

idt =  np.concatenate(([0], np.digitize(t_maze2[1:-1], tt_maze2[:])-1, [len(tt_maze2)+1]))
idtt = np.digitize(np.arange(len(tt_maze2)), idt)-1

idx = np.concatenate((np.unique(idtt), [np.max(idtt)+1]))
divisor = np.bincount(idtt)
steps = (1.0/divisor[divisor>0]) 
N = np.max(divisor)
ranges = np.multiply(np.arange(N)[np.newaxis,:], steps[:, np.newaxis])
ranges[ranges>=1] = np.nan

rangesx =x_maze2[idx[:-1], np.newaxis] + np.multiply(ranges, (x_maze2[idx[1:]] - x_maze2[idx[:-1]])[:, np.newaxis])
xx_maze2 = rangesx[~np.isnan(ranges)] 

rangesy =y_maze2[idx[:-1], np.newaxis] + np.multiply(ranges, (y_maze2[idx[1:]] - y_maze2[idx[:-1]])[:, np.newaxis])
yy_maze2 = rangesy[~np.isnan(ranges)] 

rangesz =z_maze2[idx[:-1], np.newaxis] + np.multiply(ranges, (z_maze2[idx[1:]] - z_maze2[idx[:-1]])[:, np.newaxis])
zz_maze2 = rangesz[~np.isnan(ranges)] 

rangesa =azimuth_maze2[idx[:-1], np.newaxis] + np.multiply(ranges, (azimuth_maze2[idx[1:]] - azimuth_maze2[idx[:-1]])[:, np.newaxis])
aa_maze2 = rangesa[~np.isnan(ranges)]
min_time = box_start1*res
max_time = box_end1*res
tt_box1 = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res

idt =  np.concatenate(([0], np.digitize(t_box1[1:-1], tt_box1[:])-1, [len(tt_box1)+1]))
idtt = np.digitize(np.arange(len(tt_box1)), idt)-1

idx = np.concatenate((np.unique(idtt), [np.max(idtt)+1]))
divisor = np.bincount(idtt)
steps = (1.0/divisor[divisor>0]) 
N = np.max(divisor)
ranges = np.multiply(np.arange(N)[np.newaxis,:], steps[:, np.newaxis])
ranges[ranges>=1] = np.nan

rangesx =x_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (x_box1[idx[1:]] - x_box1[idx[:-1]])[:, np.newaxis])
xx_box1 = rangesx[~np.isnan(ranges)] 

rangesy =y_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (y_box1[idx[1:]] - y_box1[idx[:-1]])[:, np.newaxis])
yy_box1 = rangesy[~np.isnan(ranges)] 

rangesz =z_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (z_box1[idx[1:]] - z_box1[idx[:-1]])[:, np.newaxis])
zz_box1 = rangesz[~np.isnan(ranges)] 

rangesa =azimuth_box1[idx[:-1], np.newaxis] + np.multiply(ranges, (azimuth_box1[idx[1:]] - azimuth_box1[idx[:-1]])[:, np.newaxis])
aa_box1 = rangesa[~np.isnan(ranges)] 
np.savez('Main/Data/tracking_roger_box', xx = xx_box1, yy = yy_box1, tt = tt_box1, zz = zz_box1, aa = aa_box1)
np.savez('Main/Data/tracking_roger_maze', xx = xx_maze1, yy = yy_maze1, tt = tt_maze1, zz = zz_maze1, aa = aa_maze1)
np.savez('Main/Data/tracking_roger_maze2', xx = xx_maze2, yy = yy_maze2, tt = tt_maze2, zz = zz_maze2, aa = aa_maze2)
np.savez('Main/Data/spikes_mod1_roger_box_7', spikes = spikes_mod1_box1_bin, sspikes = sspikes_mod1_box1)
np.savez('Main/Data/spikes_mod1_roger_maze_7', spikes = spikes_mod1_maze1_bin, sspikes = sspikes_mod1_maze1)
np.savez('Main/Data/spikes_mod1_roger_maze2_7', spikes = spikes_mod1_maze2_bin, sspikes = sspikes_mod1_maze2)

np.savez('Main/Data/spikes_mod2_roger_box_7', spikes = spikes_mod3_box1_bin, sspikes = sspikes_mod3_box1)
np.savez('Main/Data/spikes_mod2_roger_maze_7', spikes = spikes_mod3_maze1_bin, sspikes = sspikes_mod3_maze1)
np.savez('Main/Data/spikes_mod2_roger_maze2_7', spikes = spikes_mod3_maze2_bin, sspikes = sspikes_mod3_maze2)



np.savez('Main/Data/spikes_mod3_roger_box_7', spikes = spikes_mod4_box1_bin, sspikes = sspikes_mod4_box1)
np.savez('Main/Data/spikes_mod3_roger_maze_7', spikes = spikes_mod4_maze1_bin, sspikes = sspikes_mod4_maze1)
np.savez('Main/Data/spikes_mod3_roger_maze2_7', spikes = spikes_mod4_maze2_bin, sspikes = sspikes_mod4_maze2)

np.savez('Main/Data/spikes_mod4_roger_box_7', spikes = spikes_mod5_box1_bin, sspikes = sspikes_mod5_box1)
np.savez('Main/Data/spikes_mod4_roger_maze_7', spikes = spikes_mod5_maze1_bin, sspikes = sspikes_mod5_maze1)
np.savez('Main/Data/spikes_mod4_roger_maze2_7', spikes = spikes_mod5_maze2_bin, sspikes = sspikes_mod5_maze2)
res = 100000
dt = 1000
valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                              np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)

f = np.load('Main/Data/roger_mod1_box_spikes_7_all.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod1_box_spikes_7_all', spikes = spikes, sspikes = sspikes)

f = np.load('Main/Data/roger_mod2_box_spikes_7_all.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod2_box_spikes_7_all', spikes = spikes, sspikes = sspikes)


valid_times = np.concatenate((np.arange(0, (18026-16925)*res/dt),
                              np.arange((18183-16925)*res/dt, (20704-16925)*res/dt))).astype(int)

f = np.load('Main/Data/roger_mod1_maze_spikes_7_all.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod1_maze_spikes_7_all', spikes = spikes, sspikes = sspikes)

f = np.load('Main/Data/roger_mod2_maze_spikes_7_all.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod2_maze_spikes_7_all', spikes = spikes, sspikes = sspikes)
res = 100000
dt = 1000
valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                              np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)

f = np.load('Main/Data/roger_mod1_box_spikes_65.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod1_box_spikes_65', spikes = spikes, sspikes = sspikes)

f = np.load('Main/Data/roger_mod2_box_spikes_65.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod2_box_spikes_65', spikes = spikes, sspikes = sspikes)

f = np.load('Main/Data/roger_mod3_box_spikes_65.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod3_box_spikes_65', spikes = spikes, sspikes = sspikes)

f = np.load('Main/Data/roger_mod4_box_spikes_65.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod4_box_spikes_65', spikes = spikes, sspikes = sspikes)

res = 100000
dt = 1000
valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                              np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)

f = np.load('Main/Data/roger_mod1_box_spikes_03.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod1_box_spikes_03', spikes = spikes, sspikes = sspikes)

f = np.load('Main/Data/roger_mod2_box_spikes_03.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod2_box_spikes_03', spikes = spikes, sspikes = sspikes)

f = np.load('Main/Data/roger_mod3_box_spikes_03.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod3_box_spikes_03', spikes = spikes, sspikes = sspikes)

f = np.load('Main/Data/roger_mod4_box_spikes_03.npz')
spikes = f['spikes'][valid_times] 
sspikes = f['sspikes'][valid_times] 
np.savez('Main/Data/roger_mod4_box_spikes_03', spikes = spikes, sspikes = sspikes)


# In[ ]:


######## Square Torus ################ 

num_points = 2500
sspikes = np.zeros((num_points,4))
xx = np.zeros(num_points)
yy = np.zeros(num_points)
rot_mat = []
x,y = np.meshgrid(np.linspace(0,1, 52)[1:-1], np.linspace(0,1, 52)[1:-1])
x,y = np.meshgrid(np.linspace(0,1, int(np.sqrt(num_points))+2)[1:-1],  
                  np.linspace(0,1, int(np.sqrt(num_points))+2)[1:-1])
x = x.flatten()*2*np.pi
y = y.flatten()*2*np.pi
for n in range(num_points):
    u,v = x[n],y[n]
    u += np.random.rand()*0.1
    v += np.random.rand()*0.1    
    f1 = np.cos(u)
    f2 = np.sin(u)
    f3 = np.cos(v)
    f4 = np.sin(v)
    sspikes[n,:] = (f1,f2,f3,f4)
    xx[n] = u
    yy[n] = v


dim = 6
ph_classes = [0,1] # Decode the ith most persistent cohomology class
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 1
coeff = 47
num_neurons = len(sspikes[0,:])
active_times = 15000
k = 1000
num_times = 5
n_points = 1200
nbs = 800

#times_cube = np.arange(0,len(sspikes[:,0]),num_times)
#movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube,:],1))[-active_times:])
#movetimes = times_cube[movetimes]

dim_red_spikes_move_scaled,__,__ = pca(preprocessing.scale(sspikes[:,:]), dim = dim)
indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                    n_points, 1, metric)
dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
X = squareform(pdist(dim_red_spikes_move_scaled, metric))
knn_indices, knn_dists, __ = nearest_neighbors(X, n_neighbors = nbs, metric = 'precomputed', angular=True, metric_kwds = {})
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

############ Compute persistent homology ################
rips_real = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=True, distance_matrix = True)
plot_diagrams(rips_real["dgms"], plot_only = np.arange(len(rips_real["dgms"])), lifetime = True)
diagrams = rips_real["dgms"] # the multiset describing the lives of the persistence classes
cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
dists_land = rips_real["dperm2all"] # the pairwise distance between the points 
births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
deaths1[np.isinf(deaths1)] = 0
lives1 = deaths1-births1 # the lifetime for the 1-dim classes
iMax = np.argsort(lives1)
coords1 = np.zeros((num_circ, len(indstemp)))
call = np.zeros((num_circ, len(sspikes[:,0])))
threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]])*dec_tresh
for c in ph_classes:
    cocycle = cocycles[iMax[-(c+1)]]
    coords1[c,:],inds = get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)


for c in ph_classes:
    cocycle = cocycles[iMax[-(c+1)]]
    coords1[c,:],inds = get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)
centcosall = np.zeros((num_neurons, 2, 1200))
centsinall = np.zeros((num_neurons, 2, 1200))

dspk = preprocessing.scale(sspikes[indstemp,:])#[movetimes[indstemp],:])

k = 1200
for neurid in range(num_neurons):
    spktemp = dspk[:, neurid].copy()
#    spktemp = spktemp/np.sum(np.abs(spktemp))
    centcosall[neurid,:,:] = np.multiply(np.cos(coords1[:, :]*2*np.pi),spktemp)
    centsinall[neurid,:,:] = np.multiply(np.sin(coords1[:, :]*2*np.pi),spktemp)

dspk = preprocessing.scale(sspikes)
a = np.zeros((len(sspikes[:,0]), 2, num_neurons))
for n in range(num_neurons):
    a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))

c = np.zeros((len(sspikes[:,0]), 2, num_neurons))
for n in range(num_neurons):
    c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))


mtot2 = np.sum(c,2)
mtot1 = np.sum(a,2)
coordsnew = np.arctan2(mtot2,mtot1)%(2*np.pi)
plt.viridis()
plt.figure()
ax = plt.axes()
ax.scatter(xx, yy, c = np.cos(coordsnew[:,0]))
ax.set_aspect('equal', 'box')
plt.axis('off')
plt.savefig('Main/Figs/decoding_sqrtor1.png')
plt.savefig('Main/Figs/decoding_sqrtor1.pdf')
plt.figure()
ax = plt.axes()
ax.scatter(xx, yy, c = np.cos(coordsnew[:,1]))
ax.set_aspect('equal', 'box')
plt.axis('off')

plt.savefig('Main/Figs/decoding_sqrtor2.png')
plt.savefig('Main/Figs/decoding_sqrtor2.pdf')


# In[ ]:


######## Hexagonal Torus ################ 

a_1 = 1
a_2 = 1
b_3 = 1
b_1 = 1/np.sqrt(3)
b_2 = -1/np.sqrt(3)
#b_1 = np.sqrt(3)/2
#b_2 = -np.sqrt(3)/2
C_1 = 1
C_2 = 1
C_3 = 1
num_points = 2500
sspikes = np.zeros((num_points,6))
xx = np.zeros(num_points)
yy = np.zeros(num_points)
rot_mat = []
#x,y = np.meshgrid(np.linspace(0,1, 52)[1:-1], np.linspace(0,1, 52)[1:-1])
x,y = np.meshgrid(np.linspace(0,1, int(np.sqrt(num_points))+1)[:-1],  
                  np.linspace(0,1, int(np.sqrt(num_points))+1))
x = x.flatten()*2*np.pi
y = y.flatten()*2*np.pi
for n in range(num_points):
    u,v = x[n],y[n]
    u += np.random.rand()*0.1
    v += np.random.rand()*0.1    
    f1 = C_1 *np.cos(a_1*u + b_1*v)
    f2 = C_1 *np.sin(a_1*u + b_1*v)
    f3 = C_2 *np.cos(a_2*u + b_2*v)
    f4 = C_2 *np.sin(a_2*u + b_2*v)
    f5 = C_3 *np.cos(b_3*v)
    f6 = C_3 *np.sin(b_3*v)
    sspikes[n,:] = (f1,f2,f3,f4,f5,f6)
    xx[n] = u
    yy[n] = v


dim = 6
ph_classes = [0,1] # Decode the ith most persistent cohomology class
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 1
coeff = 47
num_neurons = len(sspikes[0,:])
active_times = 15000
k = 1000
num_times = 5
n_points = 1200
nbs = 800

#times_cube = np.arange(0,len(sspikes[:,0]),num_times)
#movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube,:],1))[-active_times:])
#movetimes = times_cube[movetimes]

dim_red_spikes_move_scaled,e1,e2 = pca(preprocessing.scale(sspikes[:,:]), dim = dim)

indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                    n_points, 1, metric)
dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
X = squareform(pdist(dim_red_spikes_move_scaled, metric))
knn_indices, knn_dists, __ = nearest_neighbors(X, n_neighbors = nbs, metric = 'precomputed', angular=True, metric_kwds = {})
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

############ Compute persistent homology ################
rips_real = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=True, distance_matrix = True)
plt.figure()
plot_diagrams(rips_real["dgms"], plot_only = np.arange(len(rips_real["dgms"])), lifetime = True)
diagrams = rips_real["dgms"] # the multiset describing the lives of the persistence classes
cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
dists_land = rips_real["dperm2all"] # the pairwise distance between the points 
births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
deaths1[np.isinf(deaths1)] = 0
lives1 = deaths1-births1 # the lifetime for the 1-dim classes
iMax = np.argsort(lives1)
coords1 = np.zeros((num_circ, len(indstemp)))
call = np.zeros((num_circ, len(sspikes[:,0])))
threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]])*dec_tresh

for c in ph_classes:
    cocycle = cocycles[iMax[-(c+1)]]
    coords1[c,:],inds = get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)
centcosall = np.zeros((num_neurons, 2, 1200))
centsinall = np.zeros((num_neurons, 2, 1200))

dspk = preprocessing.scale(sspikes[indstemp,:])#[movetimes[indstemp],:])

k = 1200
for neurid in range(num_neurons):
    spktemp = dspk[:, neurid].copy()
#    spktemp = spktemp/np.sum(np.abs(spktemp))
    centcosall[neurid,:,:] = np.multiply(np.cos(coords1[:, :]*2*np.pi),spktemp)
    centsinall[neurid,:,:] = np.multiply(np.sin(coords1[:, :]*2*np.pi),spktemp)

dspk = preprocessing.scale(sspikes)
a = np.zeros((len(sspikes[:,0]), 2, num_neurons))
for n in range(num_neurons):
    a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))

c = np.zeros((len(sspikes[:,0]), 2, num_neurons))
for n in range(num_neurons):
    c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))


mtot2 = np.sum(c,2)
mtot1 = np.sum(a,2)
coordsnew = np.arctan2(mtot2,mtot1)%(2*np.pi)
plt.viridis()
plt.figure()
ax = plt.axes()
ax.scatter(xx, yy, c = np.cos(coordsnew[:,0]))
ax.set_aspect('equal', 'box')
plt.axis('off')
plt.savefig('Main/Figs/decoding_hextor1.png')
plt.savefig('Main/Figs/decoding_hextor1.pdf')
plt.figure()
ax = plt.axes()
ax.scatter(xx, yy, c = np.cos(coordsnew[:,1]))
ax.set_aspect('equal', 'box')
plt.axis('off')

plt.savefig('Main/Figs/decoding_hextor2.png')
plt.savefig('Main/Figs/decoding_hextor2.pdf')


ax = plt.axes()
ax.scatter(xx, yy, c = np.cos(coordsnew[:,0] - 1*np.pi/3*coordsnew[:,1]))

ax.set_aspect('equal', 'box')
plt.axis('off')

from Main.utils import smooth_tuning_map
t = (np.concatenate(((coordsnew[:,0:1] - np.pi/3*coordsnew[:,1:2])%(2*np.pi),coordsnew[:,1:2]),1)+[1/2*np.pi,np.pi])%(2*np.pi)
spktemp = np.exp(-np.sum(np.square(np.arctan2(np.sin(t), 
                                              np.cos(t))),1))

mtot1, x_edge, y_edge, circ = binned_statistic_2d(coordsnew[:,0],coordsnew[:,1], 
    spktemp, statistic='mean', bins=51-1, range=None, expand_binnumbers=True)
nans = np.isnan(mtot1) 
mtot1[nans] = 0        
mtot1 =  smooth_tuning_map(np.rot90(mtot1),51,5, bClose = False)
#mtot1 =  smooth_image(np.rot90(mtot1,1),5)
#smooth_image
plt.figure()

plt.viridis()
ax = plt.axes()
plt.axis('off')

ax.imshow(mtot1,origin = 'lower',extent = [0,2*np.pi,0, 2*np.pi])
r_box = transforms.Affine2D().skew_deg(15,15)
for x in ax.images + ax.lines + ax.collections:
    trans = x.get_transform()
    x.set_transform(r_box+trans) 
    if isinstance(x, PathCollection):
        transoff = x.get_offset_transform()
        x._transOffset = r_box+transoff     
ax.set_xlim(0, 2*np.pi+ 3*np.pi/5)
ax.set_ylim(0, 2*np.pi+ 3*np.pi/5)
#ax.set_aspect('equal', 'box') 
ax.axis('off')

plt.savefig('Main/Figs/hexneuron.png')
plt.savefig('Main/Figs/hexneuron.pdf')


# In[ ]:


############## Num peaks #############


from scipy import signal, fftpack, special
from scipy.ndimage import filters, measurements, interpolation
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from matplotlib import pyplot
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import scipy as sp
import scipy.ndimage

_SQRT2 = np.sqrt(2.0)
_SQRT3 = np.sqrt(3.0)

npoints = 1000
#sigs = [0., 0.5, 1,2,3,4,5, 7.5, 10, 15]
sigs = [0, 1, 2, 3, 4, 5, 6,7,8,9,10]
num_peaks_all7 = {}
cs = np.array(['w','g','r','c','m','y','b','k',])[:, np.newaxis].repeat(5,1).T.flatten()
for rat_name, mod_name, sess_name in (('roger', 'mod3', 'box_rec2'),
                                        ('roger', 'mod1', 'box'),
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
    print(rat_name, mod_name, sess_name)
    if (sess_name in ('sws', 'rem')) & (rat_name == 'roger') :        
        spikes = load_spikes(rat_name, mod_name, sess_name + '_rec2', bSmooth = False, bSpeed = True)
    else:
        spikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bSpeed = True)
    if (sess_name == 'sws') & (rat_name == 'roger') & (mod_name == 'mod1'):        
        file_name = rat_name + '_' + mod_name + '_' + 'sws_c0' + '_' + 'box_rec2'
        f = np.load('Results/Orig/' + file_name + '_alignment_dec.npz', allow_pickle = True)
        coords2 = f['csess']
        f.close()
        file_name = rat_name + '_' + mod_name + '_' + 'sws_class1'
        f = np.load('Data/' + file_name + '_newdecoding.npz',  allow_pickle = True)
        times = f['times']
        f.close()

    else:
        
        file_name = rat_name + '_' + mod_name + '_' + sess_name + '_' + 'box'
        if (sess_name in ('sws', 'rem', 'box_rec2')) & (rat_name == 'roger') :        
            file_name += '_rec2'
        f = np.load('Results/Orig/' + file_name + '_alignment_dec.npz', allow_pickle = True)
        coords2 = f['csess']
        f.close()
        file_name = rat_name + '_' + mod_name + '_' + sess_name
        if (sess_name in ('sws', 'rem', )) & (rat_name == 'roger') :        
            file_name += '_rec2'
        f = np.load('Data/' + file_name + '_newdecoding.npz',  allow_pickle = True)
        times = f['times']
        f.close()

    numbins = 50
    bins = np.linspace(0,2*np.pi, numbins+1)
    num_neurons = len(spikes[0,:])
    numangsint = 51
    numangsint_1 = numangsint-1
    mid = int((numangsint_1)/2)
    num_peaks = np.zeros((num_neurons, len(sigs)))
    for nn in np.arange(num_neurons):
        mtot = binned_statistic_2d(coords2[:,0], coords2[:,1], spikes[times,nn],
                                   bins = bins, range = None, expand_binnumbers = True)[0]
        mtot[np.isnan(mtot)] = np.min(mtot[~np.isnan(mtot)])
        mtot = np.rot90(mtot,1)

        indstemp1 = np.zeros((numangsint_1,numangsint_1), dtype=int)
        indstemp1[indstemp1==0] = np.arange((numangsint_1)**2)
        indstemp1temp = indstemp1.copy()
        mtemp1_3 = mtot.copy()
        for j in range(numangsint_1):
            mtemp1_3[j,:] = np.roll(mtemp1_3[j,:],int(j/2))

        mtot_out = np.zeros_like(mtot)
        mtemp1_4 = np.concatenate((mtemp1_3, mtemp1_3, mtemp1_3),1)
        mtemp1_5 = np.zeros_like(mtemp1_4)
        mtemp1_5[:, :mid] = mtemp1_4[:, (numangsint_1)*3-mid:]  
        mtemp1_5[:, mid:] = mtemp1_4[:,:(numangsint_1)*3-mid]  
        mtot = np.concatenate((mtemp1_5,mtemp1_4,mtemp1_5))

        for s, sig in enumerate(sigs):
            data = gaussian_filter(mtot, sigma = sig, mode = 'constant')
            datatmp = data.copy().flatten()
            datatmpBU = datatmp.copy()
            datainds = np.indices(np.shape(data)).flatten().reshape((2,data.shape[0]**2)).T
            datacount = np.zeros(len(datatmp))
            XY = np.zeros((npoints,2))
            for k in range(npoints):
                ktemp = np.argmax(datatmp)
                XY[k,:] = datainds[ktemp,:]
                datacount[ktemp] += 1
                datatmp[ktemp] = datatmpBU[ktemp]/datacount[ktemp]

            X = squareform(pdist(XY, 'euclidean'))
            thresh = 5
            X[X>thresh] = -1
            knn_indices = []
            knn_dists  =[]
            F = np.zeros(npoints)
            for i in range(npoints):
                indsvals = np.where(X[i,:]>=0)[0]
                knn_indices.append(indsvals)
                knn_dists.append(X[i,indsvals])
                F[i] = np.sum(np.exp(knn_dists[i]))
            i = np.argmax(F)
            inds_all = np.arange(len(XY[:,0]))
            classes = np.zeros(len(XY[:,0]))
            classcurr = 1
            inds_left = inds_all>-1
            inds_left[i] = False
#            inds_left[knn_indices[i]] = False
            classes[i] = classcurr
            classes[knn_indices[i]] = classcurr
            for j in range(npoints-1):
                F[knn_indices[i]] -= knn_dists[i]
                Fmax = np.argmax(F[inds_left])
                i = inds_all[inds_left][Fmax]
                inds_left[i] = False
#                inds_left[knn_indices[i]] = False
                if classes[i] == 0:
                    classcurr += 1
                    classes[i] = classcurr
                    classes[knn_indices[i]] = classcurr
                else:
                    classes[knn_indices[i]] = classes[i]
            ind = classes.astype(int) -1
            binstacked = np.bincount(ind)
            indd = np.unique(ind)
                        
            inds1 = ((XY[:,0]>=numangsint_1) & 
                    (XY[:,0]<2*numangsint_1) & 
                    (XY[:,1]>=numangsint_1) & 
                    (XY[:,1]<2*numangsint_1))

            inds2 = ((XY[:,0]>=1/2*numangsint_1) & 
                    (XY[:,0]<3/2*numangsint_1) & 
                    (XY[:,1]>=numangsint_1) & 
                    (XY[:,1]<2*numangsint_1))

            inds3 = ((XY[:,0]>=3/2*numangsint_1) & 
                    (XY[:,0]<5/2*numangsint_1) & 
                    (XY[:,1]>=numangsint_1) & 
                    (XY[:,1]<2*numangsint_1))

            inds4 = ((XY[:,0]>=numangsint_1) & 
                    (XY[:,0]<2*numangsint_1) & 
                    (XY[:,1]>=1/2*numangsint_1) & 
                    (XY[:,1]<3/2*numangsint_1))

            inds5 = ((XY[:,0]>=numangsint_1) & 
                    (XY[:,0]<2*numangsint_1) & 
                    (XY[:,1]>=3/2*numangsint_1) & 
                    (XY[:,1]<5/2*numangsint_1))
            nump = []
            for inds in [inds1,inds2,inds3,inds4,inds5]:
                bintemp = np.zeros(max(ind)+1)
                ind1 = ind[inds]
                inddd = np.unique(ind1)
                bintemp[inddd] = np.bincount(ind1)[inddd]
                indsfinal = np.where(np.divide(bintemp, binstacked)>0.5)[0]
                nump.append(len(np.unique(indsfinal)))
            
            num_peaks[nn, s] = np.max(nump)
    num_peaks_all6[rat_name + '_' + mod_name + '_' + sess_name] = num_peaks


# In[ ]:


fig = plt.figure()
ax1 = fig.add_subplot(111)
sigs = [0, 1, 2, 3, 4, 5, 6,7,8,9,10]
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
    num_peaks = num_peaks_all6[rat_name + '_' + mod_name + '_' + sess_name].copy()
#    num_peaks[num_peaks==0] = 1
    num_neurons = np.shape(num_peaks)[0]
    print(rat_name + '_' + mod_name + '_' + sess_name)
    print(np.sum(num_peaks[:,4]==1,0)/num_neurons)
    ax1.plot(sigs, np.sum(num_peaks[:,:]==1,0)/num_neurons)
ax1.set_aspect(1.0/ax1.get_data_ratio())
plt.plot([2.75, 2.75], [0,1], c = 'r', ls = '--')
#fig.savefig('Figs_review/num_neurons_1field_all1',
#            bbox_inches='tight', pad_inches=0.2)


# In[ ]:


############ plot torus bump ##############

import numpy as np
from utils import *
from scipy.stats import binned_statistic_2d
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection

f = np.load('Data/quentin_mod1_box_newdecoding.npz')
coords = f['coordsnew']
f.close()
spk = load_spikes('quentin', 'mod1', 'box', bSmooth = False, bBox = False)
times = np.where(np.sum(spk>0,1)>0)[0] 
numangsint = 51
m, x_edge, y_edge, circ = binned_statistic_2d(coords[:,0],coords[:,1], 
        spk[times,8], statistic='mean', bins=np.linspace(0, 2*np.pi,numangsint-1), 
                                              range=None, expand_binnumbers=True)
m = smooth_tuning_map(m, numangsint-1, 2.75, bClose = True)
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_edge, y_edge)
X +=np.pi/5
X = X%(2*np.pi)
r1 = 1.5
r2 = 1
x = (r1 + r2*np.cos(X))*np.cos(Y) 
y = (r1 + r2*np.cos(X))*np.sin(Y)  
z =  r2*np.sin(X)

ax.plot_surface(x,y,z, facecolors=cm.viridis(np.power(m/np.max(m),1)),alpha = 1,
                       linewidth=0.1, antialiased=True,
                      rstride = 1, cstride =1, shade = False, vmin = 0,zorder = -2)
ax.set_zlim(-2,2)
ax.view_init(-125,135)
#ax.view_init(45,135)

plt.axis('off')



r_box = transforms.Affine2D().skew_deg(15,15)
fig = plt.figure()
ax = fig.add_subplot(111) 
ax.imshow(np.rot90(np.roll(np.roll(m/np.max(m),40,1),0,0),1), origin = 'lower', extent = [0,2*np.pi,0, 2*np.pi], vmin = 0, vmax = 1)
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


# In[ ]:


############ num pcs versus persistence lifetime ##############
sspikes = load_spikes('roger', 'mod1', 'box', bSmooth = True, bSpeed = True) 

dgms = {}

dim = 6
ph_classes = [0,1] 
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 1
num_neurons = len(sspikes[0,:])
active_times = 15000
num_times = 5
n_points = 1200
k = 1000
metric = 'cosine'
nbs = 800
coeff = 47
times_cube = np.arange(0,len(sspikes[:,0]),num_times)
movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube,:],1))[-active_times:])
movetimes = times_cube[movetimes]

ssspikes = preprocessing.scale(sspikes[movetimes,:],axis = 0)


for i in np.arange(3,len(ssspikes[0,:])+1):
    dim_red_spikes_move_scaled,__,__  = pca(ssspikes, dim = i)
    indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                        n_points, 1, metric)

    X = squareform(pdist(dim_red_spikes_move_scaled[indstemp,:], metric))
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
                       do_cocycles=False, distance_matrix = True)
    dgms[i] = rips_real["dgms"]
    print(i)
    
np.savez('pca_analysis_R1', dgms_all = dgms)


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

#ax.set_yticklabels(['','','','', ''])
fig.savefig('Figs_review/varPCA_analysis_mod1', bbox_inches='tight', pad_inches=0.02)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dims[:13], np.divide(lives_top3_mod1[:13,2],lives_top3_mod1[:13,0]), lw = 1)
ax.plot(dims[:13], np.divide(lives_top3_mod1[:13,1],lives_top3_mod1[:13,0]), lw = 1)
#ax.plot(dims[:13], np.divide(lives_top3_mod1[:13,0],lives_top3_mod1[:13,0]), lw = 1)
ax.scatter(dims[:13], np.divide(lives_top3_mod1[:13,2],lives_top3_mod1[:13,0]), s = 20, lw = 1)
ax.scatter(dims[:13], np.divide(lives_top3_mod1[:13,1],lives_top3_mod1[:13,0]), s = 20, lw = 1)
#ax.scatter(dims[:13], np.divide(lives_top3_mod1[:13,0],lives_top3_mod1[:13,0]), s = 20, lw = 1)
ax.plot([6,6], [ax.get_ylim()[0],ax.get_ylim()[1]], ls = '--',c = 'k', alpha = 0.6, lw = 2)
#ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]], [1,1],  ls = '--',c = 'k', alpha = 0.6, lw = 1)
ax.set_ylim([0.8, 4.2])
ax.set_xlim([2, 15.3])
ax.set_aspect(1.0/ax.get_data_ratio())
ax.set_xticks([3, 5,10,15])
#ax.set_xticks([3, 6,11,16])
ax.set_xticklabels(['','','',''])
ax.set_yticks([1, 2,3, 4])
ax.set_yticklabels(['','','',''])
fig.savefig('Figs_review/varPCA_analysis_inset_mod1', bbox_inches='tight', pad_inches=0.03)




# In[ ]:


############ plot colorbar ##############
import matplotlib.pyplot as plt
import matplotlib as mpl

fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
plt.axis('off')
col_map = plt.get_cmap('afmhot')

cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal',
                               cmap=col_map)

plt.savefig('afmhot', bbox_inches='tight')
plt.savefig('afmhot.pdf', bbox_inches='tight')
import matplotlib.pyplot as plt
import matplotlib as mpl

fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
plt.axis('off')
col_map = plt.get_cmap('viridis')

cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal',
                               cmap=col_map)

plt.savefig('viridis', bbox_inches='tight')
plt.savefig('viridis.pdf', bbox_inches='tight')


# In[ ]:


############# GLM PCA #############
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
        fg = fg==1
        X_space = preprocess_dataX2(xxss[fg,:], num_bins)
#        print(ys[fg], X_space, LAM, periodicprior, GoGaussian)
        P[:, i] = fitmodel(ys[fg], X_space, LAM, periodicprior, GoGaussian)

        xt = xxss#[~fg,:]
        X_test = X_space#preprocess_dataX2(xt, num_bins)

        if(GoGaussian):
            yt[fg] = np.dot(P[:, i], X_test)        
            Lvals[fg] = -np.sum( (ys-yt)**2 )             
        else:
            H = np.dot(P[:, i], X_test)
            expH = np.exp(H)
            yt[fg] = expH
            finthechat = (np.ravel(np.log(factorial(ys[fg]))))
            Lvals[fg] = (np.ravel(ys[fg]*H - expH)) - finthechat

    leastsq = np.sum( (ys-yt)**2 )
    #print('LEAST SQ', leastsq)
    ym = np.mean(ys)
    #return (np.sum((yt-ym)**2) / np.sum((ys-ym)**2))
    return yt, (1. - leastsq/np.sum((ys-ym)**2)), P, Lvals

P_tor_all_mod1 = np.zeros((num_neurons, num_bins**2, nF2) )
P_space_all_mod1 = np.zeros((num_neurons, num_bins**2, nF2) )
LOOtorscores_mod1 = np.zeros((num_neurons))
spacescores_mod1 = np.zeros((num_neurons))

ypt_all_mod1 = []#np.zeros_like(pca_mod1)
Lvals_tor_mod1 = []#np.zeros_like(pca_mod1)
yps_all_mod1 = []#np.zeros_like(pca_mod1)
Lvals_space_mod1 = []#np.zeros_like(pca_mod1)

for n in np.arange(0, num_neurons, 1): 
    ypt_all_mod1.append([])
    Lvals_tor_mod1.append([])
    yps_all_mod1.append([])
    Lvals_space_mod1.append([])

    ypt_all_mod1[n], LOOtorscores_mod1[n], P_tor_all_mod1[n,:, :], Lvals_tor_mod1[n] = dirtyglm(coords2[ttimes], 
                                                                                                pca_mod1[tttimes,n], 
                                                                                                num_bins, True, LAM, 
                                                                                                GoGaussian, nF2)
    yps_all_mod1[n], spacescores_mod1[n], P_space_all_mod1[n,:, :], Lvals_space_mod1[n] = dirtyglm(xxyy[times[ttimes],:],
                                                                                                   pca_mod1[tttimes,n],
                                                                                                   num_bins, False, LAM,
                                                                                                   GoGaussian, nF2)


# In[ ]:



########### Plot pca space ###################

sig = 2.75
for name in ('roger_mod2_box','shane_mod1','roger_mod1_box',  'roger_mod3_box', 'quentin_mod1', 'quentin_mod2'):
    f = np.load('C:/Users/erihe/Downloads/PCA_' + name + '.npz')
    pca_mod1 = f['pcs'] 
    xx = f['xx']
    yy = f['yy']
    mtot_all = f['pcs_binned']
    f.close()
    for i in range(8):
        mtottemp = mtot_all[i,:,:].copy()
        #m2[np.isnan(m2)] = np.mean(m2[~np.isnan(m2)])
        mtottemp = smooth_tuning_map(np.rot90(mtottemp), len(mtottemp)+1, sig, bClose = True)   
        plt.figure()
        plt.imshow(mtottemp, origin = 'lower', vmin = -2.5, vmax = 2.5)
        plt.axis('off')
        plt.savefig(name + str(i), bbox_inches='tight', pad_inches=0.02)
        plt.savefig(name + str(i) + '.pdf', bbox_inches='tight', pad_inches=0.02)
        plt.show()


# In[ ]:


################ Run persistence H2 orig ####################

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

def nearest_neighbors(
    X, n_neighbors, metric, metric_kwds, angular, verbose=False
):

    if metric == "precomputed":
        # Note that this does not support sparse distance matrices yet ...
        # Compute indices of n nearest neighbors
        knn_indices = np.argsort(X)[:, :n_neighbors]
        # Compute the nearest neighbor distances
        #   (equivalent to np.sort(X)[:,:n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

        rp_forest = []
    else:
        if callable(metric):
            distance_func = metric
        elif metric in dist.named_distances:
            distance_func = dist.named_distances[metric]
        else:
            raise ValueError("Metric is neither callable, " + "nor a recognised string")

        if metric in ("cosine", "correlation", "dice", "jaccard"):
            angular = True

        rng_state = np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        if scipy.sparse.isspmatrix_csr(X):
            if metric in sparse.sparse_named_distances:
                distance_func = sparse.sparse_named_distances[metric]
                if metric in sparse.sparse_need_n_features:
                    metric_kwds["n_features"] = X.shape[1]
            else:
                raise ValueError(
                    "Metric {} not supported for sparse " + "data".format(metric)
                )
            metric_nn_descent = sparse.make_sparse_nn_descent(
                distance_func, tuple(metric_kwds.values())
            )
            # TODO: Hacked values for now
            n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))
            rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
            leaf_array = rptree_leaf_array(rp_forest)
            knn_indices, knn_dists = metric_nn_descent(
                X.indices,
                X.indptr,
                X.data,
                X.shape[0],
                n_neighbors,
                rng_state,
                max_candidates=60,
                rp_tree_init=True,
                leaf_array=leaf_array,
                n_iters=n_iters,
                verbose=verbose,
            )
        else:
            metric_nn_descent = make_nn_descent(
                distance_func, tuple(metric_kwds.values())
            )
            # TODO: Hacked values for now
            n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))

            rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
            leaf_array = rptree_leaf_array(rp_forest)
            knn_indices, knn_dists = metric_nn_descent(
                X,
                n_neighbors,
                rng_state,
                max_candidates=60,
                rp_tree_init=True,
                leaf_array=leaf_array,
                n_iters=n_iters,
                verbose=verbose,
            )

        if np.any(knn_indices < 0):
            warn(
                "Failed to correctly find n_neighbors for some samples."
                "Results may be less than ideal. Try re-running with"
                "different parameters."
            )
    return knn_indices, knn_dists, rp_forest

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
    return components


def sample_denoising(data,  k = 10, num_sample = 500, omega = 0.2, metric = 'euclidean'):    
    n = data.shape[0]
    leftinds = np.arange(n)
    F_D = np.zeros(n)
    if metric in ("cosine", "correlation", "dice", "jaccard"):
        angular = True
    else:
        angular = False
        
    X = squareform(pdist(data, metric))
    knn_indices, knn_dists, _ = nearest_neighbors( X
        , n_neighbors = k, metric = 'precomputed', 
        metric_kwds={}, angular=angular)

    sigmas, rhos = smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    F_D = 1/n*np.array(np.sum(result,0))[0,:]
    i = np.argmax(F_D)
    inds_all = np.arange(n)
    inds_left = inds_all>-1
    inds_left[i] = False
    inds = np.zeros(num_sample, dtype = int)
    inds[0] = i
    F_S = np.zeros(n)
    for j in np.arange(1,num_sample):
        F_S[inds_left] += result[i, inds_left].toarray()[0,:]
        F = F_D[inds_left] - omega/j*F_S[inds_left]
        i = inds_all[inds_left][np.argmax(F)]
        inds_left[i] = False   
        inds[j] = i
    return inds

def sample_quantization(data, numbins = 4, thresh = 2):    
	m,n = data.shape
	i = 0
	inds = np.zeros((m,n), dtype = int)
	centers = np.zeros((n,numbins))
	#bins = np.linspace(data[:,:].min()-1e-10, data[:,:].max()+1e-10, numbins+1)
	for i in np.arange(n):
	    bins = np.linspace(data[:,i].min()-1e-10, data[:,i].max()+1e-10, numbins+1)
	    centers[i,:] = bins[:-1] + (bins[1:]-bins[:-1])/2
	    inds[:,i] = np.digitize(data[:,i], bins)-1
	del data
	    sample = {}
	count = 0
	i = 0
	while i < m-1:
	    sample[count] = i 
	    j = i+1
	    while  j < m and (np.sum(np.abs(inds[j,:] - inds[i,:]))<=thresh):# or np.max(inds[j,:])<2):#np.sum(inds[j,:])<5):
	        j+=1
	    i = j
	    count += 1
	times_cube = np.zeros(len(sample),dtype = int)
	for i in range(len(sample)):
	    times_cube[i] = sample[i]
	return times_cube


def predict_color(circ_coord_sampled, data, sampled_data, dist_measure='euclidean', num_batch =20000, k = 10):
    num_tot = len(data)
    circ_coord_tot = np.zeros(num_tot)
    circ_coord_dist = np.zeros(num_tot)
    circ_coord_tmp = circ_coord_sampled*2*np.pi
    j = -1
    for j in range(int(num_tot/num_batch)):
        dist_landmarks = cdist(data[j*num_batch:(j+1)*num_batch, :], sampled_data, metric = dist_measure)
        closest_landmark = np.argsort(dist_landmarks, 1)[:,:k]
        weights = np.array([1-dist_landmarks[i,closest_landmark[i,:]]/dist_landmarks[i,closest_landmark[i,-1]] for i in range(num_batch)])
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1
        weights /= np.sum(weights, 1)[:,np.newaxis] 
        
        sincirc = [np.dot(np.sin(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(num_batch)]
        coscirc = [np.dot(np.cos(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(num_batch)]
        circ_coord_tot[j*num_batch:(j+1)*num_batch] = np.arctan2(sincirc, coscirc)
    
    dist_landmarks = cdist(data[(j+1)*num_batch:, :], sampled_data, metric = dist_measure)
    closest_landmark = np.argsort(dist_landmarks, 1)[:,:k]
    lenrest = len(closest_landmark[:,0])
    weights = np.array([1-dist_landmarks[i,closest_landmark[i,:]]/dist_landmarks[i,closest_landmark[i,k-1:k]] for i in range(lenrest)])
    if np.shape(weights)[0] == 0:
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1 
        weights /= np.sum(weights)
    else:
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1
        weights /= np.sum(weights, 1)[:,np.newaxis] 
    sincirc = [np.dot(np.sin(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(lenrest)]
    coscirc = [np.dot(np.cos(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(lenrest)]
    circ_coord_tot[(j+1)*num_batch:] = np.arctan2(sincirc, coscirc)%(2*np.pi)
    return circ_coord_tot

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
    f = lsmr(Aw, Bw)[0]
    return f

np.savez('tracking_' + file_name, azimuth = azimuth_, xx = xx_, yy = yy_, tt = tt_)
args = sys.argv
spikes_name = args[1]
f = np.load('Data/' + spikes_name + '.npz', allow_pickle = True)
sspikes = f['sspikes'][:,:]
f.close()

################### Hyperparameters ####################
sleeptime = np.random.rand()
time.sleep(sleeptime)


filename = spikes_name + '_orig_' + datetime.now().strftime('%Y%m%d%H%M%S')


################### Hyperparameters ####################
dim = 6
ph_classes = [0,1] # Decode the ith most persistent cohomology class
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 1
coeff = 47
n_points = 1500
n_neighbors = int(n_points/2)
omega = 0.1
k = 100
active_times = 15000
sleeptime = np.random.rand()*5
time.sleep(sleeptime)
t_start = time.time()
_LOG_2PI = np.log(2 * np.pi)
num_neurons = len(sspikes[0,:])
numbins = 4
cubethresh = 2
n_points = 1500#int(0.1*len(summed_sample_num))
n_neighbors = 1000#int(n_points/2)

############  ################
times_cube = sample_quantization(sspikes, numbins = numbins, thresh = cubethresh):    
movetimes =times_cube[np.argsort(np.sum(spikes2_1[:,:],1)[times_cube])[active_times:]]
dim_red_spikes_move_scaled = pca(preprocessing.scale(sspikes[time_sample,:]), dim = dim)
indstemp = sample_denoising(data = dim_red_spikes_move_scaled,  k = k, num_sample = n_points, 
							omega = omega, metric = metric)
dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
X = squareform(pdist(dim_red_spikes_move_scaled[:,:], metric))
knn_indices, knn_dists, _ = nearest_neighbors(X, n_neighbors = n_neighbors, metric = 'precomputed', 
                                              angular=True, metric_kwds = {})
sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors, local_connectivity=0)
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


### Compute persistent homology
rips_real = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=False, distance_matrix = True)
np.savez_compressed('RipsH2_' + filename, diagrams = rips_real["dgms"])


# In[ ]:


############### Plot barcodes ################
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
import matplotlib.gridspec as grd

args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name = args[3].strip()

#if (rat_name == 'roger') & ((sess_name[:3]=='rem') | (sess_name[:3] == 'sws')):
#    sess_name += '_rec2'

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
        #cs = [[1,0.55,0.1],[1,0.55,0.2],[1,0.55,0.2]]
        cs = np.repeat([[1,0.55,0.1]],3).reshape(3,3).T
#        labels_roll = [ "$H_n$ < 0.001"]
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
        print((np.sum(lives1_all>dl1)+1)/(num_rolls+1), (np.sum(lives1_all>dl2)+1)/(num_rolls+1))   
            
        axes.fill_betweenx(0.5+np.arange(len(dinds)), x1, ytemp, color = cs[(dim)], zorder = -2, alpha = 0.3)

#        axes.set_ylabel('between y1 and 0')

#        axes.plot(ytemp, 0.5+np.arange(len(dinds)), 
#                c = cs[(dim)], lw = 2,  linestyle = ':', label=labels_roll[0])

    axes.plot([0,0], [0, indsall], c = 'k', linestyle = '-', lw = 1)
    axes.plot([0,indsall],[0,0], c = 'k', linestyle = '-', lw = 1)
    axes.set_xlim([0, infinity])
#    axes.axis([0, infinity, 0, indsall])
#    axes.text(axis_start, indsall/2, labels[dim], fontsize=20)
#diagrams_all[0][infs] = np.inf
#    axes.set_anchor('W')
#    print(axes.images[0].get_extent())
#axes.text(0, -5, str(0), fontsize=12)
#axes.text(infinity-0.5, -5, str(int(round(infinity))), fontsize=12)
#axes.text(infinity/2-1, -5, 'Radius', fontsize=12)
fig.tight_layout(pad=3.5, w_pad=0.1, h_pad=0.25)
fig.savefig('' + file_name + 'persistence_barcode_inf' + str(int(round(infinity))), bbox_inches='tight')#, pad_inches=0.0)
fig.savefig('' + file_name + 'persistence_barcode_inf' + str(int(round(infinity))) + '.pdf', bbox_inches='tight')#, pad_inches=0.0)


# In[ ]:


############### PLot stripes oriented ################

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
from scipy.ndimage import rotate
from utils import *

args = sys.argv

rat_names = args[1].strip().split(',')
mod_names = args[2].strip().split(',')
sess_names = args[3].strip().split(',')

colors = {}
colors['box'] = [0.1,0.1,0.6,0.8]
colors['maz'] = [0.4,0.4,0.9,0.8]
colors['sws'] = [0.4,0.9,0.4,0.8]
colors['rem'] = [0.1,0.6,0.1,0.8]

fig1 = plt.figure()
ax = fig1.add_subplot(111)
plt.axis('off')

for rat_name in rat_names:
    for mod_name in mod_names:
        for sess_name in [sess_names[0]]:
            if (sess_name in ('sws')) & (rat_name == 'roger') & (mod_name == 'mod1'):
                sess_name += '_c0'    
            if (sess_name[:3] in ('rem', 'sws')) & (rat_name == 'roger'):    
                sess_name += '_rec2'
            plot_neuron(ax, rat_name,  mod_name, sess_name)
file_names = ''
for rat_name in rat_names:
    file_names += rat_name + '_'
    for mod_name in mod_names:
        file_names += mod_name + '_'
        for sess_name in sess_names:
            if (sess_name in ('sws')) & (rat_name == 'roger') & (mod_name == 'mod1'):
                sess_name += '_c0'    
            if (sess_name[:3] in ('rem', 'sws')) & (rat_name == 'roger'):    
                sess_name += '_rec2'
            file_names += '_' + sess_name
            file_name = rat_name + '_' + mod_name + '_' + sess_name
            f = np.load('Results/Orig/' + file_name + '_para.npz', allow_pickle = True)
            p1b_1 = f['p1b_1']
            p2b_1 = f['p2b_1']
            xedge = f['xedge']
            yedge = f['yedge']
            m1b_1 = f['m1b_1']
            m2b_1 = f['m2b_1']
            fun = f['fun']
            f.close()
            print(file_name)
            
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
            if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
                p1 = -1

            nnans = ~np.isnan(m2b_1)
            mtot = m2b_1[nnans]%(2*np.pi)
            p = p2b_1.copy()
            x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
            pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
            pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
            p2 = 1
            if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
                p2 = -1




            print(p1b_1,p2b_1, p1, p2)
            x,y = rot_para(p1b_1,p2b_1, p1, p2)
            print(x,y)
            xmin = xedge.min()
            xedge -= xmin
            xmax = xedge.max()
            ymin = yedge.min()
            yedge -= ymin
            ymax = yedge.max()
            print(((y[0]-x[0])/(2*np.pi)*360)%360)
            print(1/x[2]*xmax)
            print(1/y[2]*ymax)
            print(fun)
            print('')
            plot_para(ax,xedge,yedge, x, y, colors[sess_name[:3]])
#            plot_stripes(xedge,yedge, p1b_1, m1b_1, file_name + '_1')
#            plot_stripes(xedge,yedge, p2b_1, m2b_1, file_name + '_2')
#            print(p1b_1,p2b_1)

ax.set_aspect('equal', 'box')
fig1.savefig('Figs/' + file_names + '_parallelogram.png', bbox_inches='tight', pad_inches=0.01)
fig1.savefig('Figs/' + file_names + '_parallelogram.pdf', bbox_inches='tight', pad_inches=0.01)
#fig1.savefig('Figs/para/' + file_names + '_parallelogram.png', bbox_inches='tight', pad_inches=0.01)
#fig1.savefig('Figs/para/' + file_names + '_parallelogram.pdf', bbox_inches='tight', pad_inches=0.01)


# In[ ]:


############### PLot stripes oriented ################

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

matplotlib.use('Agg')

args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name_0 = args[3].strip()
sess_name_1 = args[4].strip()
_2_PI = 2*np.pi
num_shuffle = 100
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)
bins_torus = np.linspace(0,_2_PI, numangsint)
file_name = rat_name + '_' + mod_name  + '_' + sess_name_0 + '_' + sess_name_1
sig1 = 15
############################## Load 1 ############################
if (rat_name == 'roger') & (sess_name_0[:3] in ('rem', 'sws')):
    sess_name_0 += '_rec2'
if sess_name_0 == 'sws_c0_rec2':    
    spikes_1 = load_spikes(rat_name, mod_name, 'sws_rec2')

else:
    spikes_1 = load_spikes(rat_name, mod_name, sess_name_0)

times_1 = np.where(np.sum(spikes_1>0, 1)>=1)[0]
spikes_1 = spikes_1[times_1,:]

file_name_1 = rat_name + '_' + mod_name + '_' +  sess_name_0
    
f = np.load('Data/' + file_name_1 + '_newdecoding.npz',  allow_pickle = True)
call = f['coordsnew']
if sess_name_0[:3] == 'box':
    callbox_1 = call.copy()
else:    
    callbox_1 = f['coordsbox']
c11all_1 = call[:,0]
c12all_1 = call[:,1]
f.close()    
############################## Load 2 ############################
if (rat_name == 'roger') & (sess_name_1[:3] in ('rem', 'sws')):
    sess_name_1 += '_rec2'
if sess_name_1 == 'sws_c0_rec2':    
    spikes_2 = load_spikes(rat_name, mod_name, 'sws_rec2')

else:
    spikes_2 = load_spikes(rat_name, mod_name, sess_name_1)
    

file_name_2 = rat_name + '_' +  mod_name + '_' + sess_name_1
xx,yy, speed = load_pos(rat_name, sess_name_1, bSpeed = True)
xx = xx[speed>2.5]
yy = yy[speed>2.5]
#spikes_2 = spikes_2[speed>2.5,:]

times_2 = np.where(np.sum(spikes_2>0, 1)>=1)[0]
spikes_2 = spikes_2[times_2,:]
xx = xx[times_2]
yy = yy[times_2]

f = np.load('Data/' + file_name_2 + '_newdecoding.npz',  allow_pickle = True)
call = f['coordsnew']
if sess_name_1[:3] == 'box':
    callbox_2 = call.copy()
else:    
    callbox_2 = f['coordsbox']
c11all_2 = call[:,0]
c12all_2 = call[:,1]
f.close()    
num_neurons = len(spikes_1[0,:])

############################## compare ############################
cells_all = range(num_neurons)

def rot_coord(params1,params2, c1, c2, p):    
    rot_mat = np.zeros((2,2))
    if np.abs(np.cos(params1[0])) < np.abs(np.cos(params2[0])):        
        print('nonrot')
        cc1 = c2.copy()
        cc2 = c1.copy()
        y = params1.copy()
        x = params2.copy()
        p = np.flip(p)
    else:   
        print('rot')
        cc1 = c1.copy()
        cc2 = c2.copy()
        x = params1.copy()
        y = params2.copy()  
    print(p, x[1], y[1])
    if p[1] ==-1:
        cc2 = (2*np.pi-cc2)
    if p[0] ==-1:
        cc1 = (2*np.pi-cc1)
    alpha = (y[0]-x[0])
    if (alpha < 0) & (np.abs(alpha) > np.pi/2):
        print('1')
        cctmp = cc2.copy()
        cc2 = cc1.copy()
        cc1 = cctmp
    if (alpha < 0) & (np.abs(alpha) < np.pi/2):
        cc1 = (2*np.pi-cc1 +  np.pi/3*cc2)
        print('2')
    elif np.abs(alpha) > np.pi/2:
        cc2 = (cc2 + np.pi/3*cc1)
        print('3')

    return np.concatenate((cc1[:,np.newaxis], cc2[:,np.newaxis]),1)%_2_PI


f = np.load('Results/Orig/' + file_name_1 + '_para.npz', allow_pickle = True)
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
    pm00 = (2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi)
    p1 = -1

nnans = ~np.isnan(m2b_1)
mtot = m2b_1[nnans]%(2*np.pi)
p = p2b_1.copy()
x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
p2 = 1
pm01 = (p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi)
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    pm01 = (2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi)
    p2 = -1


if sess_name_0[:3] not in ('box'):
    cbox1 = rot_coord(p1b_1,p2b_1, callbox_1[:,0], callbox_1[:,1], (p1,p2))
else:
    cbox1 = rot_coord(p1b_1,p2b_1, c11all_1, c12all_1, (p1,p2))

f = np.load('Results/Orig/' + file_name_2 + '_para.npz', allow_pickle = True)
p1b_2 = f['p1b_1']
p2b_2 = f['p2b_1']
m1b_2 = f['m1b_1']
m2b_2 = f['m2b_1']
f.close()

nnans = ~np.isnan(m1b_2)
mtot = m1b_2[nnans]%(2*np.pi)
p = p1b_2.copy()
x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
p1 = 1
pm11 = (p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi)
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    pm11 = (2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi)
    p1 = -1

nnans = ~np.isnan(m2b_2)
mtot = m2b_2[nnans]%(2*np.pi)
p = p2b_2.copy()
x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
p2 = 1
pm12 = (p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi)
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    pm12 = (2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi)
    p2 = -1

if sess_name_0[:3] not in ('box'):
    cbox2 = rot_coord(p1b_2,p2b_2, callbox_2[:,0], callbox_2[:,1], (p1,p2))
else:
    cbox2 = rot_coord(p1b_2,p2b_2, c11all_2, c12all_2, (p1,p2))


pshift = np.arctan2(np.mean(np.sin(cbox1 - cbox2),0), np.mean(np.cos(cbox1 - cbox2),0))%(2*np.pi)
print(pshift)
cbox1 = (cbox1 - pshift)%(2*np.pi)

m1b_1, m2b_1, xedge,yedge = get_ang_hist(cbox1[:,0], 
    cbox1[:,1], xx,yy)

fig, ax = plt.subplots(1,1)
ax.imshow(np.cos(m1b_1).T, origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
ax.set_aspect('equal', 'box')
ax.set_xticks([], [])
ax.set_yticks([], [])
fig.savefig('Figs/stripes_ori/' + file_name_1 + '_1.png', bbox_inches='tight', pad_inches=0.02)
fig.savefig('Figs/stripes_ori/' + file_name_1 + '_1.pdf', bbox_inches='tight', pad_inches=0.02)
fig, ax = plt.subplots(1,1)
ax.imshow(np.cos(m2b_1).T, origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
ax.set_aspect('equal', 'box')
ax.set_xticks([], [])
ax.set_yticks([], [])
fig.savefig('Figs/stripes_ori/' + file_name_1 + '_2.png', bbox_inches='tight', pad_inches=0.02)
fig.savefig('Figs/stripes_ori/' + file_name_1 + '_2.pdf', bbox_inches='tight', pad_inches=0.02)


m1b_1, m2b_1, xedge,yedge = get_ang_hist(cbox2[:,0], 
    cbox2[:,1], xx,yy)

fig, ax = plt.subplots(1,1)
ax.imshow(np.cos(m1b_1).T, origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
ax.set_aspect('equal', 'box')
ax.set_xticks([], [])
ax.set_yticks([], [])
fig.savefig('Figs/stripes_ori/' + file_name_2 + '_1.png', bbox_inches='tight', pad_inches=0.02)
fig.savefig('Figs/stripes_ori/' + file_name_2 + '_1.pdf', bbox_inches='tight', pad_inches=0.02)
fig, ax = plt.subplots(1,1)
ax.imshow(np.cos(m2b_1).T, origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
ax.set_aspect('equal', 'box')
ax.set_xticks([], [])
ax.set_yticks([], [])
fig.savefig('Figs/stripes_ori/' + file_name_2 + '_2.png', bbox_inches='tight', pad_inches=0.02)
fig.savefig('Figs/stripes_ori/' + file_name_2 + '_2.pdf', bbox_inches='tight', pad_inches=0.02)





fig, ax = plt.subplots(1,1)
ax.imshow(np.cos(pm00), origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
ax.set_aspect('equal', 'box')
ax.set_xticks([], [])
ax.set_yticks([], [])
fig.savefig('Figs/stripes_ori/' + file_name_1 + '_1_cos.png', bbox_inches='tight', pad_inches=0.02)
fig.savefig('Figs/stripes_ori/' + file_name_1 + '_1_cos.pdf', bbox_inches='tight', pad_inches=0.02)
fig, ax = plt.subplots(1,1)
ax.imshow(np.cos(pm01), origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
ax.set_aspect('equal', 'box')
ax.set_xticks([], [])
ax.set_yticks([], [])
fig.savefig('Figs/stripes_ori/' + file_name_1 + '_2_cos.png', bbox_inches='tight', pad_inches=0.02)
fig.savefig('Figs/stripes_ori/' + file_name_1 + '_2_cos.pdf', bbox_inches='tight', pad_inches=0.02)


fig, ax = plt.subplots(1,1)
ax.imshow(np.cos(pm11), origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
ax.set_aspect('equal', 'box')
ax.set_xticks([], [])
ax.set_yticks([], [])
fig.savefig('Figs/stripes_ori/' + file_name_2 + '_1_cos.png', bbox_inches='tight', pad_inches=0.02)
fig.savefig('Figs/stripes_ori/' + file_name_2 + '_1_cos.pdf', bbox_inches='tight', pad_inches=0.02)
fig, ax = plt.subplots(1,1)
ax.imshow(np.cos(pm12), origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
ax.set_aspect('equal', 'box')
ax.set_xticks([], [])
ax.set_yticks([], [])
fig.savefig('Figs/stripes_ori/' + file_name_2 + '_2_cos.png', bbox_inches='tight', pad_inches=0.02)
fig.savefig('Figs/stripes_ori/' + file_name_2 + '_2_cos.pdf', bbox_inches='tight', pad_inches=0.02)


# In[ ]:




######### Run ps ###########

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy import signal
import matplotlib
matplotlib.use('Agg')
from scipy import stats
import sys
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
from utils import *
from scipy.stats import wilcoxon


r_box = transforms.Affine2D().skew_deg(15,15)
args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_names = args[3].strip().split(',')
stat = args[4].strip()

numangsint = 51
sig = 2.75
xs = {}
ys = {}
spikes = {}
mtot = {}

Lams = {}
Lams['roger_mod1_box'] = np.sqrt(10)
Lams['roger_mod3_box'] = np.sqrt(10)
Lams['roger_mod4_box'] = np.sqrt(10)
Lams['roger_mod1_box_rec2'] = 1
Lams['roger_mod2_box_rec2'] = 1
Lams['roger_mod3_box_rec2'] = 1

Lams['quentin_mod1_box'] = 1
Lams['quentin_mod2_box'] = 1
Lams['shane_mod1_box'] = np.sqrt(10)

Lams['roger_mod1_maze'] = np.sqrt(10)
Lams['roger_mod3_maze'] = 1
Lams['roger_mod4_maze'] = 1
Lams['quentin_mod1_maze'] = 1
Lams['quentin_mod2_maze'] = 1
Lams['shane_mod1_maze'] = np.sqrt(10)


Lams['roger_mod1_box_space'] = np.sqrt(10)
Lams['roger_mod3_box_space'] = np.sqrt(10)
Lams['roger_mod4_box_space'] = np.sqrt(10)
Lams['roger_mod1_box_rec2_space'] = 1
Lams['roger_mod2_box_rec2_space'] = 1
Lams['roger_mod3_box_rec2_space'] = 1

Lams['quentin_mod1_box_space'] = 1
Lams['quentin_mod2_box_space'] = 1
Lams['shane_mod1_box_space'] = 1

Lams['roger_mod1_maze_space'] = 1
Lams['roger_mod3_maze_space'] = 1
Lams['roger_mod4_maze_space'] = 1
Lams['quentin_mod1_maze_space'] = 1
Lams['quentin_mod2_maze_space'] = 1
Lams['shane_mod1_maze_space'] = np.sqrt(10)



Lams['roger_mod1_sws_rec2'] = 1
Lams['roger_mod2_sws_rec2'] = np.sqrt(10)
Lams['roger_mod3_sws_rec2'] = np.sqrt(10)
Lams['quentin_mod1_sws'] = np.sqrt(10)
Lams['quentin_mod2_sws'] =  np.sqrt(10)
Lams['shane_mod1_sws'] = 1

Lams['roger_mod1_rem_rec2'] = np.sqrt(10)
Lams['roger_mod2_rem_rec2'] = 1
Lams['roger_mod3_rem_rec2'] = np.sqrt(10)
Lams['quentin_mod1_rem'] = np.sqrt(10)
Lams['quentin_mod2_rem'] = 1
Lams['shane_mod1_rem'] = 1


if (stat[:4] in ('corr')):
    sess_name_0 = sess_names[0]
    sess_name_1 = sess_names[1]
    file_name = rat_name + '_' + mod_name + '_' + sess_name_0 + '_' + sess_name_1

    sig = 2.75
    num_shuffle = 1000

    _2_PI = 2*np.pi
    numangsint = 51
    numangsint_1 = numangsint-1
    bins = np.linspace(0,_2_PI, numangsint)
    if stat[4:] == 'same':               
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

        num_neurons = np.shape(masscenters_11)[0]
        cells_all = np.arange(num_neurons)
        corr1 = np.zeros(num_neurons)
        corr2 = np.zeros(num_neurons)

        for n in cells_all:
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
            mtot_11[n,:,:]= m11
            mtot_22[n,:,:]= m22
            
            mtot_21[n,:,:]= m21
            mtot_12[n,:,:]= m12

        dist1 =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_11 - masscenters_21),
                                      np.cos(masscenters_11 - masscenters_21))),1))
        dist2 =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_12 - masscenters_22),
                                      np.cos(masscenters_12 - masscenters_22))),1))

        dist_shuffle = np.zeros((num_shuffle, num_neurons))
        corr_shuffle = np.zeros((num_shuffle, num_neurons))
        np.random.seed(47)
        for i in range(num_shuffle):
            inds = np.arange(num_neurons)
            np.random.shuffle(inds)
            for n in cells_all:
                corr_shuffle[i,n] = pearsonr(mtot_12[n,:,:].flatten(), mtot_22[inds[n],:,:].flatten())[0]
            dist_shuffle[i,:] =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_12 - masscenters_22[inds,:]),
                    np.cos(masscenters_12 - masscenters_22[inds,:]))),1))
        distshuffmeans1 = np.zeros(num_shuffle)
        corrshuffmeans1 = np.zeros(num_shuffle)
        for i in range(num_shuffle):
            distshuffmeans1[i] = np.mean(dist_shuffle[i])
            corrshuffmeans1[i] = np.mean(corr_shuffle[i])

        dist_shuffle = np.zeros((num_shuffle, num_neurons))
        corr_shuffle = np.zeros((num_shuffle, num_neurons))
        np.random.seed(47)
        for i in range(num_shuffle):
            inds = np.arange(num_neurons)
            np.random.shuffle(inds)
            for n in cells_all:
                corr_shuffle[i,n] = pearsonr(mtot_11[n,:,:].flatten(), mtot_21[inds[n],:,:].flatten())[0]
            dist_shuffle[i,:] =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_11 - masscenters_21[inds,:]),
                    np.cos(masscenters_11 - masscenters_21[inds,:]))),1))

        distshuffmeans2 = np.zeros(num_shuffle)
        corrshuffmeans2 = np.zeros(num_shuffle)
        for i in range(num_shuffle):
            distshuffmeans2[i] = np.mean(dist_shuffle[i])
            corrshuffmeans2[i] = np.mean(corr_shuffle[i])


#        print('dist1 ' + str( np.sum(distshuffmeans1<dist1.mean())))
#        print(dist1.mean(), dist1.std(), dist1.std()/np.sqrt(num_neurons))
#        print(distshuffmeans1.mean(), distshuffmeans1.std(), distshuffmeans1.std()/np.sqrt(num_neurons))
        print('corr1 ' + str( np.sum(corrshuffmeans1>corr1.mean())))
        print(corr1.mean(), corr1.std(), corr1.std()/np.sqrt(num_neurons))
        print(corrshuffmeans1.mean(), corrshuffmeans1.std(), corrshuffmeans1.std()/np.sqrt(num_neurons))
 
 #       print('dist2 ' + str( np.sum(distshuffmeans2<dist2.mean())))
 #       print(dist2.mean(), dist2.std(), dist2.std()/np.sqrt(num_neurons))
 #       print(distshuffmeans2.mean(), distshuffmeans2.std(), distshuffmeans2.std()/np.sqrt(num_neurons))
        print('corr2 ' + str( np.sum(corrshuffmeans2>corr1.mean())))
        print(corr2.mean(), corr2.std(), corr2.std()/np.sqrt(num_neurons))
        print(corrshuffmeans2.mean(), corrshuffmeans2.std(), corrshuffmeans2.std()/np.sqrt(num_neurons))
 
    else:

        f = np.load('Results/Orig/' + file_name + '_alignment2.npz', allow_pickle = True)
        mtot_1 = f['mtot_1']
        mtot_2 = f['mtot_2']
        f.close()
        f = np.load('Results/Orig/' + file_name + '_alignment3.npz', allow_pickle = True)
        masscenters_1 = f['masscenters_1']
        masscenters_2 = f['masscenters_2']
        dist = f['dist']
        corr = f['corr']
        f.close()
        num_neurons = np.shape(masscenters_1)[0]

        cells_all = np.arange(num_neurons)
        corr = np.zeros(num_neurons)


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
        num_neurons = len(dist)
        distshuffmeans = np.zeros(num_shuffle)
        corrshuffmeans = np.zeros(num_shuffle)
        for i in range(num_shuffle):
            distshuffmeans[i] = np.mean(dist_shuffle[i])
            corrshuffmeans[i] = np.mean(corr_shuffle[i])

#        print('dist' + str( np.sum(distshuffmeans<dist.mean())))
#        print(dist.mean(), dist.std(), dist.std()/np.sqrt(num_neurons))
#        print(distshuffmeans.mean(), distshuffmeans.std(), distshuffmeans.std()/np.sqrt(num_neurons))
        print('corr' + str( np.sum(corrshuffmeans>corr.mean())))
        print(corr.mean(), corr.std(), corr.std()/np.sqrt(num_neurons))
        print(corrshuffmeans.mean(), corrshuffmeans.std(), corrshuffmeans.std()/np.sqrt(num_neurons))
elif stat == 'info':
    sess_name = sess_names[0]
    if (rat_name == 'roger') & (sess_name[:3] in ('rem', 'sws')):
        sess_name += '_rec2'
    file_name = rat_name + '_' + mod_name + '_' + sess_name
    if sess_name[:3] in ('box', 'maz'):                
        f = np.load('GLM_info2/' + file_name + '_info.npz')
        I_torus = f['I_5_noise']
        I_space = f['I_1']
        f.close()
        f = np.load('GLM_info2/' + file_name + '_info_shuf.npz')
        I_shuf = np.mean(f['I_shuf'],1)
        f.close()

        num_neurons = len(I_torus)
        w, p = wilcoxon(I_torus-I_space, alternative='greater')
        print('info wilcoxon: ' + str(p))
        print('info: ' + str(np.sum(I_shuf>I_torus.mean())))
        print(I_torus.mean(), I_torus.std(), I_torus.std()/np.sqrt(num_neurons))
        print(I_shuf.mean(), I_shuf.std(), I_shuf.std()/np.sqrt(num_neurons))
        print(I_space.mean(), I_space.std(), I_space.std()/np.sqrt(num_neurons))

    else:
        f = np.load('GLM_info2/' + file_name + '_info.npz')
        I_torus = f['I_5']
        f.close()
        f = np.load('GLM_info2/' + file_name + '_info_shuf.npz')
        I_shuf = f['I_shuf']
        f.close()
        num_neurons = len(I_torus)
        print('info: ' + str(np.sum(np.sum(I_shuf-I_torus[:,np.newaxis]>0,1)>0)))
        print(I_torus.mean(), I_torus.std(), I_torus.std()/np.sqrt(num_neurons))
        I_shuf = np.mean(I_shuf,1)
        print(I_shuf.mean(), I_shuf.std(), I_shuf.std()/np.sqrt(num_neurons))
elif stat == 'EV':
    sess_name = sess_names[0]
    if (rat_name == 'roger') & (sess_name[:3] in ('rem', 'sws')):
        sess_name += '_rec2'
    file_name = rat_name + '_' + mod_name + '_' + sess_name

    LAM_tor = int(Lams[file_name])
    if sess_name[:3] in ('box', 'maz'):                
        LAM_space = int(Lams[file_name + '_space'])
        seeds = np.arange(10)
        torscores = []#np.zeros_like(spacescores)
        for i in seeds:
            f = np.load('GLM_info_res4/' + file_name + '_1_P_seed' + str(i) + 'LAM' + str(LAM_tor) + '_deviance.npz')
            torscores.append(f['LOOtorscores']/len(seeds))
            if i == 0:
                spacescores= f['spacescores']
            f.close()
        torscores = np.sum(np.array(torscores),0)
        num_neurons = len(torscores)
        w, p = wilcoxon(torscores-spacescores, alternative='greater')
        print('EV wilcoxon: ' + str(p))
        print(torscores.mean(), torscores.std(), torscores.std()/np.sqrt(num_neurons))
        print(spacescores.mean(), spacescores.std(), spacescores.std()/np.sqrt(num_neurons))

    else:
        f = np.load('GLM_info_res4/' + file_name + '_1_P_sd_LAM' + str(LAM_tor) + '_deviance.npz')
        torscores= f['LOOtorscores']
        f.close()
        num_neurons = len(torscores)
        print('EV')
        print(torscores.mean(), torscores.std(), torscores.std()/np.sqrt(num_neurons))


# In[ ]:


######### Run glm conj ###########

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
import glob

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
                LL[~fg] = -np.sum( (ys[~fg]-yt[~fg])**2 )     
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
    return 1 - (np.sum(LS) - np.sum(LL[~np.isinf(LL)]))/(np.sum(LS) - np.sum(LLnull[~np.isinf(LL)]))

Lams = np.array([1,10,100,1000])
Lams = np.sqrt(Lams).astype(int)

LOOtorscoresall = {}
GoGaussian = False
nF2 = 3
num_bins = 15

for rat_name, mod_name, sess_name in (
                                      
                                     ('quentin', 'mod1', 'box'),
                                     ('quentin', 'mod1', 'rem'),
                                     ('quentin', 'mod1', 'sws'),
                                     ('roger', 'mod1', 'sws'),                                     
                                     ('quentin', 'mod2', 'box'),
                                     ('quentin', 'mod2', 'rem'),
                                     ('quentin', 'mod2', 'sws'),

                                      ('shane', 'mod1', 'box'),
                                     ('roger', 'mod1', 'box_rec2'),
                                     ('roger', 'mod2', 'box_rec2'),
                                     ('roger', 'mod3', 'box_rec2'),
                                     
                                     ('roger', 'mod1', 'rem'),
                                     ('roger', 'mod2', 'rem'),
                                     ('roger', 'mod3', 'rem'),
                                     ('roger', 'mod2', 'sws'),
                                     ('roger', 'mod3', 'sws'),
                                     
                                     ):

    file_name = rat_name + '_' + mod_name + '_' + sess_name    

    if (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name == 'sws'):
        ############## class 1 #############
        f = np.load('Data/roger_mod1_sws_spikes_class1.npz', allow_pickle = True)
        sspikes = f['spikes']
        f.close()
        
        T, num_neurons = np.shape(sspikes)
        LOOtorscorestmp =  np.zeros((num_neurons, len(Lams)))
        
        for n in range(num_neurons):
            f = np.load('Data/roger_mod1_sws_class1_coords_N' + str(n) + '.npz', 
                        allow_pickle = True)
            coords2 = f['coords']
            times = f['times']
            f.close()
            for i, LAM in enumerate(Lams):    
                f = np.load('GLM_conj/' + file_name + '_class1_GLM' + str(LAM) + '.npz',allow_pickle = True)
                LL = f['Lt'][n]
                f.close()
                LAMcurr = float(LAM)
                LOOtorscorestmp[n,i] = compute_deviance(coords2, sspikes[times,n], GoGaussian, nF2, [], LL)            
        lamtor = np.argsort(np.sum(LOOtorscorestmp,0))[-1]
        LOOtorscores = LOOtorscorestmp[:,lamtor]
        LOOtorscoresall[file_name + 'class1'] = LOOtorscores
        
        
        
        f = np.load('Data/roger_mod1_sws_spikes_class2.npz', allow_pickle = True)
        sspikes = f['spikes']
        f.close()        
        T, num_neurons = np.shape(sspikes)
        LOOtorscorestmp =  np.zeros((num_neurons, len(Lams)))
        f = np.load('Data/roger_mod1_sws_class1_newdecoding.npz', 
                    allow_pickle = True)
        coords2 = f['coordsnew']
        times = f['times']
        f.close()
        for n in range(num_neurons):
            for i, LAM in enumerate(Lams):    
                f = np.load('GLM_conj/' + file_name + '_class2_GLM' + str(LAM) + '.npz',allow_pickle = True)
                LL = f['Lt'][n]
                f.close()
                LAMcurr = float(LAM)
                LOOtorscorestmp[n,i] = compute_deviance(coords2, sspikes[times,n], GoGaussian, nF2, [], LL)            
        lamtor = np.argsort(np.sum(LOOtorscorestmp,0))[-1]
        LOOtorscores = LOOtorscorestmp[:,lamtor]
        LOOtorscoresall[file_name + 'class1'] = LOOtorscores
        
        f = np.load('Data/roger_mod1_sws_spikes_class3.npz', allow_pickle = True)
        sspikes = f['spikes']
        f.close()        
        T, num_neurons = np.shape(sspikes)
        LOOtorscorestmp =  np.zeros((num_neurons, len(Lams)))        
        f = np.load('Data/roger_mod1_sws_class1_newdecoding.npz', 
                    allow_pickle = True)
        coords2 = f['coordsnew']
        times = f['times']
        f.close()
        for n in range(num_neurons):
            for i, LAM in enumerate(Lams):    
                f = np.load('GLM_conj/' + file_name + '_class3_GLM' + str(LAM) + '.npz')
                LL = f['Lt'][n]
                f.close()
                LAMcurr = float(LAM)
                LOOtorscorestmp[n,i] = compute_deviance(coords2, sspikes[times,n], GoGaussian, nF2, [], LL)            
        lamtor = np.argsort(np.sum(LOOtorscorestmp,0))[-1]
        LOOtorscores = LOOtorscorestmp[:,lamtor]
        LOOtorscoresall[file_name + 'class1'] = LOOtorscores
    else:
        if (rat_name == 'roger') & (sess_name in ('rem', 'sws')):
            file_name += '_rec2'
        if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
                sess_name += '_rec2'
        sspikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bConj = False)
            
        T, num_neurons = np.shape(sspikes)
        LOOtorscorestmp =  np.zeros((num_neurons, len(Lams)))        
        filenames = glob.glob('GLM_info_review/' + file_name + '_coords*')
        coordsall = {}
        if len(filenames) == 1:
            f = np.load(filenames[0], allow_pickle = True) 
            coordsall = f['coordsall'][()]
            f.close()
        else:
            for i in range(len(filenames)):
                f = np.load(filenames[i], allow_pickle = True)
                coordsall.update(f['coordstemp'][()])
                f.close()
                print(coordsall)
        for n in range(num_neurons):
            coords2 = coordsall[str(n)].copy()

            inds = np.ones(len(sspikes[0,:]))
            inds[n] = 0
            inds = np.where(inds)[0]
            times = np.where(np.sum(sspikes[:,inds]>0,1)>0)[0]

            for i, LAM in enumerate(Lams):    
                f = np.load('GLM_info_review/' + file_name + '_2_P_sd' + str(int(LAM))+ '.npz',allow_pickle = True) 
                LL = f['Lvals_tor'][n]
                f.close()
                LAMcurr = float(LAM)
                LOOtorscorestmp[n,i] = compute_deviance(coords2, sspikes[times,n], GoGaussian, nF2, [], LL)            
        lamtor = np.argsort(np.sum(LOOtorscorestmp,0))[-1]
        LOOtorscores = LOOtorscorestmp[:,lamtor]
        LOOtorscoresall[file_name + '_pure'] = LOOtorscores


        sspikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bConj = True)
        T, num_neurons = np.shape(sspikes)
        LOOtorscorestmp =  np.zeros((num_neurons, len(Lams)))        
        f = np.load('Data/' + file_name + '_newdecoding.npz', allow_pickle = True)
        coords2 = f['coordsnew']
        times = f['times']
        f.close()
        for n in range(num_neurons):
            for i, LAM in enumerate(Lams):    
                f = np.load('GLM_conj/' + file_name + '_GLM' + str(int(LAM)) + '.npz',allow_pickle = True)
                LL = f['Lt'][n]
                f.close()
                LAMcurr = float(LAM)
                LOOtorscorestmp[n,i] = compute_deviance(coords2, sspikes[times,n], GoGaussian, nF2, [], LL)            
        lamtor = np.argsort(np.sum(LOOtorscorestmp,0))[-1]
        LOOtorscores = LOOtorscorestmp[:,lamtor]
        LOOtorscoresall[file_name + '_conj'] = LOOtorscores

np.savez('deviance_all', LOOtorscoresall)


# In[ ]:



######### Run glm conj ###########

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
matplotlib.use('Agg')


args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name = args[3].strip()
GoGaussian = False
LAM = np.sqrt(float(args[4].strip()))

if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
sspikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bConj = True)
file_name = rat_name + '_' + mod_name + '_' + sess_name
xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
xx = xx[speed>2.5]
yy = yy[speed>2.5]
#sspikes = sspikes[speed>2.5,:]
T, num_neurons = np.shape(sspikes)
xxyy = np.zeros((len(xx),2))
xxyy[:,0] = xx+0
xxyy[:,1] = yy+0


nF2 = 3
ypt_all = []#np.zeros_like(sspikes)
Lvals_tor = []#np.zeros_like(sspikes)
yps_all = []#np.zeros_like(sspikes)
Lvals_space = []#np.zeros_like(sspikes)
num_bins = 15
P_tor_all = np.zeros((num_neurons, num_bins**2, nF2) )
P_space_all = np.zeros((num_neurons, num_bins**2, nF2) )
LOOtorscores = np.zeros((num_neurons))
spacescores = np.zeros((num_neurons))

   
#len_scale = (paras_all[file_name][1] + paras_all[file_name][2])/2*100
#noisestd = (1.5/len_scale)*2.*np.pi
f = np.load('Data/' + file_name + '_newdecoding.npz', allow_pickle = True)
coords2 = f['coordsnew']
times = f['times']
f.close()
#np.random.seed(47)
#coords2 = (coords+np.random.normal(0, noisestd, np.shape(coords)))%(2*np.pi)

for n in np.arange(0, num_neurons, 1): 
    ypt_all.append([])
    Lvals_tor.append([])
    yps_all.append([])
    Lvals_space.append([])	
    ypt_all[n], LOOtorscores[n], P_tor_all[n,:, :], Lvals_tor[n] = dirtyglm(coords2, sspikes[times,n], num_bins, True, LAM, GoGaussian, nF2)
    
np.savez_compressed('GLM_conj/' + file_name + '_GLM' + str(int(LAM)) , yt = ypt_all, Lt = Lvals_tor, P_tor_all = P_tor_all)
np.savez_compressed('GLM_conj/' + file_name + 'GLM' + str(int(LAM)) + '_stats', LOOtorscores= LOOtorscores)

######### Run glm noise ###########

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
matplotlib.use('Agg')


args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name = args[3].strip()
GoGaussian = (args[4].strip()=='gaussian')
LAM = np.sqrt(float(args[5].strip()))


if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
if GoGaussian:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = 15)
else:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = False)
file_name = rat_name + '_' + mod_name + '_' + sess_name
xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
xx = xx[speed>2.5]
yy = yy[speed>2.5]
sspikes = sspikes[speed>2.5,:]
T, num_neurons = np.shape(sspikes)
xxyy = np.zeros((len(xx),2))
xxyy[:,0] = xx+0
xxyy[:,1] = yy+0


nF2 = 3
ypt_all = []#np.zeros_like(sspikes)
Lvals_tor = []#np.zeros_like(sspikes)
yps_all = []#np.zeros_like(sspikes)
Lvals_space = []#np.zeros_like(sspikes)
num_bins = 15
P_tor_all = np.zeros((num_neurons, num_bins**2, nF2) )
P_space_all = np.zeros((num_neurons, num_bins**2, nF2) )
LOOtorscores = np.zeros((num_neurons))
spacescores = np.zeros((num_neurons))
import glob
filenames = glob.glob('GLM_info_res2/' + file_name + '_coords*')
coordsall = {}
if len(filenames) == 1:
	f = np.load(filenames[0], allow_pickle = True) 
	coordsall = f['coordsall'][()]
	f.close()
else:
	for i in range(len(filenames)):
		f = np.load(filenames[i], allow_pickle = True)
		coordsall.update(f['coordstemp'][()])
		f.close()
		print(coordsall)

for n in np.arange(0, num_neurons, 1): 
    coords2 = coordsall[str(n)].copy()
    ypt_all.append([])
    Lvals_tor.append([])
    yps_all.append([])
    Lvals_space.append([])

    inds = np.ones(len(sspikes[0,:]))
    inds[n] = 0
    inds = np.where(inds)[0]

    times = np.where(np.sum(sspikes[:,inds]>0,1)>0)[0]


    ypt_all[n], LOOtorscores[n], P_tor_all[n,:, :], Lvals_tor[n] = dirtyglm(coords2[times,:], sspikes[times,n], num_bins, True, LAM, GoGaussian, nF2)
    yps_all[n], spacescores[n], P_space_all[n,:, :], Lvals_space[n] = dirtyglm(xxyy[times,:], sspikes[times,n], num_bins, False, LAM, GoGaussian, nF2)
    
if GoGaussian:
    np.savez_compressed('GLM_info_res4/' + file_name + '_1_G_sd' + str(int(LAM)), P_tor_all = P_tor_all, LOOtorscores = LOOtorscores, spacescores = spacescores)
    np.savez_compressed('GLM_info_res4/' + file_name + '_2_G_sd' + str(int(LAM)), ypt_all = ypt_all, yps_all = yps_all, Lvals_space = Lvals_space, Lvals_tor = Lvals_tor)
else:
    np.savez_compressed('GLM_info_res4/' + file_name + '_1_P_sd' + str(int(LAM)), P_tor_all = P_tor_all, LOOtorscores = LOOtorscores, spacescores = spacescores)
    np.savez_compressed('GLM_info_res4/' + file_name + '_2_P_sd' + str(int(LAM)), ypt_all = ypt_all, yps_all = yps_all, Lvals_space = Lvals_space, Lvals_tor = Lvals_tor)




######### Run glm noise ###########

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
matplotlib.use('Agg')


args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name = args[3].strip()
GoGaussian = False#(args[4].strip()=='gaussian')
sd = int(args[4].strip())

Lams = {}
Lams['roger_mod1_box'] = np.sqrt(10)
Lams['roger_mod3_box'] = np.sqrt(10)
Lams['roger_mod4_box'] = np.sqrt(10)
Lams['roger_mod1_box_rec2'] = 1
Lams['roger_mod2_box_rec2'] = 1
Lams['roger_mod3_box_rec2'] = 1

Lams['quentin_mod1_box'] = 1
Lams['quentin_mod2_box'] = 1
Lams['shane_mod1_box'] = np.sqrt(10)

Lams['roger_mod1_maze'] = np.sqrt(10)
Lams['roger_mod3_maze'] = 1
Lams['roger_mod4_maze'] = 1
Lams['quentin_mod1_maze'] = 1
Lams['quentin_mod2_maze'] = 1
Lams['shane_mod1_maze'] = np.sqrt(10)


Lams['roger_mod1_box_space'] = np.sqrt(10)
Lams['roger_mod3_box_space'] = np.sqrt(10)
Lams['roger_mod4_box_space'] = np.sqrt(10)
Lams['roger_mod1_box_rec2_space'] = 1
Lams['roger_mod2_box_rec2_space'] = 1
Lams['roger_mod3_box_rec2_space'] = 1

Lams['quentin_mod1_box_space'] = 1
Lams['quentin_mod2_box_space'] = 1
Lams['shane_mod1_box_space'] = 1

Lams['roger_mod1_maze_space'] = 1
Lams['roger_mod3_maze_space'] = 1
Lams['roger_mod4_maze_space'] = 1
Lams['quentin_mod1_maze_space'] = 1
Lams['quentin_mod2_maze_space'] = 1
Lams['shane_mod1_maze_space'] = np.sqrt(10)



if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
if GoGaussian:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = 15)
else:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = False)
file_name = rat_name + '_' + mod_name + '_' + sess_name

LAM = Lams[file_name]

xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)

sspikes = sspikes[speed>2.5,:]
xx = xx[speed>2.5]
yy = yy[speed>2.5]

T, num_neurons = np.shape(sspikes)
xxyy = np.zeros((len(xx),2))
xxyy[:,0] = xx+0
xxyy[:,1] = yy+0

nF2 = 3
num_bins = 15



LOOtorscores = np.zeros((num_neurons))

np.random.seed(sd)
len_scale = (paras_all[file_name][1] + paras_all[file_name][2])/2*100
noisestd = (1.5/len_scale)*2.*np.pi

                
import glob
#filenames = glob.glob('GLM_info_res4/' + file_name + '_1_G_seed' + str(sd) + 'LAM' + str(int(LAM)) + '_deviance.npz')
if 1 == 1:#len(filenames) == 0:

    f = np.load('GLM_info_res4/' + file_name + '_1_P_seed' + str(sd) + 'LAM' + str(int(LAM)) +  '.npz', allow_pickle = True) 
    P_tor_all = f['P_tor_all']
    f.close()

    filenames = glob.glob('GLM_info_res2/' + file_name + '_coords*')
    coordsall = {}

    if len(filenames) == 1:
        f = np.load('GLM_info_res2/' + file_name + '_coords.npz', allow_pickle = True) 
        coordsall = f['coordsall'][()]
        f.close()
    else:
        for i in range(len(filenames)):
            f = np.load(filenames[i], allow_pickle = True)
            coordsall.update(f['coordstemp'][()])
            f.close()
    for n in np.arange(0, num_neurons, 1): 
        coords = coordsall[str(n)].copy()
        coords2 = (coords+np.random.normal(0, noisestd, np.shape(coords)))%(2*np.pi)
        inds = np.ones(len(sspikes[0,:]))
        inds[n] = 0
        inds = np.where(inds)[0]
        times = np.where(np.sum(sspikes[:,inds]>0,1)>0)[0]
        LOOtorscores[n] = compute_deviance(coords2[times,:], sspikes[times,n], GoGaussian, nF2, P_tor_all[n,:,:], [])
else:
    f = np.load(filenames[0], allow_pickle = True)
    LOOtorscores = f['LOOtorscores']
    f.close()

if sd == 0:
    LAM_space = Lams[file_name + '_space']
    f = np.load('GLM_info_res4/' + file_name + '_2_P_sd' + str(int(LAM_space)) + '.npz',allow_pickle = True) 
    Lvals_space = f['Lvals_space']
    f.close()
    spacescores = np.zeros((num_neurons))
    for n in np.arange(0, num_neurons, 1): 
        inds = np.ones(len(sspikes[0,:]))
        inds[n] = 0
        inds = np.where(inds)[0]
        times = np.where(np.sum(sspikes[:,inds]>0,1)>0)[0]
        spacescores[n] = compute_deviance(xxyy[times,:], sspikes[times,n], GoGaussian, nF2, [], Lvals_space[n])
    if GoGaussian:
        np.savez_compressed('GLM_info_res4/' + file_name + '_1_G_seed' + str(sd) + 'LAM' + str(int(LAM)) + '_deviance1', LOOtorscores = LOOtorscores, spacescores = spacescores)
    else:
        np.savez_compressed('GLM_info_res4/' + file_name + '_1_P_seed' + str(sd) + 'LAM' + str(int(LAM)) + '_deviance1', LOOtorscores = LOOtorscores, spacescores = spacescores)

else:
    if GoGaussian:
        np.savez_compressed('GLM_info_res4/' + file_name + '_1_G_seed' + str(sd) + 'LAM' + str(int(LAM)) + '_deviance1', LOOtorscores = LOOtorscores)
    else:
        np.savez_compressed('GLM_info_res4/' + file_name + '_1_P_seed' + str(sd) + 'LAM' + str(int(LAM)) + '_deviance1', LOOtorscores = LOOtorscores)

######### Run glm res ###########

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
matplotlib.use('Agg')


args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name = args[3].strip()
GoGaussian = (args[4].strip()=='gaussian')
LAM = np.sqrt(float(args[5].strip()))


if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
if GoGaussian:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = 15)
else:
    sspikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = False,)
file_name = rat_name + '_' + mod_name + '_' + sess_name
xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
xx = xx[speed>2.5]
yy = yy[speed>2.5]
sspikes = sspikes[speed>2.5,:]
T, num_neurons = np.shape(sspikes)
xxyy = np.zeros((len(xx),2))
xxyy[:,0] = xx+0
xxyy[:,1] = yy+0


nF2 = 3
ypt_all = []#np.zeros_like(sspikes)
Lvals_tor = []#np.zeros_like(sspikes)
yps_all = []#np.zeros_like(sspikes)
Lvals_space = []#np.zeros_like(sspikes)
num_bins = 15
P_tor_all = np.zeros((num_neurons, num_bins**2, nF2) )
P_space_all = np.zeros((num_neurons, num_bins**2, nF2) )
LOOtorscores = np.zeros((num_neurons))
spacescores = np.zeros((num_neurons))

    
f = np.load('Data/' + file_name + '_newdecoding.npz',  allow_pickle = True)
coords2 = f['coordsnew']
f.close()
times = np.where(np.sum(sspikes>0,1)>0)[0]
for n in np.arange(0, num_neurons, 1): 
    ypt_all.append([])
    Lvals_tor.append([])
    yps_all.append([])
    Lvals_space.append([])

    ypt_all[n], LOOtorscores[n], P_tor_all[n,:, :], Lvals_tor[n] = dirtyglm(coords2, sspikes[times,n], num_bins, True, LAM, GoGaussian, nF2)
#    yps_all[n], spacescores[n], P_space_all[n,:, :], Lvals_space[n] = dirtyglm(xxyy[times,:], sspikes[times,n], num_bins, False, LAM, GoGaussian, nF2)
    
    np.savez_compressed(file_name + '_GLM' , y = ypt_all, L = Lvals_tor)


######### Run glm 1 ###########

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
matplotlib.use('Agg')

args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name = args[3].strip()
GoGaussian = (args[4].strip()=='gaussian')
LAM = np.sqrt(float(args[5].strip()))

if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
if GoGaussian:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = 15)
else:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = False)
file_name = rat_name + '_' + mod_name + '_' + sess_name
#times = np.where(np.sum(sspikes>0,1)>0)[0]
#sspikes = sspikes[times,:]

T, num_neurons = np.shape(sspikes)

nF2 = 3
ypt_all = []#np.zeros_like(sspikes)
Lvals_tor = []#np.zeros_like(sspikes)

num_bins = 15


f = np.load('GLM_info_res4/' + file_name + '_1_P_sd' + str(int(LAM)) +  '.npz', allow_pickle = True) 
P_tor_all = f['P_tor_all']
f.close()

LOOtorscores = np.zeros((num_neurons))


import glob
filenames = glob.glob('GLM_info_res2/' + file_name + '_coords*')
coordsall = {}
if len(filenames) == 1:
	f = np.load(filenames[0], allow_pickle = True) 
	coordsall = f['coordsall'][()]
	f.close()
else:
	for i in range(len(filenames)):
		f = np.load(filenames[i], allow_pickle = True)
		coordsall.update(f['coordstemp'][()])
		f.close()
		print(coordsall)
f = np.load('GLM_info_res4/' + file_name + '_2_P_sd' + str(int(LAM)) + '.npz',allow_pickle = True) 
Lvals_tor = f['Lvals_tor']
f.close()
for n in np.arange(0, num_neurons, 1): 
    coords2 = coordsall[str(n)].copy()
    inds = np.ones(len(sspikes[0,:]))
    inds[n] = 0
    inds = np.where(inds)[0]
    times = np.where(np.sum(sspikes[:,inds]>0,1)>0)[0]
#    LOOtorscores[n] = compute_deviance(coords2[times,:], sspikes[times,n], GoGaussian, nF2, P_tor_all, [])
    LOOtorscores[n] = compute_deviance(coords2[times,:], sspikes[times,n], GoGaussian, nF2, [], Lvals_tor[n])    

if GoGaussian:
    np.savez_compressed('GLM_info_res4/' + file_name + '_1_G_sd_LAM' + str(int(LAM)) + '_deviance', LOOtorscores = LOOtorscores)
else:
    np.savez_compressed('GLM_info_res4/' + file_name + '_1_P_sd_LAM' + str(int(LAM)) + '_deviance', LOOtorscores = LOOtorscores)

######### Run glm 1 ###########
import numpy as np
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
from scipy.ndimage import gaussian_filter1d
from utils import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist



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

### this is a stupid (but simple) way to make these
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
            if LAM > 0:
                pp = 0.
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


def dirtyglm(xxss, ys, num_bins, periodicprior,  LAM, GoGaussian):
    T = len(xxss[:,0])
    nF = 3
    tmp = np.floor(T/nF)
    xxss[:,0] = normit(xxss[:,0])
    xxss[:,1] = normit(xxss[:,1])

    xvalscores = np.zeros(nF)
    Lvals = np.zeros(T)
    yt = np.zeros(T)
    for i in range(nF):
        fg = np.zeros(T)
        if(i==nF-1):
            fg[(int(tmp*i)):] = 1
        else:
            fg[(int(tmp*i)):(int(tmp*(i+1)))] = 1
        fg = fg<0.5
        X_space = preprocess_dataX2(xxss[fg,:], num_bins)

        P = fitmodel(ys[fg], X_space, LAM, periodicprior, GoGaussian)

        xt = xxss[~fg,:]
        X_test = preprocess_dataX2(xt, num_bins)

        if(GoGaussian):
            yt[~fg] = np.dot(P, X_test)
        
        else:
            H = np.dot(P, X_test)
            expH = np.exp(H)
            yt[~fg] = expH
            finthechat = (np.ravel(np.log(factorial(ys[~fg]))))
            Lvals[~fg] = (np.ravel(ys[~fg]*H - expH)) - finthechat
    
    leastsq = np.sum( (ys-yt)**2 )
    #print('LEAST SQ', leastsq)
    ym = np.mean(ys)
    #return (np.sum((yt-ym)**2) / np.sum((ys-ym)**2))
    return yt, (1. - leastsq/np.sum((ys-ym)**2)), P, Lvals



args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name = args[3].strip()
LAM = sqrt(float(args[4].strip()))
GoGaussian = (args[5].strip()=='gaussian')


if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
if GoGaussian:
    sspikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bBox = False)
else:
    sspikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bBox = False)
file_name = rat_name + '_' + mod_name + '_' + sess_name
xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
xx = xx[speed>2.5]
yy = yy[speed>2.5]


f = np.load('Results/Orig/'+file_name + '_pers_analysis.npz', allow_pickle = True)
cycle = f['coords']
movetimes = f['movetimes']
indstemp = f['indstemp']
f.close()


f = np.load('Info/' + file_name + '_LOO.npz', allow_pickle = True)
coordsnew = f['coordsnew'] 
coordsnewall = f['coordsnewall']
f.close()

num_neurons = len(sspikes[0,:])

LOOtorscores = np.zeros(num_neurons)
spacescores = np.zeros(num_neurons)
LOOtorscores[:] = np.nan
spacescores[:] = np.nan

pcaspikes = preprocessing.scale(sspikes)
pcacomps, evecs, evals = pca(pcaspikes, dim=10)

xxyy = np.zeros((len(xx),2))
xxyy[:,0] = xx+0
xxyy[:,1] = yy+0


ypt_all = np.zeros_like(sspikes)
Lvals_tor = np.zeros_like(sspikes)
yps_all = np.zeros_like(sspikes)
Lvals_space = np.zeros_like(sspikes)

num_bins = 15
P_tor_all = np.zeros((num_neurons, num_bins**2) )
P_space_all = np.zeros((num_neurons, num_bins**2) )

for n in np.arange(0, num_neurons, 1):
    coords2 = coordsnewall[n].copy()
    ypt_all[:,n], LOOtorscores[n], P_tor_all[n,:], Lvals_tor[:,n] = dirtyglm(coords2.T, sspikes[:,n], num_bins, True, LAM, GoGaussian)
    yps_all[:,n], spacescores[n], P_space_all[n,:], Lvals_space[:,n] = dirtyglm(xxyy, sspikes[:,n], num_bins, False, LAM, GoGaussian)
    
if GoGaussian:
    np.savez_compressed('GLM_results/' + file_name + '_1_G_LAM' + str(int(LAM)), P_tor_all = P_tor_all, LOOtorscores = LOOtorscores, spacescores = spacescores)
    np.savez_compressed('GLM_results/' + file_name + '_2_G_LAM' + str(int(LAM)), ypt_all = ypt_all, yps_all = yps_all, Lvals_space = Lvals_space, Lvals_tor = Lvals_tor)
else:
    np.savez_compressed('GLM_results/' + file_name + '_1_P_LAM' + str(int(LAM)), P_tor_all = P_tor_all, LOOtorscores = LOOtorscores, spacescores = spacescores)
    np.savez_compressed('GLM_results/' + file_name + '_2_P_LAM' + str(int(LAM)), ypt_all = ypt_all, yps_all = yps_all, Lvals_space = Lvals_space, Lvals_tor = Lvals_tor)


############ run glm sleep 1 ##########

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
matplotlib.use('Agg')

rat_name = str(sys.argv[1].strip())
mod_name = str(sys.argv[2].strip())
sess_name = str(sys.argv[3].strip()) 
GoGaussian = (sys.argv[4].strip()=='gaussian')
LAM = np.sqrt(float(sys.argv[5].strip()))

if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
file_name = rat_name + '_' + mod_name + '_' + sess_name

if sess_name == 'sws_c0_rec2':    
    sspikes = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = True, sig = 15)
    spikes = load_spikes(rat_name, mod_name, 'sws_rec2',  bSmooth = False)
else:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = 15)
    spikes = load_spikes(rat_name, mod_name, sess_name,  bSmooth = False)

if sess_name[:3] in ('box', 'maz'):
    xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
    sspikes = sspikes[speed>2.5,:]
#    spikes = spikes[speed>2.5,:]

num_neurons = len(spikes[0,:])


T, num_neurons = np.shape(sspikes)

nF2 = 3
num_bins = 15#int(sys.argv[4].strip()) 
ypt_all = []#np.zeros_like(sspikes)
Lvals_tor = []#np.zeros_like(sspikes)
yps_all = []#np.zeros_like(sspikes)
Lvals_space = []#np.zeros_like(sspikes)
P_tor_all = np.zeros((num_neurons, num_bins**2, nF2) )
P_space_all = np.zeros((num_neurons, num_bins**2, nF2) )
LOOtorscores = np.zeros((num_neurons))
spacescores = np.zeros((num_neurons))

f = np.load('Data/' + file_name + '_newdecoding.npz', allow_pickle = True)
centcosall = f['centcosall']
centsinall = f['centsinall']
f.close()

T = len(sspikes[:,0])
dspk = preprocessing.scale(sspikes[:,:]).copy()
a = np.zeros((T, 2, num_neurons))
for n in range(num_neurons):
    a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))


c = np.zeros((T, 2, num_neurons))
for n in range(num_neurons):
    c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))

for n in np.arange(0, num_neurons, 1): 
    ypt_all.append([])
    Lvals_tor.append([])
    inds = np.ones(num_neurons)
    inds[n] = 0
    inds = np.where(inds)[0]
    times = np.where(np.sum(spikes[:,inds]>0, 1)>=1)[0]
    mtot2 = np.sum(c[:,:,inds],2)
    mtot1 = np.sum(a[:,:,inds],2)
    coords2 = np.arctan2(mtot2,mtot1)%(2*np.pi)


    ypt_all[n], LOOtorscores[n], P_tor_all[n,:, :], Lvals_tor[n] = dirtyglm(coords2[times,:], spikes[times,n], num_bins, True, LAM, GoGaussian, nF2)
    LOOtorscores[n] = compute_deviance(coords2[times,:], spikes[times,n], GoGaussian, nF2, [], Lvals_tor[n])    

if GoGaussian:
    np.savez_compressed('GLM_info_res8/' + file_name + '_1_G_sd_LAM' + str(int(LAM)) + '_deviance', LOOtorscores = LOOtorscores)
else:
    np.savez_compressed('GLM_info_res8/' + file_name + '_1_P_sd_LAM' + str(int(LAM)) + '_deviance', LOOtorscores = LOOtorscores)
    np.savez_compressed('GLM_info_res8/' + file_name + '_1_P_sd_LAM' + str(int(LAM)) + '_data', P = P_tor_all, L = Lvals_tor, y = ypt_all)

    ############ run glm ##########
    
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
matplotlib.use('Agg')


args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name = args[3].strip()
GoGaussian = False#(args[4].strip()=='gaussian')
sd = int(args[4].strip())

Lams = {}
Lams['roger_mod1_box'] = np.sqrt(10)
Lams['roger_mod3_box'] = np.sqrt(10)
Lams['roger_mod4_box'] = np.sqrt(10)

Lams['quentin_mod1_box'] = 1
Lams['quentin_mod2_box'] = 1
Lams['shane_mod1_box'] = np.sqrt(10)

Lams['roger_mod1_maze'] = np.sqrt(10)
Lams['roger_mod3_maze'] = 1
Lams['roger_mod4_maze'] = 1
Lams['quentin_mod1_maze'] = 1
Lams['quentin_mod2_maze'] = 1
Lams['shane_mod1_maze'] = np.sqrt(10)


if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
if GoGaussian:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = 15)
else:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = False)
file_name = rat_name + '_' + mod_name + '_' + sess_name

LAM = Lams[file_name]

xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)

sspikes = sspikes[speed>2.5,:]
xx = xx[speed>2.5]
yy = yy[speed>2.5]

T, num_neurons = np.shape(sspikes)
xxyy = np.zeros((len(xx),2))
xxyy[:,0] = xx+0
xxyy[:,1] = yy+0

nF2 = 3
num_bins = 15



LOOtorscores = np.zeros((num_neurons))

np.random.seed(sd)
len_scale = (paras_all[file_name][1] + paras_all[file_name][2])/2*100
noisestd = (1.5/len_scale)*2.*np.pi

f = np.load('GLM_info_res4/' + file_name + '_1_P_seed' + str(sd) + 'LAM' + str(int(LAM)) +  '.npz', allow_pickle = True) 
P_tor_all = f['P_tor_all']
f.close()
                
import glob
filenames = glob.glob('GLM_info_res2/' + file_name + '_coords*')
coordsall = {}

if len(filenames) == 1:
    f = np.load('GLM_info_res2/' + file_name + '_coords.npz', allow_pickle = True) 
    coordsall = f['coordsall'][()]
    f.close()
else:
    for i in range(len(filenames)):
        f = np.load(filenames[i], allow_pickle = True)
        coordsall.update(f['coordstemp'][()])
        f.close()

for n in np.arange(0, num_neurons, 1): 
    coords = coordsall[str(n)].copy()
    coords2 = (coords+np.random.normal(0, noisestd, np.shape(coords)))%(2*np.pi)
    inds = np.ones(len(sspikes[0,:]))
    inds[n] = 0
    inds = np.where(inds)[0]
    times = np.where(np.sum(sspikes[:,inds]>0,1)>0)[0]
    LOOtorscores[n] = compute_deviance(coords2[times,:], sspikes[times,n], GoGaussian, nF2, P_tor_all[n,:,:], [])
if sd == 0:
    f = np.load('GLM_info_res4/' + file_name + '_2_P_sd' + str(int(LAM)) + '.npz',allow_pickle = True) 
    Lvals_space = f['Lvals_space']
    f.close()
    spacescores = np.zeros((num_neurons))
    for n in np.arange(0, num_neurons, 1): 
        inds = np.ones(len(sspikes[0,:]))
        inds[n] = 0
        inds = np.where(inds)[0]
        times = np.where(np.sum(sspikes[:,inds]>0,1)>0)[0]
        spacescores[n] = compute_deviance(xxyy[times,:], sspikes[times,n], GoGaussian, nF2, [], Lvals_space[n])
    if GoGaussian:
        np.savez_compressed('GLM_info_res4/' + file_name + '_1_G_seed' + str(sd) + 'LAM' + str(int(LAM)) + '_deviance', LOOtorscores = LOOtorscores, spacescores = spacescores)
    else:
        np.savez_compressed('GLM_info_res4/' + file_name + '_1_P_seed' + str(sd) + 'LAM' + str(int(LAM)) + '_deviance', LOOtorscores = LOOtorscores, spacescores = spacescores)

else:
    if GoGaussian:
        np.savez_compressed('GLM_info_res4/' + file_name + '_1_G_seed' + str(sd) + 'LAM' + str(int(LAM)) + '_deviance', LOOtorscores = LOOtorscores)
    else:
        np.savez_compressed('GLM_info_res4/' + file_name + '_1_P_seed' + str(sd) + 'LAM' + str(int(LAM)) + '_deviance', LOOtorscores = LOOtorscores)


# In[ ]:


############ run H1 ##########

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
matplotlib.use('Agg')

rat_name = str(sys.argv[1].strip())
mod_name = str(sys.argv[2].strip())
sess_name = str(sys.argv[3].strip()) 
if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
sspikes, sspikes_box, xx, yy = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bBox = True)
file_name = rat_name + '_' + mod_name + '_' + sess_name

################### Hyperparameters ####################
dim = 6
ph_classes = [0,1] # Decode the ith most persistent cohomology class
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 2
coeff = 47
num_neurons = len(sspikes[0,:])
active_times = 15000
k = 1000
num_times = 5
nbs = 250


n_points = 300

times_cube = np.arange(0,len(sspikes[:,0]),num_times)
movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube,:],1))[-active_times:])
movetimes = times_cube[movetimes]

dim_red_spikes_move_scaled,__,__ = pca(preprocessing.scale(sspikes[movetimes,:]), dim = dim)
indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                    n_points, 1, metric)
dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
X = squareform(pdist(dim_red_spikes_move_scaled, metric))
knn_indices, knn_dists, __ = nearest_neighbors(X, n_neighbors = nbs, metric = 'precomputed', angular=True, metric_kwds = {})
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
np.savez(file_name + '_d', d = d)
############ Compute persistent homology ################
rips_real = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=True, distance_matrix = True)
plt.figure()
plot_diagrams(
    rips_real["dgms"],
    plot_only=np.arange(maxdim+1),
    lifetime = True)
plt.savefig('Figs/' + file_name + '_persistence_diagram_H' + str(maxdim))

############ Decode cocycles ################
diagrams = rips_real["dgms"] # the multiset describing the lives of the persistence classes
cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
dists_land = rips_real["dperm2all"] # the pairwise distance between the points 
births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
deaths1[np.isinf(deaths1)] = 0
lives1 = deaths1-births1 # the lifetime for the 1-dim classes
iMax = np.argsort(lives1)
coords1 = np.zeros((num_circ, len(indstemp)))

times = np.arange(0, len(sspikes_box[:,0]), int(len(sspikes_box[:,0])/30000)+1)
call = np.zeros((num_circ, len(times)))
threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]])*dec_tresh


births2 = diagrams[2][:, 0] #the time of birth for the 1-dim classes
deaths2 = diagrams[2][:, 1] #the time of death for the 1-dim classes
deaths2[np.isinf(deaths2)] = 0
lives2 = deaths2-births2 # the lifetime for the 1-dim classes
iMax2 = np.argsort(lives2)

print(deaths1[iMax[-2]], births2[iMax2[-1]])

for c in ph_classes:
    cocycle = cocycles[iMax[-(c+1)]]
    coords1[c,:],inds = get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

    num_sampled = dists_land.shape[1]
    num_tot = np.shape(indstemp)[0]
    # GET CIRCULAR COORDINATES
    call[c,:] = predict_color(coords1[c,:], sspikes_box[times, :], sspikes[movetimes[indstemp],:], 
                               dist_measure=metric, num_batch = 4098, k = 10)



if (sess_name in ('rem', 'sws','sws_c0', 'box_rec2')) & (rat_name == 'roger'):    
    f = np.load('Data/tracking_' + rat_name + '_' + 'box_rec2.npz', allow_pickle = True)
    xx = f['xx']
    yy = f['yy']
    f.close()
else:
    f = np.load('Data/tracking_' + rat_name + '_' + 'box.npz', allow_pickle = True)
    xx = f['xx']
    yy = f['yy']
    f.close()

xxs = gaussian_filter1d(xx-np.min(xx), sigma = 100)
yys = gaussian_filter1d(yy-np.min(yy), sigma = 100)
dx = (xxs[1:] - xxs[:-1])*100
dy = (yys[1:] - yys[:-1])*100
speed = np.sqrt(dx**2+ dy**2)/0.01
speed = np.concatenate(([speed[0]],speed))
xx = xx[speed>2.5]
yy = yy[speed>2.5]
if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
file_name = rat_name + '_' + mod_name + '_' + sess_name

for c in [0,1]:
    plt.figure()
    ax = plt.axes()
    plt.hsv()
    ax.scatter(xx[times], yy[times], c = call[c,:])
    ax.set_aspect('equal', 'box')
    plt.savefig('Figs/' + file_name + '_decoding_' + str(c))
    plt.close()


# In[ ]:


############ run ifno rec ###########

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
matplotlib.use('Agg')

rat_name = str(sys.argv[1].strip())
mod_name = str(sys.argv[2].strip())
sess_name = str(sys.argv[3].strip()) 
numbins = 15#int(sys.argv[4].strip()) 

if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
file_name = rat_name + '_' + mod_name + '_' + sess_name

if sess_name == 'sws_c0_rec2':    
    sspikes = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = True, sig = 15)
    spikes = load_spikes(rat_name, mod_name, 'sws_rec2',  bSmooth = False)
else:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = 15)
    spikes = load_spikes(rat_name, mod_name, sess_name,  bSmooth = False)

if sess_name[:3] in ('box', 'maz'):
    xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
    sspikes = sspikes[speed>2.5,:]

num_neurons = len(spikes[0,:])

#I_1 = np.zeros(num_neurons)
I_5 = np.zeros(num_neurons)
I_5_noise = np.zeros(num_neurons)
I_5_noise_std = np.zeros(num_neurons)
bins = np.linspace(0,2*np.pi, numbins+1)

#f = np.load('Data/' + file_name + '_newdecoding.npz', allow_pickle = True)
#coords = f['coordsnew']
#f.close()

f = np.load('Data/' + file_name + '_newdecoding.npz', allow_pickle = True)
centcosall = f['centcosall']
centsinall = f['centsinall']
f.close()

T = len(sspikes[:,0])
dspk = preprocessing.scale(sspikes[:,:]).copy()
a = np.zeros((T, 2, num_neurons))
for n in range(num_neurons):
    a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))


c = np.zeros((T, 2, num_neurons))
for n in range(num_neurons):
    c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))

for n in range(num_neurons):
    inds = np.ones(len(sspikes[0,:]))
    inds[n] = 0
    inds = np.where(inds)[0]
    times = np.where(np.sum(spikes[:,inds]>0, 1)>=1)[0]
    mtot2 = np.sum(c[:,:,inds],2)
    mtot1 = np.sum(a[:,:,inds],2)
    coords = np.arctan2(mtot2,mtot1)%(2*np.pi)

#    spksum = 100/np.sum(spikes[times,n])
    rIs = np.zeros(100)
    for i in range(len(rIs)):
        spktmp = np.roll(spikes[times,n], int(np.random.rand()*len(times)))
        mtot5, x_edge, y_edge, c5 = binned_statistic_2d(coords[times,0],coords[times,1], 
            spktmp, statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)
        rIs[i] = information_score(mtot5.copy(), c5, spikes[times,n].mean())#*spksum
    I_5_noise[n] = np.mean(rIs)
    I_5_noise_std[n] = np.std(rIs)
    mtot5, x_edge, y_edge, c5 = binned_statistic_2d(coords[times,0],coords[times,1], 
            spikes[times,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)            
    I_5[n] = information_score(mtot5.copy(), c5, spikes[times,n].mean())#*spksum
np.savez_compressed('GLM_info_res7/' + file_name + '_info_sleep', I_5 = I_5, I_5_noise = I_5_noise, I_5_noise_std = I_5_noise_std)

############ run info conj ##########

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
from scipy.stats import binned_statistic_2d, pearsonr
import functools
from scipy import signal
from scipy import optimize
import glob
from utils import *


def information_score(mtemp, circ, mu):
    numangsint = mtemp.shape[0]+1
#    print(np.unique(circ), np.shape(mtemp))
    circ = np.ravel_multi_index(circ-1, np.shape(mtemp))
    mtemp = mtemp.flatten() 

    p = np.bincount(circ, minlength = (numangsint-1)**2)/len(circ)
    logtemp = np.log2(mtemp/mu)
    mtemp = np.multiply(np.multiply(mtemp,p), logtemp)
    return np.sum(mtemp[~np.isnan(mtemp)])


matplotlib.use('Agg')

args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name = args[3].strip()

if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
spikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bConj = True)

numbins = 15#int(sys.argv[4].strip()) 

if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
file_name = rat_name + '_' + mod_name + '_' + sess_name


if sess_name[:3] in ('box', 'maz'):
    xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
    xx = preprocessing.minmax_scale(xx[speed>2.5])
    yy = preprocessing.minmax_scale(yy[speed>2.5])
#    aa = aa[speed>2.5]

num_neurons = len(spikes[0,:])

I_torus = np.zeros(num_neurons)
I_space = np.zeros(num_neurons)
#I_head = np.zeros(num_neurons)
bins = np.linspace(0,1, numbins+1)

len_scale = (paras_all[file_name][1] + paras_all[file_name][2])/2*100
noisestd = (1.5/len_scale)*2.*np.pi
f = np.load('Data/' + file_name + '_newdecoding.npz', allow_pickle = True)
coords = f['coordsnew']
times = f['times']
f.close()
np.random.seed(47)
coords = preprocessing.minmax_scale((coords+np.random.normal(0, noisestd, np.shape(coords)))%(2*np.pi))

for n in range(num_neurons):
    mtot5, x_edge, y_edge, c5 = binned_statistic_2d(coords[:,0],coords[:,1], 
            spikes[times,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)            
    I_torus[n] = information_score(mtot5.copy(), c5, spikes[times,n].mean())#*spksum
    mtot5, x_edge, y_edge, c5 = binned_statistic_2d(xx[times],yy[times], 
            spikes[times,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)            
    I_space[n] = information_score(mtot5.copy(), c5, spikes[times,n].mean())#*spksum

#    mtot5, x_edge, y_edge, c5 = binned_statistic_2d(xx[times],yy[times], 
#            spikes[times,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)            
#    I_space[n] = information_score(mtot5.copy(), c5, spikes[times,n].mean())#*spksum

np.savez_compressed('Info_conj/' + file_name + '_conj_info', I_torus = I_torus, I_space = I_space)

############ run info rec ##########

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
matplotlib.use('Agg')

rat_name = str(sys.argv[1].strip())
mod_name = str(sys.argv[2].strip())
sess_name = str(sys.argv[3].strip()) 
numbins = 15#int(sys.argv[4].strip()) 

if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
file_name = rat_name + '_' + mod_name + '_' + sess_name

if sess_name == 'sws_c0_rec2':    
    sspikes = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = True, sig = 15)
    spikes = load_spikes(rat_name, mod_name, 'sws_rec2',  bSmooth = False)
else:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = 15)
    spikes = load_spikes(rat_name, mod_name, sess_name,  bSmooth = False)

if sess_name[:3] in ('box', 'maz'):
    xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)

num_neurons = len(spikes[0,:])
T = sum(speed>2.5)
sspikes = sspikes[speed>2.5,:]
xx = xx[speed>2.5]
yy = yy[speed>2.5]

len_scale = (paras_all[file_name][1] + paras_all[file_name][2])/2*100
noisestd = (1.5/len_scale)*2.*np.pi

I_1 = np.zeros(num_neurons)
I_5 = np.zeros(num_neurons)
I_5_noise = np.zeros(num_neurons)
I_5_noise_std = np.zeros(num_neurons)
bins = np.linspace(0,2*np.pi, numbins+1)

if sess_name[:3] == 'box':
    file_name1 = rat_name + '_' + mod_name + '_' + 'maze'
else:
    file_name1 = rat_name + '_' + mod_name + '_' + 'box'
    if (rat_name == 'roger') and (sess_name[:3] in ('sws', 'rem')):
        file_name1 += '_rec2'

#f = np.load('Data/' + file_name1 + '_newdecoding.npz', allow_pickle = True)
#centcosall = f['centcosall']
#centsinall = f['centsinall']
#f.close()

#dspk = preprocessing.scale(sspikes[:,:]).copy()
#a = np.zeros((len(xx), 2, num_neurons))
#for n in range(num_neurons):
#    a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))


#c = np.zeros((len(xx), 2, num_neurons))
#for n in range(num_neurons):
#    c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))
#coordstemp = {}

filenames = glob.glob('GLM_info_res2/' + file_name + '_coords*')
coordsall = {}
if len(filenames) == 1:
    f = np.load('GLM_info_res2/' + file_name + '_coords.npz', allow_pickle = True) 
    coordsall = f['coordsall'][()]
    f.close()
else:
    for i in range(len(filenames)):
        f = np.load(filenames[i], allow_pickle = True)
        coordsall.update(f['coordstemp'][()])
        f.close()
times = np.where(np.sum(spikes[:,:]>0, 1)>=1)[0]

for n in range(num_neurons):
    inds = np.ones(len(sspikes[0,:]))
    inds[n] = 0
    inds = np.where(inds)[0]
    mtot2, x_edge, y_edge, c2 = binned_statistic_2d(xx[times],yy[times], 
       spikes[times,n], statistic = 'mean', bins=numbins, range=None, expand_binnumbers=True)
    I_1[n] = information_score(mtot2.copy(), c2, spikes[times,n].mean())*100
#    mtot2 = np.sum(c[:,:,inds],2)
#    mtot1 = np.sum(a[:,:,inds],2)

#    coords = np.arctan2(mtot2,mtot1)%(2*np.pi)

    coords = coordsall[str(n)].copy() 
    rIs = np.zeros(100)
    for i in range(len(rIs)):

        coords5 = (coords+np.random.normal(0, noisestd, np.shape(coords)))%(2*np.pi)

        mtot5, x_edge, y_edge, c5 = binned_statistic_2d(coords5[times,0],coords5[times,1], 
            spikes[times,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)
        rIs[i] = information_score(mtot5.copy(), c5, spikes[times,n].mean())*100
    I_5_noise[n] = np.mean(rIs)
    I_5_noise_std[n] = np.std(rIs)

    mtot5, x_edge, y_edge, c5 = binned_statistic_2d(coords[times,0],coords[times,1], 
            spikes[times,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)            
    I_5[n] = information_score(mtot5.copy(), c5, spikes[times,n].mean())*100
#    coordsall[str(n)] = coords
np.savez_compressed('GLM_info_res4/' + file_name + '_info', I_1 = I_1, I_5 = I_5, I_5_noise = I_5_noise, I_5_noise_std = I_5_noise_std)
#np.savez_compressed('GLM_info_res3/' + file_name + '_coords', coordsall = coordsall, force_zip64=True)

############ run info shuf ##########

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
matplotlib.use('Agg')

rat_name = str(sys.argv[1].strip())
mod_name = str(sys.argv[2].strip())
sess_name = str(sys.argv[3].strip()) 
numbins = 15#int(sys.argv[4].strip()) 

if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
file_name = rat_name + '_' + mod_name + '_' + sess_name

if sess_name == 'sws_c0_rec2':    
    sspikes = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = True, sig = 15)
    spikes = load_spikes(rat_name, mod_name, 'sws_rec2',  bSmooth = False)
else:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = 15)
    spikes = load_spikes(rat_name, mod_name, sess_name,  bSmooth = False)

if sess_name[:3] in ('maz', 'box'):
    xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
    sspikes = sspikes[speed>2.5,:]
    xx = xx[speed>2.5]
    yy = yy[speed>2.5]

num_neurons = len(spikes[0,:])

len_scale = (paras_all[file_name][1] + paras_all[file_name][2])/2*100
noisestd = (1.5/len_scale)*2.*np.pi

coordsall = {}
num_shufs = 100
I_shuf = np.zeros((num_neurons, num_shufs))
bins = np.linspace(0,2*np.pi, numbins+1)

file_name1 = rat_name + '_' + mod_name + '_' + 'box'
if (rat_name == 'roger') and (sess_name[:3] in ('sws', 'rem')):
    file_name1 += '_rec2'

f = np.load('Data/' + file_name1 + '_newdecoding.npz', allow_pickle = True)
centcosall = f['centcosall']
centsinall = f['centsinall']
f.close()

dspk = preprocessing.scale(sspikes[:,:]).copy()
a = np.zeros((len(spikes[:,0]), 2, num_neurons))
for n in range(num_neurons):
    a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))


c = np.zeros((len(spikes[:,0]), 2, num_neurons))
for n in range(num_neurons):
    c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))
coordstemp = {}
np.random.seed(47)
import glob
filenames = glob.glob('GLM_info_res2/' + file_name + '_coords*')
coordsall = {}

if len(filenames) == 1:
    f = np.load('GLM_info_res2/' + file_name + '_coords.npz', allow_pickle = True) 
    coordsall = f['coordsall'][()]
    f.close()
else:
    for i in range(len(filenames)):
        f = np.load(filenames[i], allow_pickle = True)
        coordsall.update(f['coordstemp'][()])
        f.close()

for n in np.arange(0, num_neurons, 1): 
    coords = coordsall[str(n)].copy()

    inds = np.ones(len(sspikes[0,:]))
    inds[n] = 0
    inds = np.where(inds)[0]
    times = np.where(np.sum(spikes[:,inds]>0, 1)>=1)[0]
    for i in range(num_shufs):
        spktmp = np.roll(spikes[times,n], int(np.random.rand()*len(times)))
        mtot5, x_edge, y_edge, c5 = binned_statistic_2d(coords[times,0],coords[times,1], 
            spktmp, statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)
        I_shuf[n,i] = information_score(mtot5.copy(), c5, spikes[times,n].mean())*100
np.savez_compressed('GLM_info_res2/' + file_name + '_info_shuf', I_shuf = I_shuf)


# In[ ]:


############ run info sleep ###############
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
matplotlib.use('Agg')

rat_name = str(sys.argv[1].strip())
mod_name = str(sys.argv[2].strip())
sess_name = str(sys.argv[3].strip()) 
numbins = 15#int(sys.argv[4].strip()) 

if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
file_name = rat_name + '_' + mod_name + '_' + sess_name

if sess_name == 'sws_c0_rec2':    
    sspikes = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = True, sig = 15)
    spikes = load_spikes(rat_name, mod_name, 'sws_rec2',  bSmooth = False)
else:
    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = 15)
    spikes = load_spikes(rat_name, mod_name, sess_name,  bSmooth = False)

num_neurons = len(spikes[0,:])
coordsall = {}
I_5 = np.zeros(num_neurons)
I_5_noise = np.zeros(num_neurons)
I_5_noise_std = np.zeros(num_neurons)
bins = np.linspace(0,2*np.pi, numbins+1)

file_name1 = rat_name + '_' + mod_name + '_' + 'box'
if (rat_name == 'roger') and (sess_name[:3] in ('sws', 'rem')):
    file_name1 += '_rec2'

#f = np.load('Data/' + file_name1 + '_newdecoding.npz', allow_pickle = True)
#centcosall = f['centcosall']
#centsinall = f['centsinall']
#f.close()

#dspk = preprocessing.scale(sspikes[:,:]).copy()
#a = np.zeros((len(spikes[:,0]), 2, num_neurons))
#for n in range(num_neurons):
#    a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))


#c = np.zeros((len(spikes[:,0]), 2, num_neurons))
#for n in range(num_neurons):
#    c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))
#coordstemp = {}
filenames = glob.glob('GLM_info_res2/' + file_name + '_coords*')
coordsall = {}
if len(filenames) == 1:
    f = np.load('GLM_info_res2/' + file_name + '_coords.npz', allow_pickle = True) 
    coordsall = f['coordsall'][()]
    f.close()
else:
    for i in range(len(filenames)):
        f = np.load(filenames[i], allow_pickle = True)
        coordsall.update(f['coordstemp'][()])
        f.close()
times = np.where(np.sum(spikes[:,:]>0, 1)>=1)[0]

for n in range(num_neurons):
    inds = np.ones(len(sspikes[0,:]))
    inds[n] = 0
    inds = np.where(inds)[0]
#    times = np.where(np.sum(spikes[:,inds]>0, 1)>=1)[0]

#    mtot2 = np.sum(c[:,:,inds],2)
#    mtot1 = np.sum(a[:,:,inds],2)

#    coords = np.arctan2(mtot2,mtot1)%(2*np.pi)
    coords = coordsall[str(n)].copy() 
    mtot5, x_edge, y_edge, c5 = binned_statistic_2d(coords[times,0],coords[times,1], 
            spikes[times,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)            
    I_5[n] = information_score(mtot5.copy(), c5, spikes[times,n].mean())*100
np.savez_compressed('GLM_info_res4/' + file_name + '_info', I_5 = I_5)


# In[ ]:


########### Run info 15 ############

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
matplotlib.use('Agg')

rat_name = str(sys.argv[1].strip())
mod_name = str(sys.argv[2].strip())
sess_name = str(sys.argv[3].strip()) 

file_name = rat_name + '_' + mod_name + '_' + sess_name
f = np.load('Data/' + file_name + '_spk_times.npz', allow_pickle = True) # or change to roger_mod3_spk_times.npz
spikes_mod21 = f['spiketimes'][()]
f.close()

times_all = {}
times_all['quentin_box'] = (27826, 31223)
times_all['quentin_sleep'] = (9576, 18812)
times_all['quentin_maze'] = (18977, 25355)
times_all['shane_box'] = (9939, 12363)
times_all['shane_maze'] = (13670, 14847)
times_all['shane_maze2'] = (23186, 24936)
times_all['roger_box_rec2'] = (10617, 13004)
times_all['roger_sleep_rec2'] = (396, 9941)
times_all['roger_box'] = (7457, 16045)
times_all['roger_maze'] = (16925, 20704)
start1, end1 = times_all[rat_name + '_' + sess_name]

res = 100000
sigma = 1500
thresh = sigma*5
dt = 1000

num_thresh = int(thresh/dt)
num2_thresh = int(2*num_thresh)
sig2 = 1/(2*(sigma/res)**2)
ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
kerwhere = np.arange(-num_thresh,num_thresh)*dt

min_time = start1*res
max_time = end1*res

sspikes_mod21 = np.zeros((1,len(spikes_mod21)))
tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

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
    res = 100000
    dt = 1000

    valid_times = np.concatenate((np.arange(0, (14778-7457)*res/dt),
                                  np.arange((14890-7457)*res/dt, (16045-7457)*res/dt))).astype(int)
    sspikes_mod21 = sspikes_mod21[valid_times,:]
    spikes_mod21_bin = spikes_mod21_bin[valid_times,:]
elif rat_name + sess_name == 'rogermaze':
    valid_times = np.concatenate((np.arange(0, (18026-16925)*res/dt),
                                  np.arange((18183-16925)*res/dt, (20704-16925)*res/dt))).astype(int)
    sspikes_mod21 = sspikes_mod21[valid_times,:]
    spikes_mod21_bin = spikes_mod21_bin[valid_times,:]
elif rat_name + sess_name == 'shanemaze':
    file_name1 = rat_name + '_' + mod_name + '_' + sess_name + '2'
    f = np.load('Data/' + file_name1 + '_spk_times.npz', allow_pickle = True) # or change to roger_mod3_spk_times.npz
    spikes_mod22 = f['spiketimes'][()]
    f.close()

    start1, end1 = times_all[rat_name + '_' + sess_name + '2']

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
    
    sspikes_mod21 = np.concatenate((sspikes_mod21, sspikes_mod22),0)
    spikes_mod21_bin = np.concatenate((spikes_mod21_bin, spikes_mod22_bin),0)




xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
sspikes = sspikes_mod21
sspikes = sspikes[speed>2.5,:]        
spikes = spikes_mod21_bin
spikes = spikes[speed>2.5,:]      
xx = xx[speed>2.5]
yy = yy[speed>2.5]  


dspk = preprocessing.scale(sspikes).copy()
num_neurons = len(spikes[0,:])

centcosall = np.zeros((num_neurons, 2, 1200))
centsinall = np.zeros((num_neurons, 2, 1200))

file_name = rat_name + '_' + mod_name + '_' + sess_name
f = np.load('Results/Orig/' + file_name + '_pers_analysis.npz', allow_pickle = True)
cycle = f['coords']
movetimes = f['movetimes']
indstemp = f['indstemp']
f.close()
for neurid in range(num_neurons):
    spktemp = dspk[movetimes[indstemp],neurid].copy()
    centcosall[neurid,:,:] = np.multiply(np.cos(cycle*2*np.pi),spktemp)
    centsinall[neurid,:,:] = np.multiply(np.sin(cycle*2*np.pi),spktemp)


a = np.zeros((len(xx), 2, num_neurons))
for n in range(num_neurons):
    a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))


c = np.zeros((len(xx), 2, num_neurons))
for n in range(num_neurons):
    c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))

mtot2 = np.sum(c,2)
mtot1 = np.sum(a,2)

coordsnew = np.arctan2(mtot2,mtot1)%(2*np.pi)

I_1 = np.zeros(num_neurons)
I_4 = np.zeros(num_neurons)
I_5 = np.zeros(num_neurons)
coordsnewall = []
numbins = 15
bins = np.linspace(0,2*np.pi, numbins+1)
for n in range(num_neurons):
    inds = np.ones(len(sspikes[0,:]))
    inds[n] = 0
    inds = np.where(inds)[0]
    fig1 = plt.figure()
    mtot2, x_edge, y_edge, c2 = binned_statistic_2d(xx[:],yy[:], 
       spikes[:,n], statistic = 'mean', bins=20, range=None, expand_binnumbers=True)
    I_1[n] = information_score(mtot2.copy(), c2, spikes[:,n].mean())
    ax11 = fig1.add_subplot(131)
    ax11.set_xticks([],[])
    ax11.set_yticks([],[])
    ax11.set_title('I=' + str(np.round(I_1[n]*100,2)))
    ax11.imshow(mtot2)
    
    mtot2, x_edge, y_edge, c2 = binned_statistic_2d(coordsnew[:,0],coordsnew[:,1], 
        spikes[:,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)
    I_4[n] = information_score(mtot2.copy(), c2, spikes[:,n].mean())
    ax4 = fig1.add_subplot(132)
    ax4.set_xticks([],[])
    ax4.set_yticks([],[])
    ax4.imshow(mtot2)
    ax4.set_title('I=' + str(np.round(I_4[n]*100,2)))
    
    mtot2 = np.sum(c[:,:,inds],2)
    mtot1 = np.sum(a[:,:,inds],2)

    coords = np.arctan2(mtot2,mtot1)%(2*np.pi)

    mtot2, x_edge, y_edge, c2 = binned_statistic_2d(coords[:,0],coords[:,1], 
        spikes[:,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)
    I_5[n] = information_score(mtot2.copy(), c2, spikes[:,n].mean())
    ax5 = fig1.add_subplot(133)
    ax5.set_xticks([],[])
    ax5.set_yticks([],[])
    ax5.imshow(mtot2)
    ax5.set_title('I=' + str(np.round(I_5[n]*100,2)))
    fig1.savefig('Info15/rate_maps/' + file_name + '_new_n' + str(n),bbox_inches='tight', pad_inches=0)

    coordsnewall.append(coords)
    
    print(n)
    plt.close(fig1)


I_11 = np.zeros(num_neurons)
I_41 = np.zeros(num_neurons)
I_51 = np.zeros(num_neurons)
for n in range(num_neurons):
    inds = np.ones(len(sspikes[0,:]))
    inds[n] = 0
    inds = np.where(inds)[0]
    times = np.where(np.sum(spikes[:,inds]>0, 1)>=2)[0]
    fig1 = plt.figure()
    mtot2, x_edge, y_edge, c2 = binned_statistic_2d(xx[times],yy[times], 
       spikes[times,n], statistic = 'mean', bins=20, range=None, expand_binnumbers=True)
    I_11[n] = information_score(mtot2.copy(), c2, spikes[times,n].mean())
    ax11 = fig1.add_subplot(131)
    ax11.set_xticks([],[])
    ax11.set_yticks([],[])
    ax11.set_title('I=' + str(np.round(I_11[n]*100,2)))
    ax11.imshow(mtot2)
    
    mtot2, x_edge, y_edge, c2 = binned_statistic_2d(coordsnew[times,0],coordsnew[times,1], 
        spikes[times,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)
    I_41[n] = information_score(mtot2.copy(), c2, spikes[times,n].mean())
    ax4 = fig1.add_subplot(132)
    ax4.set_xticks([],[])
    ax4.set_yticks([],[])
    ax4.imshow(mtot2)
    ax4.set_title('I=' + str(np.round(I_41[n]*100,2)))
    
    coords = coordsnewall[n].copy()
    mtot2, x_edge, y_edge, c2 = binned_statistic_2d(coords[times,0],coords[times,1], 
        spikes[times,n], statistic = 'mean', bins=bins, range=None, expand_binnumbers=True)
    I_51[n] = information_score(mtot2.copy(), c2, spikes[times,n].mean())
    ax5 = fig1.add_subplot(133)
    ax5.set_xticks([],[])
    ax5.set_yticks([],[])
    ax5.imshow(mtot2)
    ax5.set_title('I=' + str(np.round(I_51[n]*100,2)))
    fig1.savefig('Info15/rate_maps_times/' + file_name + '_newtimes_n' + str(n),bbox_inches='tight', pad_inches=0)

    print(n)
    plt.show()

np.savez_compressed('Info15/' + file_name + '_info_coords_15', I_1 = I_1,
                     I_4 = I_4, I_5 = I_5, I_11 = I_11, I_41 = I_41, I_51 = I_51,
                    coordsnewall = coordsnewall, coordsnew = coordsnew)



from scipy.stats import wilcoxon

maxEV =  max(np.max(I_5), np.max(I_1))
minEV =  min(np.min(I_5), np.min(I_1))
fig = plt.figure()
ax8 = fig.add_subplot(111)
ax8.scatter(I_1, I_5, s = 10, c = 'orange')

ax8.plot([minEV, maxEV], [minEV, maxEV], c='k', zorder = -5)
ax8.set_xlim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_ylim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])

w, p = wilcoxon(I_5-I_1, alternative='greater')
plt.title(file_name)
plt.text(minEV + 0.01*(maxEV-minEV), 0.9*maxEV, 'p = ' + str(np.round(p,4)))
plt.savefig('Info15/' + file_name + 'LOO')


from scipy.stats import wilcoxon

maxEV =  max(np.max(I_51), np.max(I_11))
minEV =  min(np.min(I_51), np.min(I_11))
fig = plt.figure()
ax8 = fig.add_subplot(111)
ax8.scatter(I_11, I_51, s = 10, c = 'orange')

ax8.plot([minEV, maxEV], [minEV, maxEV], c='k', zorder = -5)
ax8.set_xlim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_ylim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])

w, p = wilcoxon(I_51-I_11, alternative='greater')
plt.title(file_name)
plt.text(minEV + 0.01*(maxEV-minEV), 0.9*maxEV, 'p = ' + str(np.round(p,4)))
plt.savefig('Info15/' + file_name + 'LOOtimes')



from scipy.stats import wilcoxon

maxEV =  max(np.max(I_4), np.max(I_1))
minEV =  min(np.min(I_4), np.min(I_1))
fig = plt.figure()
ax8 = fig.add_subplot(111)
ax8.scatter(I_1, I_4, s = 10, c = 'orange')

ax8.plot([minEV, maxEV], [minEV, maxEV], c='k', zorder = -5)
ax8.set_xlim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_ylim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])

w, p = wilcoxon(I_4-I_1, alternative='greater')
plt.title(file_name)
plt.text(minEV + 0.01*(maxEV-minEV), 0.9*maxEV, 'p = ' + str(np.round(p,4)))
plt.savefig('Info15/' + file_name + 'all')


from scipy.stats import wilcoxon

maxEV =  max(np.max(I_41), np.max(I_11))
minEV =  min(np.min(I_41), np.min(I_11))
fig = plt.figure()
ax8 = fig.add_subplot(111)
ax8.scatter(I_11, I_41, s = 10, c = 'orange')

ax8.plot([minEV, maxEV], [minEV, maxEV], c='k', zorder = -5)
ax8.set_xlim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])
ax8.set_ylim([minEV-(maxEV-minEV)*0.05,maxEV+(maxEV-minEV)*0.05])

w, p = wilcoxon(I_41-I_11, alternative='greater')
plt.title(file_name)
plt.text(minEV + 0.01*(maxEV-minEV), 0.9*maxEV, 'p = ' + str(np.round(p,4)))
plt.savefig('Info15/' + file_name + 'alltimes')


# In[ ]:


###################### Run new decoding #############
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
matplotlib.use('Agg')

rat_name = str(sys.argv[1].strip())
mod_name = str(sys.argv[2].strip())
sess_name = str(sys.argv[3].strip()) 
if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
file_name = rat_name + '_' + mod_name + '_' + sess_name
"""
sspikes, sspikes_box, xx, yy = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bBox = True)
file_name = rat_name + '_' + mod_name + '_' + sess_name

################### Hyperparameters ####################
dim = 6
ph_classes = [0,1] # Decode the ith most persistent cohomology class
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 1
coeff = 47
num_neurons = len(sspikes[0,:])
active_times = 15000
k = 1000
num_times = 5
n_points = 1200
nbs = 800

times_cube = np.arange(0,len(sspikes[:,0]),num_times)
movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube,:],1))[-active_times:])
movetimes = times_cube[movetimes]

dim_red_spikes_move_scaled,__,__ = pca(preprocessing.scale(sspikes[movetimes,:]), dim = dim)
indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                    n_points, 1, metric)
dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
X = squareform(pdist(dim_red_spikes_move_scaled, metric))
knn_indices, knn_dists, __ = nearest_neighbors(X, n_neighbors = nbs, metric = 'precomputed', angular=True, metric_kwds = {})
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

############ Compute persistent homology ################
rips_real = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=True, distance_matrix = True)
############ Decode cocycles ################
diagrams = rips_real["dgms"] # the multiset describing the lives of the persistence classes
cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
dists_land = rips_real["dperm2all"] # the pairwise distance between the points 
births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
deaths1[np.isinf(deaths1)] = 0
lives1 = deaths1-births1 # the lifetime for the 1-dim classes
iMax = np.argsort(lives1)
coords1 = np.zeros((num_circ, len(indstemp)))
threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]])*dec_tresh
for c in ph_classes:
    cocycle = cocycles[iMax[-(c+1)]]
    coords1[c,:],inds = get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

#    num_sampled = dists_land.shape[1]
#    num_tot = np.shape(indstemp)[0]
    # GET CIRCULAR COORDINATES
#    call[c,:] = predict_color(coords1[c,:], sspikes_box[:, :], sspikes[movetimes[indstemp],:], 
#                               dist_measure=metric, num_batch = 4098, k = 10)
np.savez_compressed('Results/Orig/' + file_name + '_pers_analysis' , coords = coords1, 
    movetimes = movetimes, indstemp = indstemp, dist = d, diagrams = diagrams)
"""
f = np.load('Results/Orig/' + file_name + '_pers_analysis.npz' , allow_pickle = True)

coords1 = f['coords'] 
movetimes = f['movetimes']
indstemp = f['indstemp']
f.close()

if sess_name[:6] == 'sws_c0':
    sspikes = load_spikes(rat_name, mod_name, 'sws_rec2', bSmooth = True, bBox = False, bSpeed = False)
else:
    sspikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bBox = False, bSpeed = False)


#sig = 15
#if sess_name[:6] == 'sws_c0':
#    sspikes = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = True, sig = sig)
#    spikes = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = False)
#else:
#    sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = sig)
#    spikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = False)
if sess_name[:3] in ('box', 'maz'):
    xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
    xx = xx[speed>2.5]
    yy = yy[speed>2.5]
    sspikes = sspikes[speed>2.5,:]
#    spikes = spikes[speed>2.5,:]

num_neurons = len(sspikes[0,:])
centcosall = np.zeros((num_neurons, 2, 1200))
centsinall = np.zeros((num_neurons, 2, 1200))
dspk = preprocessing.scale(sspikes[movetimes[indstemp],:])

k = 1200
for neurid in range(num_neurons):
    spktemp = dspk[:, neurid].copy()
#    spktemp = spktemp/np.sum(np.abs(spktemp))
    centcosall[neurid,:,:] = np.multiply(np.cos(coords1[:, :]*2*np.pi),spktemp)
    centsinall[neurid,:,:] = np.multiply(np.sin(coords1[:, :]*2*np.pi),spktemp)
filenames = glob.glob('Data/' + file_name + '_newdecoding.npz')
if len(filenames) == np.inf:
    f = np.load('Data/' + file_name + '_newdecoding.npz', allow_pickle = True)
    coordsnew = f['coordsnew']
    if sess_name[:3] == 'box':
        coordsbox = coordsnew.copy()
    else:           
        coordsbox = f['coordsbox']
#    times = f['times']
    f.close()

    np.savez_compressed('Data/' + file_name + '_newdecoding', 
        coordsnew = coordsnew, coordsbox = coordsbox, 
        #times = times, 
        centcosall = centcosall, centsinall = centsinall)
else:
    sig = 15
    if sess_name[:6] == 'sws_c0':
        sspikes = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = True, sig = sig)
        spikes = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = False)
    else:
        sspikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = True, sig = sig)
        spikes = load_spikes_dec(rat_name, mod_name, sess_name, bSmooth = False)
    if sess_name[:3] in ('box', 'maz'):
        xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
        sspikes = sspikes[speed>2.5,:]
        spikes = spikes[speed>2.5,:]    

    times = np.where(np.sum(spikes>0, 1)>=1)[0]
    dspk = preprocessing.scale(sspikes)
    sspikes = sspikes[times,:]
    dspk = dspk[times,:]

    a = np.zeros((len(sspikes[:,0]), 2, num_neurons))
    for n in range(num_neurons):
        a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))

    c = np.zeros((len(sspikes[:,0]), 2, num_neurons))
    for n in range(num_neurons):
        c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))


    mtot2 = np.sum(c,2)
    mtot1 = np.sum(a,2)
    coordsnew = np.arctan2(mtot2,mtot1)%(2*np.pi)
    if sess_name[:3] == 'box':
        coordsbox = coordsnew.copy()
        times_box = times.copy()
#        xx = xx[times]
#        yy = yy[times]
    else:
        boxname = 'box'
        if (rat_name == 'roger') and (sess_name[-5:] in ('_rec2')):
            boxname += '_rec2'
        sspikes = load_spikes_dec(rat_name, mod_name,boxname, bSmooth = True, sig = sig)
        spikes = load_spikes_dec(rat_name, mod_name, boxname, bSmooth = False)

        xx,yy, speed = load_pos(rat_name, boxname, bSpeed = True)
        sspikes = sspikes[speed>2.5,:]
        spikes = spikes[speed>2.5,:]
#        xx = xx[speed>2.5]
#        yy = yy[speed>2.5]
        dspk =preprocessing.scale(sspikes)  
        times_box = np.where(np.sum(spikes>0, 1)>=1)[0]
        dspk = dspk[times_box,:]
#        xx = xx[times]
#        yy = yy[times]

        a = np.zeros((len(times_box), 2, num_neurons))
        for n in range(num_neurons):
            a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))

        c = np.zeros((len(times_box), 2, num_neurons))
        for n in range(num_neurons):
            c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))

        mtot2 = np.sum(c,2)
        mtot1 = np.sum(a,2)
        coordsbox = np.arctan2(mtot2,mtot1)%(2*np.pi)

    np.savez_compressed('Data/' + file_name + '_newdecoding', 
        coordsnew = coordsnew, coordsbox = coordsbox, 
        times = times, times_box = times_box, centcosall = centcosall, centsinall = centsinall)

#    m1b_1, m2b_1, xedge,yedge = get_ang_hist(coordsbox[:,0], 
#        coordsbox[:,1], xx,yy)

#    fig, ax = plt.subplots(1,1)
#    ax.imshow(np.cos(m1b_1).T, origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
#    ax.set_aspect('equal', 'box')
#    ax.set_xticks([], [])
#    ax.set_yticks([], [])
#    fig.savefig('Figs/newdec/' + file_name + '_newdecoding1box_tmp.png', bbox_inches='tight', pad_inches=0.02)
#    fig.savefig('Figs/newdec/' + file_name + '_newdecoding1box_tmp.pdf', bbox_inches='tight', pad_inches=0.02)

#    fig, ax = plt.subplots(1,1)
#    ax.imshow(np.cos(m2b_1).T, origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
#    ax.set_aspect('equal', 'box')
#    ax.set_xticks([], [])
#    ax.set_yticks([], [])
#    fig.savefig('Figs/newdec/' + file_name + '_newdecoding2box_tmp.png', bbox_inches='tight', pad_inches=0.02)
#    fig.savefig('Figs/newdec/' + file_name + '_newdecoding2box_tmp.pdf', bbox_inches='tight', pad_inches=0.02)


# In[ ]:


########### Run persistence H2 ################
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
matplotlib.use('Agg')

rat_name = str(sys.argv[1].strip())
mod_name = str(sys.argv[2].strip())
sess_name = str(sys.argv[3].strip()) 
if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
sspikes, sspikes_box, xx, yy = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bBox = True)
file_name = rat_name + '_' + mod_name + '_' + sess_name

################### Hyperparameters ####################
dim = 6
ph_classes = [0,1] # Decode the ith most persistent cohomology class
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 1
coeff = 47
num_neurons = len(sspikes[0,:])
active_times = 15000
k = 1000
num_times = 5
n_points = 1200
nbs = 800

times_cube = np.arange(0,len(sspikes[:,0]),num_times)
movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube,:],1))[-active_times:])
movetimes = times_cube[movetimes]

dim_red_spikes_move_scaled,__,__ = pca(preprocessing.scale(sspikes[movetimes,:]), dim = dim)
indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                    n_points, 1, metric)
dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
X = squareform(pdist(dim_red_spikes_move_scaled, metric))
knn_indices, knn_dists, __ = nearest_neighbors(X, n_neighbors = nbs, metric = 'precomputed', angular=True, metric_kwds = {})
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

############ Compute persistent homology ################
rips_real = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=True, distance_matrix = True)
############ Decode cocycles ################
diagrams = rips_real["dgms"] # the multiset describing the lives of the persistence classes
cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
dists_land = rips_real["dperm2all"] # the pairwise distance between the points 
births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
deaths1[np.isinf(deaths1)] = 0
lives1 = deaths1-births1 # the lifetime for the 1-dim classes
iMax = np.argsort(lives1)
coords1 = np.zeros((num_circ, len(indstemp)))
threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]])*dec_tresh
for c in ph_classes:
    cocycle = cocycles[iMax[-(c+1)]]
    coords1[c,:],inds = get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

#    num_sampled = dists_land.shape[1]
#    num_tot = np.shape(indstemp)[0]
    # GET CIRCULAR COORDINATES
#    call[c,:] = predict_color(coords1[c,:], sspikes_box[:, :], sspikes[movetimes[indstemp],:], 
#                               dist_measure=metric, num_batch = 4098, k = 10)
np.savez_compressed('Results/Orig/' + file_name + '_pers_analysis' , coords = coords1, 
    movetimes = movetimes, indstemp = indstemp, dist = d, diagrams = diagrams)

########### Run persistence H2 ################


import numpy as np 
from ripser import ripser
import sys

rat_name = sys.argv[1].strip()
mod_name = sys.argv[2].strip()
sess_name = sys.argv[3].strip()

if (rat_name == 'roger') & ((sess_name[:3]=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'

file_name = rat_name + '_' + mod_name + '_' + sess_name

f = np.load('Results/Orig/' + file_name + '_pers_analysis.npz')
d = f['dist']
f.close()

maxdim = 2
coeff = 47
### Compute persistent homology
rips_real = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=False, distance_matrix = True)
np.savez_compressed('Results/Orig/' + file_name + '_H2', diagrams = rips_real["dgms"])


# In[ ]:


########### Run persistence roll ################

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
matplotlib.use('Agg')

rat_name = str(sys.argv[1].strip())
mod_name = str(sys.argv[2].strip())
sess_name = str(sys.argv[3].strip()) 
if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
sspikes, sspikes_box, xx, yy = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bBox = True)
file_name = rat_name + '_' + mod_name + '_' + sess_name


################## Roll ####################
num_neurons = len(sspikes[0,:])
sleeptime = np.random.rand()
time.sleep(sleeptime)
s = np.random.randint(99999999)
np.random.seed(s)
shift = np.zeros(num_neurons, dtype = int)
for n in range(num_neurons):
    shifti = int(np.random.rand()*len(sspikes[:,0]))
    sspikes[:,n] = np.roll(sspikes[:,n].copy(), shifti)
    shift[n] = shifti

################### Hyperparameters ####################
dim = 6
ph_classes = [0,1] # Decode the ith most persistent cohomology class
num_circ = len(ph_classes)
dec_tresh = 0.99
metric = 'cosine'
maxdim = 2
coeff = 47
active_times = 15000
k = 1000
num_times = 5
n_points = 1200
nbs = 800

times_cube = np.arange(0,len(sspikes[:,0]),num_times)
movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube,:],1))[-active_times:])
movetimes = times_cube[movetimes]

dim_red_spikes_move_scaled,__,__ = pca(preprocessing.scale(sspikes[movetimes,:]), dim = dim)
indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                    n_points, 1, metric)
dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
X = squareform(pdist(dim_red_spikes_move_scaled, metric))
knn_indices, knn_dists, __ = nearest_neighbors(X, n_neighbors = nbs, metric = 'precomputed', angular=True, metric_kwds = {})
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

############ Compute persistent homology ################
rips_real = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=False, distance_matrix = True, shift = shift)
rollname = file_name + '_H2_roll_' + datetime.now().strftime('%Y%m%d%H%M%S') + '_' + str(s)
np.savez_compressed('Results/Roll/' + rollname + '.npz', diagrams = rips_real["dgms"])


# In[ ]:


########### source code barcodes ###############

import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import sys
import matplotlib.gridspec as grd
import pandas as pd

with pd.ExcelWriter('barcodes.xlsx',
                    mode='w') as writer:  
    for (rat_name, mod_name, sess_name) in (('roger', 'mod1', 'box'),
                                         ('roger', 'mod3', 'box'),
                                         ('roger', 'mod4', 'box'),
                                         ('roger', 'mod1', 'maze'),
                                         ('roger', 'mod3', 'maze'),
                                         ('roger', 'mod4', 'maze'),
                                         ('roger', 'mod1', 'box_rec2'),
                                         ('roger', 'mod2', 'box_rec2'),
                                         ('roger', 'mod3', 'box_rec2'),
                                         ('roger', 'mod1', 'rem_rec2'),
                                         ('roger', 'mod2', 'rem_rec2'),
                                         ('roger', 'mod3', 'rem_rec2'),
                                         ('roger', 'mod1', 'sws_rec2'),
                                          ('roger', 'mod1', 'sws_class1'),
                                          ('roger', 'mod1', 'sws_class2'),
                                         ('roger', 'mod2', 'sws_rec2'),
                                         ('roger', 'mod3', 'sws_rec2'), 
                                         ('quentin', 'mod1', 'box'),
                                         ('quentin', 'mod2', 'box'),
                                         ('quentin', 'mod1', 'maze'),
                                         ('quentin', 'mod2', 'maze'),
                                         ('quentin', 'mod1', 'rem'),
                                         ('quentin', 'mod2', 'rem'),
                                         ('quentin', 'mod1', 'sws'),
                                         ('quentin', 'mod2', 'sws'),
                                         ('shane', 'mod1', 'box'),
                                         ('shane', 'mod1', 'maze'),
                                         ('shane', 'mod1', 'rem'),
                                         ('shane', 'mod1', 'sws'),
                                         ('shane', 'mod1', 'sws_class2')):
    data = []
    data_names = []
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

    indsall =  0
    labels = ["$H_0$", "$H_1$", "$H_2$"]
    for dit, dim in enumerate(dims):
        d = np.copy(persistence[dim])
        d[np.isinf(d[:,1]),1] = infinity
        dlife = (d[:,1] - d[:,0])
        dinds = np.argsort(dlife)[-30:]#[int(0.5*len(dlife)):]
        dl1,dl2 = dlife[dinds[-2:]]
        if dim>0:
            dinds = dinds[np.flip(np.argsort(d[dinds,0]))]


        data.append(pd.Series(dlife[dinds]))
        data_names.extend(['Dim_' + str(dim) + '_lifetime'])
        data.append(pd.Series(d[dinds,0]))
        data_names.extend(['Dim_' + str(dim) + '_births'])
        indsall = len(dinds)
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
        ytemp = x1 + np.max(lives1_all)
        data.append(pd.Series(ytemp))
        data_names.extend(['Dim_' + str(dim) + '_shuffled_line'])

        if sess_name == 'maze':
            sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_WW'
        elif sess_name == 'box':
            sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_OF'
        elif sess_name == 'box_rec2':
            sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_OF2'
        elif sess_name == 'sws_class1':
            sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_SWS_Bursty'
        elif sess_name == 'sws_class2':
            sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_SWS_Theta'
        else:
            sheet_name = np.str.upper(rat_name[0]) + mod_name[3] + '_' + np.str.upper(sess_name[:3])
    print(sheet_name)
    df = pd.concat(data, ignore_index=True, axis=1)           
    df.columns = data_names
    df.to_excel(writer, sheet_name=sheet_name)


# In[ ]:


########### subpopulations

import scipy.io as sio
import numpy as np
from utils import *


def get_inds(rat_name, mod_name, sess_name):

    if rat_name == 'shane':
        tot_path_all = 'shane/rec1/shane_all.mat'
        tot_path_mod = 'shane/rec1/shane_mod_final.mat'
        if mod_name == 'mod1':
            v = 'v3'
    elif rat_name == 'quentin':
        tot_path_all = 'quentin/rec1/quentin_all.mat'
        tot_path_mod = 'quentin/rec1/quentin_mod_final.mat'
        if mod_name == 'mod1':
            v = 'v3'
        elif mod_name == 'mod2':
            v = 'v4'        
    elif (rat_name == 'roger') & (sess_name in ('box_rec2', 'sws_rec2', 'rem_rec2')):
        tot_path_all = 'roger/rec2/roger_all2.mat'
        tot_path_mod = 'roger/rec2/roger_rec2_mod_final.mat'
        if mod_name == 'mod1':
            v = 'v3'
        elif mod_name == 'mod2':
            v = 'v4'
        elif mod_name == 'mod3':
            v = 'v5'
    elif (rat_name == 'roger') & (sess_name in ('box', 'maze')):
        tot_path_all = 'roger/rec1/roger_all1.mat'
        tot_path_mod = 'roger/rec1/roger_mod_final.mat'
        if mod_name == 'mod1':
            v = 'v3'
        elif mod_name == 'mod2':
            v = 'v4'
        elif mod_name == 'mod3':
            v = 'v5'
        elif mod_name == 'mod4':
            v = 'v6' 
    marvin = sio.loadmat(tot_path_all)
    mall = np.zeros(len(marvin['vv'][0,:]), dtype=int)
    mall1 = np.zeros(len(marvin['vv'][0,:]), dtype=int)
    for i,m in enumerate(marvin['vv'][0,:]):
        mall[i] = int(m[0][0])
        mall1[i] = int(m[0][2:])

    marvin = sio.loadmat(tot_path_mod)
    m2 = np.zeros(len(marvin[v][:,0]), dtype=int)
    m22 = np.zeros(len(marvin[v][:,0]), dtype=int)
    for i,m in enumerate(marvin[v][:,0]):
        m2[i] = int(m[0][0])
        m22[i] = int(m[0][2:])            

    inds = np.zeros(len(m2), dtype = int)
    for i in range(len(m2)):
        inds[i] = np.where((mall==m2[i]) & (mall1==m22[i]))[0]

    if (sess_name[-4:] == 'rec2'):
        rec_name = 'rec2'
    else:
        rec_name = 'rec1' 

    tot_path = rat_name + '/' + rec_name + '/' + 'data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')
    num_neurons = len(inds)
    mvl = np.zeros((num_neurons))
    for j,i in enumerate(inds):
        if len(marvin[marvin["clusters"]["tc"][0][i]])==4:
            mvl[j] = marvin[marvin["clusters"]["tc"][0][i]]['hd']['mvl'][()]
    inds = inds[mvl<0.3]
    m2 = mall[inds]
    m22 = mall1[inds]
    if (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name in ('box', 'maze')):
        inds1, m21,m221 = get_inds(rat_name, 'mod2', sess_name)
        m2 =  np.concatenate((m2, m21))
        m22 =  np.concatenate((m22, m221))
        inds = np.concatenate((inds, inds1))
    return inds, m2, m22 

def get_theta_tuning(rat_name, mod_name, sess_name):
    if (sess_name[-4:] == 'rec2'):
        rec_name = 'rec2'
    else:
        rec_name = 'rec1'
    inds = get_inds(rat_name, mod_name, sess_name)[0]
    tot_path = rat_name + '/' + rec_name + '/' + 'data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')

    num_neurons = len(inds)
    #zs_shane_rec1 = np.zeros((num_neurons, 60))
    mu = np.zeros((num_neurons))
    mvl = np.zeros((num_neurons))
    for j,i in enumerate(inds):
        if len(marvin[marvin["clusters"]["tc"][0][i]])==4:
            #zs[j,:] = marvin[marvin["clusters"]["tc"][0][i]]['thetaPhase']['z'][()]
            mu[j] = marvin[marvin["clusters"]["tc"][0][i]]['thetaPhase']['mu'][()]
            mvl[j] = marvin[marvin["clusters"]["tc"][0][i]]['thetaPhase']['mvl'][()]
    return mu, mvl

def get_start_end(rat_name, sess_name):
    if rat_name == 'roger':
        if sess_name == 'box':
            start = 7457
            end = 16045
        elif sess_name == 'maze':
            start = 16925
            end = 20704
        elif sess_name == 'maze1':    
            start = 20895
            end = 21640
        elif sess_name == 'box_rec2':
            start = 10617
            end = 13004
        elif sess_name[:4] in ('sws_', 'rem_'):
            start = 396
            end = 9941
        elif sess_name[:4] in ('sws1', 'rem1'):
            start = 13143
            end = 15973
    elif rat_name == 'quentin':
        if sess_name[:3] == 'box':
            start = 27826
            end = 31223
        elif sess_name[:4] == 'maze':
            start = 18977
            end = 25355
        elif sess_name[:3] in ('sws', 'rem'):
            start = 9576
            end = 18812
    elif rat_name == 'shane':
        if sess_name == 'box':
            start = 9939
            end = 12363
        elif sess_name == 'maze':
            start = 13670
            end = 14847
        elif sess_name in ('sws', 'rem'):
            start = 14942
            end = 23133
        elif sess_name in ('sws', 'rem'):
            start = 25403
            end = 27007
        elif sess_name == 'maze1':
            start = 23186
            end = 24936
    return start, end


"""def get_isi(rat_name, mod_name, sess_name):
    def bin_isi(isi, numbins = 100, maxt = 0.1):
        bins = np.linspace(0, maxt, numbins)
        num_neurons = len(isi) 
        isi_bincount = np.zeros((num_neurons, numbins + 1)) 
        for i in range(num_neurons):
            isi_binned = np.digitize(isi[i], bins)
            isi_bincount[i,:] = np.bincount(isi_binned, minlength = numbins+1)
        return isi_bincount    

    inds = get_inds(rat_name, mod_name, sess_name)
    start, end = get_start_end(rat_name, sess_name)
    if (sess_name[-4:] == 'rec2'):
        rec_name = 'rec2'
    else:
        rec_name = 'rec1'

    tot_path = rat_name + '/' + rec_name + '/' + 'data_bendunn.mat'
    marvin = h5py.File(tot_path, 'r')

    isi = []
    if sess_name[:3] in ('sws', 'rem'):
        for i,m in enumerate(inds):
            isi.append([])
            s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
            for r in range(sum(marvin['sleepTimes'][sess_name[:3]][()][0,:]<=end)):
                sleep_start1 = marvin['sleepTimes'][sess_name[:3]][()][0, r]
                sleep_end1 = marvin['sleepTimes'][sess_name[:3]][()][1, r]
                spktmp = s[(s>= sleep_start1) & (s< sleep_end1)]
                isi[i].extend(spktmp[1:]-spktmp[:-1]) 
    elif (rat_name == 'roger') & (sess_name in ('maze', 'box')):
        valid_times = {}
        valid_times['maze'] = [[16925,18026], [18183, 20704]]
        valid_times['box'] = [[7457,14778], [14890, 16045]]
        for i,m in enumerate(inds):
            isi.append([])
            s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
            for r in range(len(valid_times)):
                valid_start = valid_times[sess_name][r][0]
                valid_end = valid_times[sess_name][r][1]
                spktmp = s[(s>= valid_start) & (s< valid_end)]
                isi[i].extend(spktmp[1:]-spktmp[:-1])         
    else:
        for i,m in enumerate(inds):
            isi.append([])
            s = marvin[marvin['clusters']['spikeTimes'][0,:][m]][()][0, :]
            spktmp = s[(s>=start) & (s<end)]
            isi[i].extend(spktmp[1:]-spktmp[:-1])        
    isi_bincount = bin_isi(isi)
    return isi_bincount
"""
times_all = {}
times_all['quentin_box'] = ((27826, 31223),)
times_all['quentin_sleep'] = ((9576, 18812),)
times_all['quentin_maze'] = ((18977, 25355),)
times_all['shane_box'] = ((9939, 12363),)
times_all['shane_maze'] = ((13670, 14847),)
times_all['shane_maze2'] = ((23186, 24936),)
times_all['roger_box_rec2'] = ((10617, 13004),)
times_all['roger_sleep_rec2'] = ((396, 9941),)
times_all['roger_box'] = ((7457,14778), 
                           (14890, 16045))
times_all['roger_maze'] = ((16925, 18026),
                            (18183, 20704))
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

def get_isi_acorr(rat_name, mod_name, sess_name, maxt = 0.1):
    def bin_isi(isi, numbins = 100, maxt = 0.1):
        bins = np.linspace(0, maxt, numbins)
        num_neurons = len(isi) 
        isi_bincount = np.zeros((num_neurons, numbins + 1)) 
        for i in range(num_neurons):
            isi_binned = np.digitize(isi[i], bins)
            isi_bincount[i,:] = np.bincount(isi_binned, minlength = numbins+1)
        return isi_bincount    

    f = np.load('Data/' + rat_name + '_' + mod_name + '_' + sess_name + '_spk_times.npz', allow_pickle=True)
    spikes_mod21 = f['spiketimes'][()]
    f.close()
    times1 = times_all[rat_name + '_' + sess_name]
    num_neurons = len(spikes_mod21)
    spk = np.zeros((1,num_neurons))
    bin_times = np.linspace(-maxt-1e-5,maxt+1e-5, 101)
    acorr = np.zeros((num_neurons, len(bin_times)), dtype = int)
    isi = []
    for i in range(num_neurons):
        isi.append([])

    for start1, end1 in times1: 
        min_time = start1
        max_time = end1
        for i, n in enumerate(spikes_mod21):
            spike_times = np.array(spikes_mod21[n]-min_time)
            spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
#            print(spike_times)
            isi[i].extend(spike_times[1:]-spike_times[:-1]) 
            for ss in spike_times:
                stemp = spike_times[(spike_times<ss+maxt) & (spike_times>ss-maxt)]
                dd = stemp-ss
                acorr[i,:] += np.bincount(np.digitize(dd, bin_times)-1, minlength=101)
#    print(np.argmax(acorr,1))                        
    #acorr_distr = acorr[:, 51:76] + acorr[:, 25:50]
    acorr_distr = acorr[:, 51:] + np.flip(acorr[:, :50])
    isi_bincount = bin_isi(isi)
    isi = np.array(isi)
    isimean = np.zeros(len(isi))
    for i in range(len(isi)):
        isitmp = np.array(isi[i])
        isitmp = isitmp[isitmp<0.1]
        isimean[i] = np.mean(isitmp)*1000 
#    print(isi_bincount.shape)
    #return isi_bincount, acorr_distr, bin_times[51:76] 
    return isi_bincount, acorr_distr, bin_times[51:], isimean 

def get_burst_stats(isi, acorr,bin_times, burst_thresh = 20):
    num_neurons, num_bins = np.shape(isi)
    isi_mc = np.zeros(num_neurons)
    isi_rat = np.zeros(num_neurons)
    bins_torus = np.linspace(0,num_bins-1, num_bins)
    bins1 = bins_torus[:-1] + (bins_torus[1] - bins_torus[0])/2
    for i in range(num_neurons):    
        isi_mc[i] = np.dot(bins1,isi[i,:-1]/np.sum(isi[i,:-1]))
        isi_rat[i] = np.sum(isi[i,:burst_thresh])/np.sum(isi[i,:])
    acorr_mc = np.zeros(num_neurons)
    acorr_rat = np.zeros(num_neurons)
    thresh = 0.02
#    print(bin_times)
#    print(bin_times, acorr)
    for i in range(num_neurons):    
#        plt.figure()
#        plt.plot(acorr[i,bin_times <= thresh]/ sum(acorr[i,:])*100)
#        plt.savefig('acorrtmp'+ str(i))
#        plt.figure()
#        plt.plot(acorr[i,:]/ sum(acorr[i,:])*100)
#        plt.savefig('acorrtmp'+ str(i) + 'all')

#        acorr_mc[i] = np.dot(np.arange(1,26),acorr[i,:]/np.sum(acorr[i,:]))
        #acorr_mc[i] = np.dot(np.arange(1,51),acorr[i,:50]/np.sum(acorr[i,:50]))
        acorr_mc[i] = np.dot(np.arange(1,51),acorr[i,:]/np.sum(acorr[i,:]))
        acorr_rat[i] = sum(acorr[i,bin_times <= thresh]) / sum(acorr[i,:]) * 100;
    return isi_mc, isi_rat, acorr_mc, acorr_rat



def get_distr2(rat_name, mod_name, sess_name, var1, var2, var1_thresh, var2_thresh):
    labels = np.zeros(len(var1), dtype = int)
    labels[((var1<var1_thresh) & ((var2>=var2_thresh) | (var2<=-var2_thresh)))] = 1
    labels[((var1>var1_thresh) & ((var2<var2_thresh) | (var2>-var2_thresh)))] = 2
    distr = np.zeros(3)
    for i in [0,1,2]:
        distr[i] = sum(labels==i)/len(labels)
    return distr, labels

def get_distr1(rat_name, mod_name, sess_name, var1, var1_thresh):
    labels = np.zeros(len(var1), dtype = int)
    labels[(var1<var1_thresh)] = 1
    distr = np.zeros(2)
    for i in [0,1]:
        distr[i] = sum(labels==i)/len(labels)
    return distr, labels

def get_distr3(rat_name, mod_name, sess_name, var1, var2, var1_thresh, var2_thresh, var3, var3_thresh):
    labels = np.zeros(len(var1), dtype = int)
    labels[((var1<var1_thresh) & ((var2>=var2_thresh) | (var2<=-var2_thresh)))] = 1
    labels[((var1>var1_thresh) & ((var2<var2_thresh) | (var2>-var2_thresh)))] = 2
    distr = np.zeros(3)
    for i in [0,1,2]:
        distr[i] = sum(labels==i)/len(labels)
    return distr, labels


def plot_distr(distrs, stats):
    i = 0
    plt.figure()
#    cs = list(colors_mods.keys())
#    print(cs)
    for j in range(np.shape(distrs)[0]):
        distr = distrs[j,:]
#        print(distr)
#        plt.bar(np.arange(i,i+len(distr)), distr, color = colors_mods[cs[j]])
        plt.bar(np.arange(i,i+len(distr)), distr,width = 0.8,  
            edgecolor = [[0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.]],  lw = 0.5,
            color = ['#1f77b4', '#ff7f0e', '#2ca02c', [0.4,0.4,0.4]])
        i += len(distr)+1

  
    plt.ylim([0,0.9])
    plt.xlim([-1, i-1])
    plt.xticks([],[])
    plt.yticks([],[])
    #plt.title(stats)
    plt.gca().axes.spines['top'].set_visible(False)
    plt.gca().axes.spines['right'].set_visible(False)
    plt.savefig('Figs/' + stats + '_distr_grey1.png' , bbox_inches='tight', pad_inches=0.02)
    plt.savefig('Figs/' + stats + '_distr_grey1.pdf', bbox_inches='tight', pad_inches=0.02)

def plot_stats(rat_name, mod_name, sess_name, xs, ys, labels, stats, zs = []):
    plt.figure()
    ax = plt.axes()
    cs = ['#1f77b4', '#ff7f0e', '#2ca02c', [1,1,1]]

    mvl_thresh = 0.5
    mu_thresh = [-2*np.pi,-3*np.pi/2, -np.pi/2, np.pi/2, 3/2*np.pi,2*np.pi]
    isi_thresh = 39
    if stats[:2] == 'mu':
        sz = 30
        sz1 = 0.2 
        sz2 = 0.5
        sz3 = 1.5
    else:
        sz = 60 
        sz1 = 0.4
        sz2 = 0.8
        sz3 = 2
    if stats[:4] == 'mumv':
        ax.set_xlim([-2*np.pi,2*np.pi])
        ax.set_ylim([0,1])
        for m in range(len(mu_thresh[1:-1])):
            mus = mu_thresh[m+1]
            mus_prev = mu_thresh[m]
            ax.plot([mus,mus],[0,1],ls = '--', c='k', alpha = 0.5, lw = sz3)
            if m%2== 0: 
                ax.fill_between([mus_prev,mus], [mvl_thresh, mvl_thresh], [1,1], color = cs[3], alpha =0.05)
                ax.fill_between([mus_prev,mus], [0,0], [mvl_thresh, mvl_thresh], color = cs[1], alpha=0.05)
            else:
                ax.fill_between([mus_prev,mus], [mvl_thresh, mvl_thresh], [1,1], color = cs[2], alpha=0.05)
                ax.fill_between([mus_prev,mus], [0,0], [mvl_thresh, mvl_thresh], color = cs[0], alpha=0.05)

        ax.fill_between([mus, mu_thresh[-1]], [mvl_thresh, mvl_thresh], [1,1], color = cs[3], alpha=0.05)
        ax.fill_between([mus, mu_thresh[-1]], [0,0], [mvl_thresh, mvl_thresh], color = cs[1], alpha=0.05)

        ax.plot([-2*np.pi,2*np.pi], [mvl_thresh,mvl_thresh],ls = '--', c='k', alpha = 0.5, lw = sz3)
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        xxs = [-2*np.pi, -3/2*np.pi, -np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3/2*np.pi, 2*np.pi]
        ax.set_xticks(xxs)
        ax.set_xticklabels(np.zeros(len(xxs),dtype=str))
        ax.xaxis.set_tick_params(width=1, length =5)
        yys = [0, 0.25, 0.5, 0.75, 1]
        ax.set_yticks(yys)
        ax.set_yticklabels(np.zeros(len(yys),dtype=str))
        ax.yaxis.set_tick_params(width=1, length =5)

        ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*2))
    elif stats[:4] == 'mumc':
#        ax.set_xlim([-np.pi,np.pi])
        ax.set_xlim([-2*np.pi,2*np.pi])
        ax.set_ylim([10,60])
        for m in range(len(mu_thresh[1:-1])):
            mus = mu_thresh[m+1]
            mus_prev = mu_thresh[m]
            mus_next = mu_thresh[m+2]
            ax.plot([mus,mus],[10,60],ls = '--', c='k', alpha = 0.5, lw = sz2)
            ax.fill_between([mus_prev,mus], [isi_thresh, isi_thresh], [1,1], color = cs[3], alpha=0.05)
            ax.fill_between([mus,mus_next], [isi_thresh, isi_thresh], [1,1], color = cs[2], alpha=0.05)
            ax.fill_between([mus_prev,mus], [0,0], [isi_thresh, isi_thresh], color = cs[1], alpha=0.05)
            ax.fill_between([mus,mus_next], [0,0], [isi_thresh, isi_thresh], color = cs[0], alpha=0.05)
#        ax.plot([-mu_thresh,-mu_thresh],[10,60],ls = '--', c='k')
#        ax.plot([-np.pi,np.pi], [isi_thresh,isi_thresh],ls = '--', c='k')
        ax.plot([-2*np.pi,2*np.pi], [isi_thresh,isi_thresh],ls = '--', c='k', alpha = 0.5, lw = sz2)
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*2))

    elif stats[:5] == 'mvlmc':
        ax.set_ylim([0,1])
        ax.set_xlim([10,60])
        ax.plot([10,60],[mvl_thresh,mvl_thresh],ls = '--', c='k', alpha = 0.5, lw = sz3)
        ax.plot([isi_thresh, isi_thresh],[0,1],ls = '--', c='k', alpha = 0.5, lw = sz3)
        ax.fill_between([10,isi_thresh], [mvl_thresh, mvl_thresh], [1,1], color = cs[2], alpha=0.05)
        ax.fill_between([isi_thresh,60], [mvl_thresh, mvl_thresh], [1,1], color = cs[3], alpha=0.05)
        ax.fill_between([10,isi_thresh], [0,0], [mvl_thresh, mvl_thresh], color = cs[0], alpha=0.05)
        ax.fill_between([isi_thresh,60], [0,0], [mvl_thresh, mvl_thresh], color = cs[1], alpha=0.05)
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        xxs = [10, 22.5, 35, 47.5, 60]
        ax.set_xticks(xxs)
        ax.set_xticklabels(np.zeros(len(xxs),dtype=str))
        ax.xaxis.set_tick_params(width=1, length =7.5)
        
        yys = [0, 0.25, 0.5, 0.75, 1]
        ax.set_yticks(yys)
        ax.set_yticklabels(np.zeros(len(yys),dtype=str))
        ax.yaxis.set_tick_params(width=1, length =7.5)


        ax.set_aspect(abs(x1-x0)/(abs(y1-y0)))
    if len(zs)==0:
        for i in np.flip(np.unique(labels)):
            if i == 3:
                ax.scatter(xs[labels==i],ys[labels==i], marker = 'x', c = [0.4,0.4,0.4], s =  sz)
            else:
                ax.scatter(xs[labels==i],ys[labels==i], edgecolor = [0.2,0.2,0.2], lw = sz1, c = cs[i], s =  sz)
    else:
        for i in np.flip(np.unique(labels)):
            if i == 3:
                ax.scatter(xs[labels==i],ys[labels==i], marker = 'x', c = [0.4,0.4,0.4], s =  sz)
            else:
                ax.scatter(xs[labels==i],ys[labels==i], edgecolor = [0.2,0.2,0.2], lw = sz1, c = zs[labels == i], s = sz)
    if stats[:2] == 'mu':
        xs += 2*np.pi
        for i in np.flip(np.unique(labels)):
            if i == 3:
                ax.scatter(xs[labels==i],ys[labels==i], marker = 'x', c = [0.4,0.4,0.4], s =  sz)
            else:
                ax.scatter(xs[labels==i],ys[labels==i], edgecolor = [0.2,0.2,0.2], lw = sz1, c = cs[i], s = sz)
    [ax.spines[i].set_linewidth(sz2) for i in ax.spines]

    #plot_stats
#    ax.set_xticks([],[])
#    ax.set_yticks([],[])

#    plt.title(stats)
    plt.savefig('Figs/theta' + rat_name + '_' + mod_name + '_' + sess_name + '_' + stats + 'box_ind.png', bbox_inches='tight', pad_inches=0.02)
    plt.savefig('Figs/theta' + rat_name + '_' + mod_name + '_' + sess_name + '_' + stats + 'box_ind.pdf', bbox_inches='tight', pad_inches=0.02)


def clustering(rat_name, mod_name, sess_name, bSmooth, dist_thresh = 11, aff = 'l1', link = 'complete'):
    spk = load_spikes(rat_name, mod_name, sess_name, bSmooth = bSmooth, bBox = False)
    cov1 = np.corrcoef(spk.T)
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters = None,linkage = link,
                                         affinity = aff,
                                         distance_threshold = dist_thresh).fit(cov1)
    ind = clustering.labels_
   

    plt.viridis()
    indssort = np.argsort(ind)
    cov2 = cov1[indssort,:]
    cov2 = cov2[:,indssort]
#    print(np.median(cov2[np.triu_indices(cov2.shape[0],1)]),cov2.min())
#    print(3*np.mean(np.abs(cov2[np.triu_indices(cov2.shape[0],1)])), np.min(np.abs(cov2)))
    plt.figure()
    plt.imshow(np.abs(cov2),vmax = 0.15, vmin = 0)

    bind = np.bincount(ind).astype(str)
    btmp = ''
    for i in bind:
        btmp += i + ','
#    plt.title(btmp)
    plt.xticks([],[])
    plt.yticks([],[])
    
    plt.savefig('Figs/' + rat_name + '_' + mod_name + '_' + sess_name + '_corrmat_015_both.png', bbox_inches='tight', pad_inches=0.02)
    plt.savefig('Figs/' + rat_name + '_' + mod_name + '_' + sess_name + '_corrmat_015_both.pdf', bbox_inches='tight', pad_inches=0.02)
    return ind
def plot_toroidal_stats(rat_name, mod_name, sess_name, labels):
    def compute_entropy_infoscore(rat_name, mod_name, sess_name):  
        numangsint = 51
        numangsint_1 = numangsint-1
        bins_torus = np.linspace(0,2*np.pi, numangsint)
        sig = 2
        if sess_name == 'sws_c0_rec2':
            spikes = load_spikes(rat_name, mod_name, 'sws_rec2', bSmooth = False, bBox = False) 
        else:
            spikes = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bBox = False) 
        f = np.load('Results/Orig/' + rat_name + '_' + mod_name + '_' + sess_name + '_decoding.npz',  allow_pickle = True)
        c11all_1 = f['c11all']
        c12all_1 = f['c12all']    
        f.close()    
        num_neurons = np.shape(spikes)[1]
#        print(num_neurons)
        from scipy.stats import entropy
        ent1 = np.zeros(num_neurons)
        Itor = np.zeros(num_neurons)
        nnans2 = ~np.isnan(c11all_1)
        for neurid in range(num_neurons):
            mtot_all, x_edge, y_edge, circ = binned_statistic_2d(c11all_1[nnans2],c12all_1[nnans2], 
                spikes[nnans2,neurid], statistic='mean', bins=bins_torus, range=None, expand_binnumbers=True)
            mtot_all[np.isnan(mtot_all)] = 0
#            ent1[neurid] = entropy(mtot_all.flatten())#smooth_tuning_map(mtot_all,numangsint, 2).flatten())
            Itor[neurid]= information_score(mtot_all.copy(), circ, spikes[nnans2,neurid].mean())
        return ent1, Itor

    ent1, Itor1 = compute_entropy_infoscore(rat_name, mod_name, 'box_rec2')
    tit = 'e/I: '
    plt.figure()
    for i in np.unique(labels):
#        plt.scatter(np.arange(sum(labels<i),
#                              sum(labels<i)+sum(labels==i)),
        plt.scatter(Itor1[labels==i],
                    ent1[labels==i])
        tit += str(np.round(np.mean(ent1[labels==i]),3)) + '/'
        tit += str(np.round(np.mean(Itor1[labels==i]),3)) + ', '
#    plt.title(tit)
#    plt.legend(np.unique(labels).astype(str))    
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig('Figs/' + rat_name + '_' + mod_name + '_' + sess_name + '_class_entropy_infoscore_box', bbox_inches='tight', pad_inches=0.02)

    ent2, Itor2 = compute_entropy_infoscore(rat_name, mod_name, 'sws_c0_rec2')
    tit = 'e/I: '
    plt.figure()
    for i in np.unique(labels):
#        plt.scatter(np.arange(sum(labels<i),
#                              sum(labels<i)+sum(labels==i)),
        plt.scatter(Itor2[labels==i],
                    ent2[labels==i])
        tit += str(np.round(np.mean(ent2[labels==i]),3)) + '/'
        tit += str(np.round(np.mean(Itor2[labels==i]),3)) + ', '
#    plt.title(tit)
#    plt.legend(np.unique(labels).astype(str))    
    plt.xticks([], [])
    plt.yticks([], [])
    
    plt.savefig('Figs/' + rat_name + '_' + mod_name + '_' + sess_name + '_class_entropy_infoscore_sws', bbox_inches='tight', pad_inches=0.02)  


    f = np.load('Results/Orig/' + rat_name + '_' + mod_name + '_sws_c0_box_rec2_alignment_stats.npz', allow_pickle = True)
    dist = f['dist']
    corr = f['corr']
    f.close()

    plt.figure()
    tit = 'dist/corr: '
    for i in np.unique(labels):
        plt.scatter(corr[labels==i],
                    dist[labels==i])
        tit += str(np.round(np.mean(dist[labels==i]),3)) + '/'
        tit += str(np.round(np.mean(corr[labels==i]),3)) + ', '

#    plt.title(tit)
#    plt.legend(np.unique(labels).astype(str))    
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig('Figs/' + rat_name + '_' + mod_name + '_' + sess_name + '_class_corr_dist', bbox_inches='tight', pad_inches=0.02)

    tit = 'e/I: '
    plt.figure()
    for i in np.unique(labels):
#        plt.scatter(np.arange(sum(labels<i),
#                              sum(labels<i)+sum(labels==i)),
        plt.scatter(dist[labels==i],
                    ent1[labels==i])
        tit += str(np.round(np.mean(ent1[labels==i]),3)) + '/'
        tit += str(np.round(np.mean(dist[labels==i]),3)) + ', '
#    plt.title(tit)
#    plt.legend(np.unique(labels).astype(str))    
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig('Figs/' + rat_name + '_' + mod_name + '_' + sess_name + '_class_entropy_dist_box', bbox_inches='tight', pad_inches=0.02)  

    tit = 'e/I: '
    plt.figure()
    for i in np.unique(labels):
#        plt.scatter(np.arange(sum(labels<i),
#                              sum(labels<i)+sum(labels==i)),
        plt.scatter(dist[labels==i],
                    ent2[labels==i])
        tit += str(np.round(np.mean(ent2[labels==i]),3)) + '/'
        tit += str(np.round(np.mean(dist[labels==i]),3)) + ', '
#    plt.title(tit)
#    plt.legend(np.unique(labels).astype(str))    
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig('Figs/' + rat_name + '_' + mod_name + '_' + sess_name + '_class_entropy_dist_sws', bbox_inches='tight', pad_inches=0.02)

def circmean(angs):
    return np.arctan2(np.mean(np.sin(angs)),np.mean(np.cos(angs)) )

def circstd(angs):
    return np.sqrt(-2 * np.log(np.sqrt(np.square(np.sum(np.sin(angs))) + np.square(np.sum(np.cos(angs)))) / len(angs)))



def optimize_distr(rat_name, mod_name, sess_name, var1, var2, var3, var4, inds):    
    suminds = np.zeros(3)
    for i in range(3):
        suminds[i] = sum(inds==i)
    labels = np.zeros(len(var1), dtype = int)
    distr = np.zeros((len(var1)))
    a = 0.5
    b = np.pi/2
    for i3,c in enumerate(var3):
            labels[((var1<a) & ((var2>=b) | (var2<=-b))) & (var3 < c)] = 0
            labels[((var1>=a) & ((var2>=b) | (var2<=-b))) & (var3 < c)] = 1

            labels[((var1<a) & ((var2>b) | (var2>-b))) & (var3 < c)] = 2
            labels[((var1>=a) & ((var2>b) | (var2>-b))) & (var3 < c)] = 3
            
            labels[((var1<a) & ((var2>=b) | (var2<=-b))) & (var3 >= c)] = 4
            labels[((var1>=a) & ((var2>=b) | (var2<=-b))) & (var3 >= c)] = 5

            labels[((var1<a) & ((var2>b) | (var2>-b))) & (var3 >= c)] = 6
            labels[((var1>=a) & ((var2>b) | (var2>-b))) & (var3 >= c)] = 7
            distmp = np.zeros((8, 3))
            for i in [0,1,2]:
                for j in range(8):
                    distmp[j, i] = sum((labels==j) & (inds == i))
                distmp[:, i] /= suminds[i]
            distmp = np.max(distmp,1)
            distr[i3] = sum(distmp[np.argsort(distmp)[-3:]]) 

    maxtmp = np.max(distr)
    maxargtmp = np.unravel_index(np.argmax(distr), np.shape(distr))
    distr = np.zeros((len(var1)))
    for i3,c in enumerate(var4):
            labels[((var1<a) & ((var2>=b) | (var2<=-b))) & (var4 < c)] = 0
            labels[((var1>=a) & ((var2>=b) | (var2<=-b))) & (var4 < c)] = 1

            labels[((var1<a) & ((var2>b) | (var2>-b))) & (var4 < c)] = 2
            labels[((var1>=a) & ((var2>b) | (var2>-b))) & (var4 < c)] = 3
            
            labels[((var1<a) & ((var2>=b) | (var2<=-b))) & (var4 >= c)] = 4
            labels[((var1>=a) & ((var2>=b) | (var2<=-b))) & (var4 >= c)] = 5

            labels[((var1<a) & ((var2>b) | (var2>-b))) & (var4 >= c)] = 6
            labels[((var1>=a) & ((var2>b) | (var2>-b))) & (var4 >= c)] = 7
            distmp = np.zeros((8, 3))
            for i in [0,1,2]:
                for j in range(8):
                    distmp[j, i] = sum((labels==j) & (inds == i))
                distmp[:, i] /= suminds[i]
            distmp = np.max(distmp,1)
            distr[i3] = sum(distmp[np.argsort(distmp)[-3:]]) 
    maxtmp1 = np.max(distr)
    if maxtmp > maxtmp1:
        maxargs = maxargtmp
        print('1 ' + str(maxtmp))
        var5 = var3.copy()
    else:
        maxargs = np.unravel_index(np.argmax(distr), np.shape(distr))
        print('2 ' + str(maxtmp1))
        var5 = var4.copy()
    print(maxargs)
    print(var5[maxargs])
    labels[((var1<a) & ((var2>=b) | (var2<=-b))) & (var5 < c)] = 0
    labels[((var1>=a) & ((var2>=b) | (var2<=-b))) & (var5 < c)] = 1

    labels[((var1<a) & ((var2>b) | (var2>-b))) & (var5 < c)] = 2
    labels[((var1>=a) & ((var2>b) | (var2>-b))) & (var5 < c)] = 3
    
    labels[((var1<a) & ((var2>=b) | (var2<=-b))) & (var5 >= c)] = 4
    labels[((var1>=a) & ((var2>=b) | (var2<=-b))) & (var5 >= c)] = 5

    labels[((var1<a) & ((var2>b) | (var2>-b))) & (var5 >= c)] = 6
    labels[((var1>=a) & ((var2>b) | (var2>-b))) & (var5 >= c)] = 7
    distmp = np.zeros((8, 3))
    for i in [0,1,2]:
        for j in range(8):
            distmp[j, i] = sum((labels==j) & (inds == i))
        distmp[:, i] /= suminds[i]
    print(distmp)
    print(np.bincount(labels))
    return maxargs
rat_names = str(sys.argv[1].strip()).split(',')
mod_names = str(sys.argv[2].strip()).split(',')
sess_names = str(sys.argv[3].strip()).split(',')
distr_theta = []
distr_burst = []
distr_all = []
mvl_thresh = 0.5
mu_thresh = np.pi/2
isi_thresh = 39
submodule_classification = {}
import glob
fn = glob.glob('distr_all*')
if len(fn) == 1:
    for rat_name in rat_names:
        for mod_name in mod_names:
            for sess_name in sess_names:
                if ((rat_name == 'shane') & (mod_name in ('mod2','mod3', 'mod4'))) | ((rat_name == 'quentin') & (mod_name in ('mod3', 'mod4'))):
                    continue
                if (rat_name == 'roger') & (sess_name[:3] in ('rem', 'sws')):
                    sess_name += '_rec2'
                print(rat_name, sess_name, mod_name)
                if (rat_name == 'roger') & (mod_name == 'mod1') & (sess_name[:3] == 'sws'):
                    ind = clustering(rat_name,mod_name,sess_name, bSmooth = True)
                    inds, m2, m22 = get_inds(rat_name, mod_name, sess_name)
                    print(inds,m2,m22, ind)
                    labels_all = np.concatenate((inds[:, np.newaxis], m2[:, np.newaxis], m22[:, np.newaxis], ind[:, np.newaxis]),1)
                    np.savez('roger_mod1_subclasses', labels = labels_all)

                    mu, mvl = get_theta_tuning(rat_name, mod_name, sess_name)
                    isi, acorr, bin_times,isimean  = get_isi_acorr(rat_name, mod_name, sess_name, maxt = 0.1)
                    isi_mc, isi_rat, acorr_mc, acorr_rat = get_burst_stats(isi,acorr, bin_times)
                    print(isi_mc,isimean)
                    isi_mc = isimean
                    for isi_thresh in [39]:
                        for mu_thresh in [np.pi/2]:#np.linspace(np.pi/2-0.5,np.pi/2+0.5, 100):
                            for mvl_thresh in [0.5]:#np.linspace(0.4, 0.8, 100):
                                labels = np.zeros(len(ind), dtype = int)
                                labels[((mvl<=mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh))) & (isi_mc<=isi_thresh)] = 3
                                labels[((mvl<=mvl_thresh) & ((mu>=mu_thresh) | (mu<=-mu_thresh))) & (isi_mc>isi_thresh)] = 2
                                labels[((mvl>mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh))) & (isi_mc<=isi_thresh)] = 1
#                                labels[((mvl<=mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh)))] = 3
#                                labels[((mvl<=mvl_thresh) & ((mu>=mu_thresh) | (mu<=-mu_thresh)))] = 2
#                                labels[((mvl>mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh)))] = 1
#                                labels[((mu<mu_thresh) & (mu>-mu_thresh))] = 3
#                                labels[((mu>=mu_thresh) | (mu<=-mu_thresh))] = 2
#                                labels[((mu<mu_thresh) & (mu>-mu_thresh))] = 1
#                                labels[(isi_mc<=isi_thresh)] = 3
#                                labels[(isi_mc>isi_thresh)] = 2
                                labels = 3-labels
                                distr = np.zeros(4)
                                for i in [0,1,2,3]:
                                    distr[i] = sum(labels==i)/len(labels)
                                distr_all.append(distr)            
                                numind = np.bincount(ind)
                                distmp = np.zeros((3))
                                distmp1 = np.zeros((3))
                                for i in [0,1,2]:
                                    distmp[i] = sum((labels==i) & (ind == i))/numind[i]
                                    distmp1[i] = sum((labels==i) & (ind == i))
                                print(isi_thresh,mu_thresh,mvl_thresh, ':', sum(distmp), distmp, sum(distmp1), distmp1)

#                    plot_stats(rat_name, mod_name, sess_name, mu-np.pi, mvl, ind, 'mumvl')#, zs = isi_mc)
#                    plot_stats(rat_name, mod_name, sess_name, mu-np.pi, isi_mc, ind, 'mumc')#, zs = mvl)
#                    plot_stats(rat_name, mod_name, sess_name, isi_mc, mvl, ind, 'mvlmc')#, zs = mu)
                    inds_tmp,m2_tmp, m22_tmp = get_inds(rat_name, mod_name, sess_name)
                    cell_id = []
                    for i in range(len(m2_tmp)):
                        cell_id.append(str(m2_tmp[i]) + '_' + str(m22_tmp[i]))

                    submodule_classification[rat_name + '_' + mod_name] = {'cell_id':cell_id, 'classes': labels} 
                    plot_stats(rat_name, mod_name, sess_name, mu-np.pi, mvl, labels, 'mumvl')#, zs = isi_mc)
                    plot_stats(rat_name, mod_name, sess_name, mu-np.pi, isi_mc, labels, 'mumc')#, zs = mvl)
                    plot_stats(rat_name, mod_name, sess_name, isi_mc, mvl, labels, 'mvlmc')#, zs = mu)


                else:
                    mu, mvl = get_theta_tuning(rat_name, mod_name, sess_name)
                    isi, acorr, bin_times,isimean  = get_isi_acorr(rat_name, mod_name, sess_name, maxt = 0.1)
 #                   isi = get_isi(rat_name, mod_name, sess_name)
  #                  isi_mc, isi_rat = get_burst_stats(isi)
                    isi_mc, isi_rat, acorr_mc, acorr_rat = get_burst_stats(isi,acorr, bin_times)
                    print(isi_mc,isimean)
                    isi_mc = isimean 
                    labels = np.zeros(len(isi_mc), dtype = int)
                    labels[((mvl<=mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh))) & (isi_mc<=isi_thresh)] = 3
                    labels[((mvl<=mvl_thresh) & ((mu>=mu_thresh) | (mu<=-mu_thresh))) & (isi_mc>isi_thresh)] = 2
                    labels[((mvl>mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh))) & (isi_mc<=isi_thresh)] = 1

                    #labels[((mvl<=mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh))) & (isi_mc<=isi_thresh)] = 3
                    #labels[((mvl<mvl_thresh) & ((mu>=mu_thresh) | (mu<=-mu_thresh))) & (isi_mc>isi_thresh)] = 2
                    #labels[((mvl>=mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh))) & (isi_mc<=isi_thresh)] = 1
#                    labels[((mu<mu_thresh) & (mu>-mu_thresh))] = 3
#                    labels[((mu>=mu_thresh) | (mu<=-mu_thresh))] = 2

#                    labels[(isi_mc<=isi_thresh)] = 3
#                    labels[(isi_mc>isi_thresh)] = 2
                    labels = 3-labels
                    distr = np.zeros(4)
                    for i in [0,1,2,3]:
                        distr[i] = sum(labels==i)/len(labels)
                    distr_all.append(distr)            
                    plot_stats(rat_name, mod_name, sess_name, mu-np.pi, mvl, labels, 'mumvl')#, zs = isi_mc)
                    plot_stats(rat_name, mod_name, sess_name, mu-np.pi, isi_mc, labels, 'mumc')#, zs = mvl)
                    plot_stats(rat_name, mod_name, sess_name, isi_mc, mvl, labels, 'mvlmc')#, zs = mu)

                    inds_tmp,m2_tmp, m22_tmp = get_inds(rat_name, mod_name, sess_name)
                    cell_id = []
                    for i in range(len(m2_tmp)):
                        cell_id.append(str(m2_tmp[i]) + '_' + str(m22_tmp[i]))

                    submodule_classification[rat_name + '_' + mod_name] = {'cell_id':cell_id, 'classes': labels} 

#    np.savez('thetamediansubmodule_classifications', submodule_classification=submodule_classification)
#    np.savez('distr_all', distr_all = distr_all)
else:
    print(fn)
    f = np.load(fn[0], allow_pickle = True)
    distr_all = f['distr_all'][()]
    f.close()
    print(distr_all)
    plot_distr(distr_all, 'all')
#plot_distr(distr_burst, 'burst')

"""                        
                    plot_stats(acorr_rat, isi_mc, labels, '8')
                    print(' ')
#                    print(acorr_rat, acorr_mc)    
                    for isi_thresh in np.linspace(15,35, 50):
                        labels = np.zeros(len(ind), dtype = int)

                        labels[((mvl<=mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh))) & (acorr_rat>isi_thresh)] = 3
                        labels[((mvl<mvl_thresh) & ((mu>=mu_thresh) | (mu<=-mu_thresh))) & (acorr_rat<=isi_thresh)] = 2
                        labels[((mvl>=mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh))) & (acorr_rat>isi_thresh)] = 1
                        labels = 3-labels
                        distr = np.zeros(4)
                        for i in [0,1,2,3]:
                            distr[i] = sum(labels==i)/len(labels)
                        numind = np.bincount(ind)
                        distmp = np.zeros((3))
                        distmp1 = np.zeros((3))
                        for i in [0,1,2]:
                            distmp[i] = sum((labels==i) & (ind == i))/numind[i]
                            distmp1[i] = sum((labels==i) & (ind == i))
                        print(isi_thresh, ':', sum(distmp), distmp, sum(distmp1), distmp1)
                        print(distr)
                    plot_stats(mu, acorr_rat, ind, '3')
                    plot_stats(mu, acorr_rat, labels, '4')
                    plot_stats(acorr_rat, isi_mc, labels, '9')


                    for isi_thresh in [42]:
                        labels = np.zeros(len(ind), dtype = int)
                        labels[((mvl<=mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh))) & (acorr_mc<=isi_thresh)] = 3
                        labels[((mvl<mvl_thresh) & ((mu>=mu_thresh) | (mu<=-mu_thresh))) & (acorr_mc>isi_thresh)] = 2
                        labels[((mvl>=mvl_thresh) & ((mu<mu_thresh) & (mu>-mu_thresh))) & (acorr_mc<=isi_thresh)] = 1
                        labels = 3-labels
                        distr = np.zeros(4)
                        for i in [0,1,2,3]:
                            distr[i] = sum(labels==i)/len(labels)
                        numind = np.bincount(ind)
                        distmp = np.zeros((3))
                        distmp1 = np.zeros((3))
                        for i in [0,1,2]:
                            distmp[i] = sum((labels==i) & (ind == i))/numind[i]
                            distmp1[i] = sum((labels==i) & (ind == i))
                        print(isi_thresh, ':', sum(distmp), distmp, sum(distmp1), distmp1)
                        print(distr)
#                    distr_all.append(distr)
                    plot_stats(mu, acorr_mc, ind, '1')
                    plot_stats(mu, acorr_mc, labels, '2')
                    plot_stats(acorr_mc, acorr_rat, ind, '5')
                    plot_stats(acorr_mc, isi_mc, ind, '6')
                    plot_stats(acorr_rat, isi_mc, ind, '7')
"""
#                    plot_stats(mu, mvl, ind, 'mumvl_ind')
#                    plot_stats(mu, mvl, labels, 'mumvl')

#                    plot_stats(mu, isi_mc, ind, 'mumc_ind')
#                    plot_stats(mu, isi_mc, labels, 'mumc')

#                    plot_stats(mvl, isi_mc, ind, 'mvlmc_ind')
#                    plot_stats(mvl, isi_mc, labels, 'mvlmc')


# In[ ]:


#################### Toroidal alignment ##################

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

matplotlib.use('Agg')

args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name_0 = args[3].strip()
sess_name_1 = args[4].strip()
_2_PI = 2*np.pi
num_shuffle = 100
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)
bins_torus = np.linspace(0,_2_PI, numangsint)
file_name = rat_name + '_' + mod_name  + '_' + sess_name_0 + '_' + sess_name_1
sig1 = 15
############################## Load 1 ############################
if (rat_name == 'roger') & (sess_name_0[:3] in ('rem', 'sws')):
    sess_name_0 += '_rec2'
if sess_name_0 == 'sws_c0_rec2':    
    spikes_1 = load_spikes(rat_name, mod_name, 'sws_rec2')
#    sspikes_1 = load_spikes_dec(rat_name, mod_name,'sws_rec2', bSmooth = True, sig = sig1)
    sspikes_1 = load_spikes(rat_name, mod_name,'sws_rec2', bSmooth = True, bSpeed = True)

else:
    spikes_1 = load_spikes(rat_name, mod_name, sess_name_0)
#    sspikes_1 = load_spikes_dec(rat_name, mod_name,sess_name_0, bSmooth = True, sig = sig1)
    sspikes_1 = load_spikes(rat_name, mod_name,sess_name_0, bSmooth = True, bSpeed = True)
#if sess_name_0[:3] in ('box', 'maz'):
#    xx,yy, speed = load_pos(rat_name, sess_name_0, bSpeed = True)
#    sspikes_1 = sspikes_1[speed>2.5,:] 

times_1 = np.where(np.sum(spikes_1>0, 1)>=1)[0]
spikes_1 = spikes_1[times_1,:]
sspikes_1 = sspikes_1[times_1,:]

file_name_1 = rat_name + '_' + mod_name + '_' +  sess_name_0
    
f = np.load('Data/' + file_name_1 + '_newdecoding.npz',  allow_pickle = True)
call = f['coordsnew']
if sess_name_0[:3] == 'box':
    callbox_1 = call.copy()
else:    
    callbox_1 = f['coordsbox']
c11all_1 = call[:,0]
c12all_1 = call[:,1]
f.close()    
############################## Load 2 ############################
if (rat_name == 'roger') & (sess_name_1[:3] in ('rem', 'sws')):
    sess_name_1 += '_rec2'
if sess_name_1 == 'sws_c0_rec2':    
    spikes_2 = load_spikes(rat_name, mod_name, 'sws_rec2')
#    sspikes_2 = load_spikes_dec(rat_name, mod_name,'sws_rec2', bSmooth = True, sig = sig1)
    sspikes_2 = load_spikes(rat_name, mod_name,'sws_rec2', bSmooth = True, bSpeed = True)

else:
    spikes_2 = load_spikes(rat_name, mod_name, sess_name_1)
#    sspikes_2 = load_spikes_dec(rat_name, mod_name,sess_name_1, bSmooth = True, sig = sig1)
    sspikes_2 = load_spikes(rat_name, mod_name,sess_name_1, bSmooth = True, bSpeed = True)
    

file_name_2 = rat_name + '_' +  mod_name + '_' + sess_name_1

times_2 = np.where(np.sum(spikes_2>0, 1)>=1)[0]
spikes_2 = spikes_2[times_2,:]
if sess_name_1[:3] in ('box'):
    xx,yy, speed = load_pos(rat_name, sess_name_1, bSpeed = True)
    xx = xx[speed>2.5]
    yy = yy[speed>2.5]
    sspikes_2 = sspikes_2[speed>2.5,:]
    xx = xx[times_2]
    yy = yy[times_2]
sspikes_2 = sspikes_2[times_2,:]


f = np.load('Data/' + file_name_2 + '_newdecoding.npz',  allow_pickle = True)
call = f['coordsnew']
if sess_name_1[:3] == 'box':
    callbox_2 = call.copy()
else:    
    callbox_2 = f['coordsbox']
c11all_2 = call[:,0]
c12all_2 = call[:,1]
f.close()    
num_neurons = len(spikes_1[0,:])

############################## compare ############################
cells_all = range(num_neurons)

def rot_coord(params1,params2, c1, c2, p):    
    rot_mat = np.zeros((2,2))
    if np.abs(np.cos(params1[0])) < np.abs(np.cos(params2[0])):        
        print('nonrot')
        cc1 = c2.copy()
        cc2 = c1.copy()
        y = params1.copy()
        x = params2.copy()
        p = np.flip(p)
    else:   
        print('rot')
        cc1 = c1.copy()
        cc2 = c2.copy()
        x = params1.copy()
        y = params2.copy()  
    print(p, x[1], y[1])
    if p[1] ==-1:
        cc2 = (2*np.pi-cc2)
    if p[0] ==-1:
        cc1 = (2*np.pi-cc1)
    alpha = (y[0]-x[0])
    if (alpha < 0) & (np.abs(alpha) > np.pi/2):
        print('1')
        cctmp = cc2.copy()
        cc2 = cc1.copy()
        cc1 = cctmp
    if (alpha < 0) & (np.abs(alpha) < np.pi/2):
        cc1 = (2*np.pi-cc1 +  np.pi/3*cc2)
        print('2')
    elif np.abs(alpha) > np.pi/2:
        cc2 = (cc2 + np.pi/3*cc1)
        print('3')

    return np.concatenate((cc1[:,np.newaxis], cc2[:,np.newaxis]),1)%_2_PI


f = np.load('Results/Orig/' + file_name_1 + '_para.npz', allow_pickle = True)
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
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p1 = -1

nnans = ~np.isnan(m2b_1)
mtot = m2b_1[nnans]%(2*np.pi)
p = p2b_1.copy()
x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
p2 = 1
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p2 = -1


cc1 = rot_coord(p1b_1,p2b_1, c11all_1, c12all_1, (p1,p2))
if sess_name_0[:3] not in ('box'):
    cbox1 = rot_coord(p1b_1,p2b_1, callbox_1[:,0], callbox_1[:,1], (p1,p2))
else:
    cbox1 = cc1.copy()

f = np.load('Results/Orig/' + file_name_2 + '_para.npz', allow_pickle = True)
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
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p1 = -1

nnans = ~np.isnan(m2b_2)
mtot = m2b_2[nnans]%(2*np.pi)
p = p2b_2.copy()
x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
p2 = 1
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p2 = -1
cc2 = rot_coord(p1b_2,p2b_2, c11all_2, c12all_2, (p1,p2))

if sess_name_0[:3] not in ('box'):
    cbox2 = rot_coord(p1b_2,p2b_2, callbox_2[:,0], callbox_2[:,1], (p1,p2))
else:
    cbox2 = cc2.copy()


pshift = np.arctan2(np.mean(np.sin(cbox1 - cbox2),0), np.mean(np.cos(cbox1 - cbox2),0))%(2*np.pi)
print(pshift)
cbox1 = (cbox1 - pshift)%(2*np.pi)

"""if sess_name_1[:3] in ('box'):
    m1b_1, m2b_1, xedge,yedge = get_ang_hist(cbox1[:,0], 
        cbox1[:,1], xx,yy)

    fig, ax = plt.subplots(1,1)
    ax.imshow(m1b_1.T, origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
    ax.set_aspect('equal', 'box')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    fig.savefig('Figs/' + file_name_1 + '_box_tmp11.png', bbox_inches='tight', pad_inches=0.02)
    fig, ax = plt.subplots(1,1)
    ax.imshow(m2b_1.T, origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
    ax.set_aspect('equal', 'box')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    fig.savefig('Figs/' + file_name_1 + '_box_tmp12.png', bbox_inches='tight', pad_inches=0.02)


    m1b_1, m2b_1, xedge,yedge = get_ang_hist(cbox2[:,0], 
        cbox2[:,1], xx,yy)
    fig, ax = plt.subplots(1,1)
    ax.imshow(m1b_1.T, origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
    ax.set_aspect('equal', 'box')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    fig.savefig('Figs/' + file_name_2 + '_box_tmp11.png', bbox_inches='tight', pad_inches=0.02)
    fig, ax = plt.subplots(1,1)
    ax.imshow(m2b_1.T, origin = 'lower', extent = [xx.min(),xx.max(),yy.min(),yy.max()])
    ax.set_aspect('equal', 'box')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    fig.savefig('Figs/' + file_name_2 + '_box_tmp12.png', bbox_inches='tight', pad_inches=0.02)

"""


from scipy.stats import binned_statistic

numangsint = 16
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)
masscenters_1 = np.zeros((num_neurons,2))
masscenters_2 = np.zeros((num_neurons,2))

cc1 = (cc1 - pshift)%_2_PI
mtot_2 = np.zeros((num_neurons, numangsint_1, numangsint_1))
mtot_1 = np.zeros((num_neurons, numangsint_1, numangsint_1))

xv, yv = np.meshgrid(bins[0:-1] + (bins[1:] -bins[:-1])/2, 
                 bins[0:-1] + (bins[1:] -bins[:-1])/2)
pos  = np.concatenate((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis]),1)
ccos = np.cos(pos)
csin = np.sin(pos)
sig = 2.75
corr = np.zeros(num_neurons)
print(cc1.shape, sspikes_1.shape)
for n in np.arange(num_neurons):
    mtot_1[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc1[:,0],cc1[:,1], 
                                                              sspikes_1[:,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)

    
    mtot0 = np.rot90(mtot_1[n, :, :].copy(), 1).T
    m0 = mtot0.copy()
    m0[np.isnan(m0)] = np.mean(m0[~np.isnan(m0)])
    m0 = smooth_tuning_map(m0, numangsint, sig, bClose = False)
    mtot0 = mtot0.flatten()
    nnans = ~np.isnan(mtot0)
    centcos = np.sum(np.multiply(ccos[nnans],mtot0[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot0[nnans,np.newaxis]),0)
    masscenters_1[n,:] = np.arctan2(centsin,centcos)%_2_PI

    mtot_2[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc2[:,0],cc2[:,1], 
                                                              sspikes_2[:,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)
    
    mtot1 = np.rot90(mtot_2[n, :, :].copy(), 1).T
    m1 = mtot1.copy()
    m1[np.isnan(m1)] = np.mean(m1[~np.isnan(m1)])
    m1 = smooth_tuning_map(m1, numangsint, sig, bClose = False)
    mtot1 = mtot1.flatten()
    nnans = ~np.isnan(mtot1)
    centcos = np.sum(np.multiply(ccos[nnans],mtot1[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot1[nnans,np.newaxis]),0)
    masscenters_2[n,:] = np.arctan2(centsin,centcos)%_2_PI
    corr[n] = pearsonr(m0.flatten(), m1.flatten())[0]
dist = np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2), np.cos(masscenters_1 - masscenters_2))))


print(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2), np.cos(masscenters_1 - masscenters_2)))))

np.savez_compressed('Results/Orig/' + file_name + '_alignment3',
                    mtot_1 = mtot_1,
                    mtot_2 = mtot_2,
                    masscenters_1 = masscenters_1,
                    masscenters_2 = masscenters_2,
                    corr = corr,
                    dist = dist
                   )
#################### Toroidal alignment same proj ##################

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

matplotlib.use('Agg')

args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name_1 = args[3].strip()
sess_name_2 = args[4].strip()
_2_PI = 2*np.pi
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)
bins_torus = np.linspace(0,_2_PI, numangsint)
file_name = rat_name + '_' + mod_name  + '_' + sess_name_1 + '_' + sess_name_2

############################## Load 1 ############################
if (rat_name == 'roger') & (sess_name_1[:3] in ('rem', 'sws')):
    sess_name_1 += '_rec2'
if sess_name_1 == 'sws_c0_rec2':    
    sspikes_1 = load_spikes(rat_name, mod_name, 'sws_rec2', bSmooth = True)
    spikes_1 = load_spikes(rat_name, mod_name, 'sws_rec2')
else:
    spikes_1 = load_spikes(rat_name, mod_name, sess_name_1)
    sspikes_1 = load_spikes(rat_name, mod_name, sess_name_1, bSmooth = True)
num_neurons = len(spikes_1[0,:])

times_1 = np.where(np.sum(spikes_1>0, 1)>=1)[0]
spikes_1 = spikes_1[times_1,:]

file_name_1 = rat_name + '_' + mod_name + '_' +  sess_name_1
    
f = np.load('Data/' + file_name_1 + '_newdecoding.npz',  allow_pickle = True)
call = f['coordsnew']
c11all_11 = call[:,0]
c12all_11 = call[:,1]
f.close()    


f = np.load('Results/Orig/' + file_name_1 + '_pers_analysis.npz' , allow_pickle = True)
coords_1 = f['coords'] 
movetimes_1 = f['movetimes']
indstemp_1 = f['indstemp']
f.close()

centcosall_1 = np.zeros((num_neurons, 2, 1200))
centsinall_1 = np.zeros((num_neurons, 2, 1200))
dspk =preprocessing.scale(sspikes_1[movetimes_1[indstemp_1],:])

k = 1200
for neurid in range(num_neurons):
    spktemp = dspk[:, neurid].copy()
    centcosall_1[neurid,:,:] = np.multiply(np.cos(coords_1[:, :]*2*np.pi),spktemp)
    centsinall_1[neurid,:,:] = np.multiply(np.sin(coords_1[:, :]*2*np.pi),spktemp)


############################## Load 2 ############################
if (rat_name == 'roger') & (sess_name_2[:3] in ('rem', 'sws')):
    sess_name_2 += '_rec2'
if sess_name_2 == 'sws_c0_rec2':    
    spikes_2 = load_spikes(rat_name, mod_name, 'sws_rec2')
    sspikes_2 = load_spikes(rat_name, mod_name, 'sws_rec2', bSmooth = True)
else:
    spikes_2 = load_spikes(rat_name, mod_name, sess_name_2)
    sspikes_2 = load_spikes(rat_name, mod_name, sess_name_2, bSmooth = True)

file_name_2 = rat_name + '_' +  mod_name + '_' + sess_name_2

times_2 = np.where(np.sum(spikes_2>0, 1)>=1)[0]
spikes_2 = spikes_2[times_2,:]

f = np.load('Data/' + file_name_2 + '_newdecoding.npz',  allow_pickle = True)
call = f['coordsnew']
c11all_22 = call[:,0]
c12all_22 = call[:,1]
f.close()    


f = np.load('Results/Orig/' + file_name_2 + '_pers_analysis.npz' , allow_pickle = True)
coords_2 = f['coords'] 
movetimes_2 = f['movetimes']
indstemp_2 = f['indstemp']
f.close()

centcosall_2 = np.zeros((num_neurons, 2, 1200))
centsinall_2 = np.zeros((num_neurons, 2, 1200))
dspk =preprocessing.scale(sspikes_2[movetimes_2[indstemp_2],:])
for neurid in range(num_neurons):
    spktemp = dspk[:, neurid].copy()
    centcosall_2[neurid,:,:] = np.multiply(np.cos(coords_2[:, :]*2*np.pi),spktemp)
    centsinall_2[neurid,:,:] = np.multiply(np.sin(coords_2[:, :]*2*np.pi),spktemp)
############################## map 2 onto 1 ############################

sig = 15
if sess_name_1[:6] == 'sws_c0':
    dspk = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = True, sig = sig)
else:
    dspk = load_spikes_dec(rat_name, mod_name, sess_name_1, bSmooth = True, sig = sig)

if sess_name_1[:3] in ('box', 'maz'):
    xx,yy, speed = load_pos(rat_name, sess_name_1, bSpeed = True)
    dspk = dspk[speed>2.5,:]


dspk = preprocessing.scale(dspk)
dspk = dspk[times_1,:]

a = np.zeros((len(dspk[:,0]), 2, num_neurons))
for n in range(num_neurons):
    a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall_2[n,:,:],1))

c = np.zeros((len(dspk[:,0]), 2, num_neurons))
for n in range(num_neurons):
    c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall_2[n,:,:],1))

mtot2 = np.sum(c,2)
mtot1 = np.sum(a,2)
coordsnew_12 = np.arctan2(mtot2,mtot1)%(2*np.pi)
c11all_12 = coordsnew_12[:,0]
c12all_12 = coordsnew_12[:,1]


############################## map 2 onto 1 ############################

sig = 15
if sess_name_2[:6] == 'sws_c0':
    dspk = load_spikes_dec(rat_name, mod_name, 'sws_rec2', bSmooth = True, sig = sig)
else:
    dspk = load_spikes_dec(rat_name, mod_name, sess_name_2, bSmooth = True, sig = sig)

if sess_name_2[:3] in ('box', 'maz'):
    xx,yy, speed = load_pos(rat_name, sess_name_2, bSpeed = True)
    dspk = dspk[speed>2.5,:]


dspk = preprocessing.scale(dspk)
dspk = dspk[times_2,:]

a = np.zeros((len(dspk[:,0]), 2, num_neurons))
for n in range(num_neurons):
    a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall_1[n,:,:],1))

c = np.zeros((len(dspk[:,0]), 2, num_neurons))
for n in range(num_neurons):
    c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall_1[n,:,:],1))

mtot2 = np.sum(c,2)
mtot1 = np.sum(a,2)
coordsnew_21 = np.arctan2(mtot2,mtot1)%(2*np.pi)
c11all_21 = coordsnew_21[:,0]
c12all_21 = coordsnew_21[:,1]


############################## compare ############################
sspikes_1 = sspikes_1[times_1,:]
sspikes_2 = sspikes_2[times_2,:]


cells_all = range(num_neurons)
masscenters_11 = np.zeros((num_neurons,2))
masscenters_12 = np.zeros((num_neurons,2))
masscenters_22 = np.zeros((num_neurons,2))
masscenters_21 = np.zeros((num_neurons,2))
mtot_22 = np.zeros((num_neurons, numangsint_1, numangsint_1))
mtot_21 = np.zeros((num_neurons, numangsint_1, numangsint_1))
mtot_11 = np.zeros((num_neurons, numangsint_1, numangsint_1))
mtot_12 = np.zeros((num_neurons, numangsint_1, numangsint_1))

cc22 = np.concatenate((c11all_22[:,np.newaxis], c12all_22[:,np.newaxis]),1)
nnans2 =  ~np.isnan(cc22[:,0])
cc21 = np.concatenate((c11all_21[:,np.newaxis], c12all_21[:,np.newaxis]),1)

cc11 = np.concatenate((c11all_11[:,np.newaxis], c12all_11[:,np.newaxis]),1)
nnans1 =  ~np.isnan(cc11[:,0])
cc12 = np.concatenate((c11all_12[:,np.newaxis], c12all_12[:,np.newaxis]),1)

xv, yv = np.meshgrid(bins[0:-1] + (bins[1:] -bins[:-1])/2, 
                 bins[0:-1] + (bins[1:] -bins[:-1])/2)
pos  = np.concatenate((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis]),1)
ccos = np.cos(pos)
csin = np.sin(pos)
for n in cells_all:
    mtot_11[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc11[:,0],cc11[:,1], 
                                                              spikes_1[nnans1,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)

    mtot_12[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc12[:,0],cc12[:,1], 
                                                              spikes_1[nnans1,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)
    
    mtot_21[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc21[:,0],cc21[:,1], 
                                                              spikes_2[nnans2,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)
    mtot_22[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc22[:,0],cc22[:,1], 
                                                              spikes_2[nnans2,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)

    mtot = mtot_11[n:n+1, :, :].copy().T
    mtot = mtot.flatten()
    nnans = ~np.isnan(mtot)
#    mtot /= mtot[nnans].sum()
    centcos = np.sum(np.multiply(ccos[nnans],mtot[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot[nnans,np.newaxis]),0)
    masscenters_11[n,:] = np.arctan2(centsin,centcos)%_2_PI

    mtot = mtot_12[n:n+1, :, :].copy().T
    mtot = mtot.flatten()
    nnans = ~np.isnan(mtot)
#    mtot /= mtot[nnans].sum()
    centcos = np.sum(np.multiply(ccos[nnans],mtot[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot[nnans,np.newaxis]),0)
    masscenters_12[n,:] = np.arctan2(centsin,centcos)%_2_PI

    mtot = mtot_22[n:n+1, :, :].copy().T
    mtot = mtot.flatten()
    nnans = ~np.isnan(mtot)
#    mtot /= mtot[nnans].sum()
    centcos = np.sum(np.multiply(ccos[nnans],mtot[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot[nnans,np.newaxis]),0)
    masscenters_22[n,:] = np.arctan2(centsin,centcos)%_2_PI

    mtot = mtot_21[n:n+1, :, :].copy().T
    mtot = mtot.flatten()
    nnans = ~np.isnan(mtot)
#    mtot /= mtot[nnans].sum()
    centcos = np.sum(np.multiply(ccos[nnans],mtot[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot[nnans,np.newaxis]),0)
    masscenters_21[n,:] = np.arctan2(centsin,centcos)%_2_PI

np.savez_compressed('Results/Orig/' + file_name + '_alignment_same_proj2',
                    mtot_11 = mtot_11,
                    mtot_22 = mtot_22,
                    mtot_12 = mtot_12,
                    mtot_21 = mtot_21,
                    masscenters_11 = masscenters_11,
                    masscenters_22 = masscenters_22,
                    masscenters_12 = masscenters_12,
                    masscenters_21 = masscenters_21,
                   )

#################### Toroidal alignment shufle ##################
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

matplotlib.use('Agg')

args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name_0 = args[3].strip()
sess_name_1 = args[4].strip()
SIN_PI_3 = np.sin(np.pi/3)
_2_PI = 2*np.pi
rot_mat = np.array([[1, -SIN_PI_3], [0, 1]])
num_shuffle = 1000
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)
sig = 2
PI_2 = np.pi/2
bounds = np.array([[0, _2_PI], [0, _2_PI], [-np.pi, np.pi]]).T
npoints = 1000
bins_torus = np.linspace(0,_2_PI, numangsint)
file_name = rat_name + '_' + mod_name  + '_' + sess_name_0 + '_' + sess_name_1

############################## Load 1 ############################
if (rat_name == 'roger') & (sess_name_0[:3] in ('rem', 'sws')):
    sess_name_0 += '_rec2'
if sess_name_0 == 'sws_c0_rec2':    
    spikes_1 = load_spikes(rat_name, mod_name, 'sws_rec2')
else:
    spikes_1 = load_spikes(rat_name, mod_name, sess_name_0)

file_name_1 = rat_name + '_' + mod_name + '_' +  sess_name_0
    
f = np.load('Results/Orig/' + file_name_1 + '_decoding.npz',  allow_pickle = True)
c11all_1 = f['c11all']
c12all_1 = f['c12all']    
f.close()    
############################## Load 2 ############################
if (rat_name == 'roger') & (sess_name_1[:3] in ('rem', 'sws')):
    sess_name_1 += '_rec2'
if sess_name_1 == 'sws_c0_rec2':    
    spikes_2 = load_spikes(rat_name, mod_name, 'sws_rec2')
else:
    spikes_2 = load_spikes(rat_name, mod_name, sess_name_1)
file_name_2 = rat_name + '_' +  mod_name + '_' + sess_name_1

f = np.load('Results/Orig/' + file_name_2 + '_decoding.npz',  allow_pickle = True)
c11all_2 = f['c11all']
c12all_2 = f['c12all']    
f.close()    

############################## Acorr 1 ############################
num_neurons = len(spikes_1[0,:])
acorr1 = np.zeros((numangsint_1, numangsint_1))
nnans1 = ~np.isnan(c11all_1)
for neurid in range(num_neurons):
    mtot_all, x_edge, y_edge, circ = binned_statistic_2d(c11all_1[nnans1],c12all_1[nnans1], 
        spikes_1[nnans1,neurid], statistic='mean', bins=bins_torus, range=None, expand_binnumbers=True)
    mtot_all[np.isnan(mtot_all)] = 0
    mtot1 =  normscale(mtot_all.copy())
    acorr1 += pearson_correlate2d(mtot1, mtot1, fft = True, mode = 'same')

############################## Fit Gaussian to autocorrelation ############################
cindex = np.unravel_index(acorr1.argmax(), acorr1.shape)
acorr1[cindex[0], cindex[1]] = 0
acorr1[cindex[0], cindex[1]] = acorr1.max()+1e-2
acorr1 = normscale(acorr1)

pos = np.indices((numangsint-1, numangsint-1)).T/(numangsint-1)*_2_PI
datatmp = acorr1.flatten()/acorr1.sum()*npoints
datatmp = np.round(datatmp,0)
datatmp = datatmp.astype(int)
n_points = datatmp.sum()
X = np.zeros((n_points,2))
num = np.zeros(n_points)
prev = 0
for i in range(datatmp.shape[0]):
    if datatmp[i]> 0 :
        X[prev:prev+ datatmp[i],:] = np.unravel_index(i, acorr1.shape)
        num[prev:prev+ datatmp[i]]  = datatmp[i]
        prev += datatmp[i]
x,y = X.T/acorr1.shape[0]*_2_PI
                  
    
params = fit_bivariate_normal(x,y)
params = np.max(np.concatenate((np.array(params)[np.newaxis,:], bounds[0:1,:]+1e-20),0),0)
params = np.min(np.concatenate((np.array(params)[np.newaxis,:], bounds[1:2,:]-1e-20),0),0)
errorfunction = lambda p: np.ravel(gaussian_2d(*p)(pos) - acorr1.T)
(sigma2, sigma1, alpha)  = optimize.least_squares(errorfunction, params, bounds = bounds)['x']
alpha = (alpha + PI_2)%_2_PI
if alpha>np.pi:
    alpha = -(_2_PI - alpha)
deg = alpha/np.pi*180
deg_true0 = ~((deg >0) & (deg <=90)) | ((deg >-180) & (deg <=-90))

############################## acorr2 2 ############################

acorr2 = np.zeros((numangsint_1, numangsint_1))
nnans2 = ~np.isnan(c11all_2)
for neurid in range(num_neurons):
    mtot_all, x_edge, y_edge, circ = binned_statistic_2d(c11all_2[nnans2],c12all_2[nnans2], 
        spikes_2[nnans2,neurid], statistic='mean', bins=bins_torus, range=None, expand_binnumbers=True)
    mtot_all[np.isnan(mtot_all)] = 0
    mtot1 =  normscale(mtot_all.copy())
    acorr2 += pearson_correlate2d(mtot1, mtot1, fft = True, mode = 'same')

############################## Fit Gaussian to autocorrelation ############################
cindex = np.unravel_index(acorr2.argmax(), acorr2.shape)
acorr2[cindex[0], cindex[1]] = 0
acorr2[cindex[0], cindex[1]] = acorr2.max()+1e-2
acorr2 = normscale(acorr2)

pos = np.indices((numangsint-1, numangsint-1)).T/(numangsint-1)*_2_PI
datatmp = acorr2.flatten()/acorr2.sum()*npoints
datatmp = np.round(datatmp,0)
datatmp = datatmp.astype(int)
n_points = datatmp.sum()
X = np.zeros((n_points,2))
num = np.zeros(n_points)
prev = 0
for i in range(datatmp.shape[0]):
    if datatmp[i]> 0 :
        X[prev:prev+ datatmp[i],:] = np.unravel_index(i, acorr2.shape)
        num[prev:prev+ datatmp[i]]  = datatmp[i]
        prev += datatmp[i]
x,y = X.T/acorr2.shape[0]*_2_PI
                  
    
params = fit_bivariate_normal(x,y)
params = np.max(np.concatenate((np.array(params)[np.newaxis,:], bounds[0:1,:]+1e-20),0),0)
params = np.min(np.concatenate((np.array(params)[np.newaxis,:], bounds[1:2,:]-1e-20),0),0)
errorfunction = lambda p: np.ravel(gaussian_2d(*p)(pos) - acorr2.T)
(sigma2, sigma1, alpha)  = optimize.least_squares(errorfunction, params, bounds = bounds)['x']
alpha = (alpha + PI_2)%_2_PI
if alpha>np.pi:
    alpha = -(_2_PI - alpha)
deg = alpha/np.pi*180
deg_true1 = ~((deg >0) & (deg <=90)) | ((deg >-180) & (deg <=-90))


############################## compare ############################
cells_all = range(num_neurons)
masscenters_10 = np.zeros((num_neurons,2))
masscenters_11 = np.zeros((num_neurons,2))
masscenters_12 = np.zeros((num_neurons,2))
masscenters_2 = np.zeros((num_neurons,2))
mtot_2 = np.zeros((num_neurons, numangsint_1, numangsint_1))
mtot_10 = np.zeros((num_neurons, numangsint_1, numangsint_1))
mtot_11 = np.zeros((num_neurons, numangsint_1, numangsint_1))
mtot_12 = np.zeros((num_neurons, numangsint_1, numangsint_1))

if deg_true1:
    cc2 = np.matmul(np.concatenate((c11all_2[:,np.newaxis], c12all_2[:,np.newaxis]),1)/(2*np.pi),rot_mat)%1*_2_PI
else:
    cc2 = np.concatenate((c11all_2[:,np.newaxis], c12all_2[:,np.newaxis]),1)
nnans2 =  ~np.isnan(cc2[:,0])
ccsin2 = np.sin(cc2[nnans2,:])
cccos2 = np.cos(cc2[nnans2,:])
cc2 = np.arctan2(ccsin2, cccos2)+ np.pi

if ~deg_true0:
    rot_mat[0,1]*=-1    
cc10 = np.concatenate((c11all_1[:,np.newaxis], c12all_1[:,np.newaxis]),1)/(2*np.pi)
cc11 = np.matmul(cc10, rot_mat)%1*_2_PI
rot_mat = np.flip(rot_mat,0)
cc12 = np.matmul(cc10, rot_mat)%1*_2_PI
cc10 *=_2_PI

nnans1 =  ~np.isnan(cc10[:,0])
ccsin10 = np.sin(cc10[nnans1,:])
cccos10 = np.cos(cc10[nnans1,:])
cc10 = np.arctan2(ccsin10, cccos10)+ np.pi
ccsin11 = np.sin(cc11[nnans1,:])
cccos11 = np.cos(cc11[nnans1,:])
cc11 = np.arctan2(ccsin11, cccos11)+ np.pi
ccsin12 = np.sin(cc12[nnans1,:])
cccos12 = np.cos(cc12[nnans1,:])
cc12 = np.arctan2(ccsin12, cccos12)+ np.pi


xv, yv = np.meshgrid(bins[0:-1] + (bins[1:] -bins[:-1])/2, 
                 bins[0:-1] + (bins[1:] -bins[:-1])/2)
pos  = np.concatenate((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis]),1)
ccos = np.cos(pos)
csin = np.sin(pos)

for n in cells_all:
    mtot_10[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc10[:,0],cc10[:,1], 
                                                              spikes_1[nnans1,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)
    mtot_10[n, np.isnan(mtot_10[n,:])] = 0
    mtot_11[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc11[:,0],cc11[:,1], 
                                                              spikes_1[nnans1,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)
    mtot_11[n, np.isnan(mtot_11[n,:])] = 0    
    mtot_12[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc12[:,0],cc12[:,1], 
                                                              spikes_1[nnans1,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)
    mtot_12[n, np.isnan(mtot_12[n,:])] = 0
    mtot_2[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc2[:,0],cc2[:,1], 
                                                              spikes_2[nnans2,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)
    mtot_2[n, np.isnan(mtot_2[n,:])] = 0

    mtot = mtot_10[n:n+1, :, :].copy().T
    mtot = mtot.flatten()
    nnans = ~np.isnan(mtot)
    mtot /= mtot[nnans].sum()
    centcos = np.sum(np.multiply(ccos[nnans],mtot[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot[nnans,np.newaxis]),0)
    masscenters_10[n,:] = np.arctan2(centsin,centcos)%_2_PI

    mtot = mtot_11[n:n+1, :, :].copy().T
    mtot = mtot.flatten()
    nnans = ~np.isnan(mtot)
    mtot /= mtot[nnans].sum()
    centcos = np.sum(np.multiply(ccos[nnans],mtot[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot[nnans,np.newaxis]),0)
    masscenters_11[n,:] = np.arctan2(centsin,centcos)%_2_PI

    mtot = mtot_12[n:n+1, :, :].copy().T
    mtot = mtot.flatten()
    nnans = ~np.isnan(mtot)
    mtot /= mtot[nnans].sum()
    centcos = np.sum(np.multiply(ccos[nnans],mtot[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot[nnans,np.newaxis]),0)
    masscenters_12[n,:] = np.arctan2(centsin,centcos)%_2_PI

    mtot = mtot_2[n:n+1, :, :].copy().T
    mtot = mtot.flatten()
    nnans = ~np.isnan(mtot)
    mtot /= mtot[nnans].sum()
    centcos = np.sum(np.multiply(ccos[nnans],mtot[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot[nnans,np.newaxis]),0)
    masscenters_2[n,:] = np.arctan2(centsin,centcos)%_2_PI

############################## Optimize ############################
dist_shuf = np.zeros((num_shuffle,num_neurons))
corr_shuf = np.zeros((num_shuffle,num_neurons))
np.random.seed(47)
for shuf in range(num_shuffle):
    inds = np.arange(num_neurons)
    np.random.shuffle(inds)
    mtot_2_shuf = mtot_2[inds,:,:]
    masscenters_2_shuf = masscenters_2[inds,:]


    p = np.zeros((3,2,4,numangsint_1,numangsint_1))
    p[:] = np.inf
    for h in range(3):
        if h == 0:
            mtemp_1_bu0 = masscenters_10.copy()
        elif h == 1:
            mtemp_1_bu0 = masscenters_11.copy()
        elif h == 2:
            mtemp_1_bu0 = masscenters_12.copy()
        for i in range(2):
            if i == 0:
                mtemp_1_bu1 = mtemp_1_bu0.copy()
            elif i == 1:
                mtemp_1_bu1 = np.flip(mtemp_1_bu0.copy(),1)
            for j in range(4):
                mtemp_1_bu2 = mtemp_1_bu1.copy()
                if j == 1:
                    mtemp_1_bu2[:,0] = _2_PI - mtemp_1_bu1[:,0]
                elif j == 2:
                    mtemp_1_bu2[:,1] = _2_PI - mtemp_1_bu1[:,1]
                elif j == 3:
                    mtemp_1_bu2[:,0] = _2_PI - mtemp_1_bu1[:,0]
                    mtemp_1_bu2[:,1] = _2_PI - mtemp_1_bu1[:,1]
                for k in range(numangsint_1):
                    mtemp = mtemp_1_bu2.copy()                
                    mtemp[:,0] = (mtemp_1_bu2[:,0] + bins[k])%_2_PI
                    for l in range(numangsint_1):
                        mtemp[:,1] = (mtemp_1_bu2[:,1] + bins[l])%_2_PI
                        p[h,i,j,k,l] = np.sum(np.abs(np.arctan2(np.sin(mtemp - masscenters_2), np.cos(mtemp - masscenters_2_shuf))))

    h,i,j,k,l = np.unravel_index(np.argmin(p), np.shape(p))
    if h == 0:
        masscenters_1 = masscenters_10.copy()
        mtot_1 = mtot_10.copy()
    elif h == 1:
        masscenters_1 = masscenters_11.copy()
        mtot_1 = mtot_11.copy()
    elif h == 2:
        masscenters_1 = masscenters_12.copy()
        mtot_1 = mtot_12.copy()

    if i == 1:
        mtot_1 = np.transpose(mtot_1, (0,2,1))
        masscenters_1 = np.flip(masscenters_1,1)
        

    if j == 1:
        mtot_1 = np.flip(mtot_1,1)
        masscenters_1[:,0] =  _2_PI - masscenters_1[:,0]
    elif j == 2:
        mtot_1 = np.flip(mtot_1,2)
        masscenters_1[:,1] =  _2_PI - masscenters_1[:,1]
    elif j == 3:
        mtot_1 = np.flip(np.flip(mtot_1,1),2)
        masscenters_1[:,0] =  _2_PI - masscenters_1[:,0]
        masscenters_1[:,1] =  _2_PI - masscenters_1[:,1]
     
    mtot_1 = np.roll(mtot_1, k, 1)
    mtot_1 = np.roll(mtot_1, l, 2)
    masscenters_1[:,0] = (masscenters_1[:,0] + bins[k])%_2_PI
    masscenters_1[:,1] = (masscenters_1[:,1] + bins[l])%_2_PI
    
    mtot_1[np.isnan(mtot_1)] = 0
    mtot_2_shuf[np.isnan(mtot_2_shuf)] = 0
    for n in np.arange(num_neurons):
        mtot_1[n,:,:] = smooth_tuning_map(mtot_1[n,:,:], numangsint, sig)
        mtot_2_shuf[n,:,:] = smooth_tuning_map(mtot_2_shuf[n,:,:], numangsint, sig)
        corr_shuf[shuf, n] = pearsonr(mtot_1[n,:,:].flatten(), mtot_2_shuf[n,:,:].flatten())[0]
    dist_shuf[shuf, :] =  np.sum(np.abs(np.arctan2(np.sin(masscenters_1 - masscenters_2),
                                  np.cos(masscenters_1 - masscenters_2_shuf))),1)

np.savez_compressed('Results/Orig/' + file_name + '_alignment_shuffle',
                    corr_shuf = corr_shuf,
                    dist_shuf = dist_shuf
                   )

############### Toroidal alignment 1 ########################3

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

matplotlib.use('Agg')

args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name_0 = args[3].strip()
sess_name_1 = args[4].strip()
_2_PI = 2*np.pi
num_shuffle = 100
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)
bins_torus = np.linspace(0,_2_PI, numangsint)
file_name = rat_name + '_' + mod_name  + '_' + sess_name_0 + '_' + sess_name_1
sig1 = 15
############################## Load 1 ############################
if (rat_name == 'roger') & (sess_name_0[:3] in ('rem', 'sws')):
    sess_name_0 += '_rec2'
if sess_name_0 == 'sws_c0_rec2':    
    spikes_1 = load_spikes(rat_name, mod_name, 'sws_rec2')
#    sspikes_1 = load_spikes_dec(rat_name, mod_name,'sws_rec2', bSmooth = True, sig = sig1)
    sspikes_1 = load_spikes(rat_name, mod_name,'sws_rec2', bSmooth = True, bSpeed = True)

else:
    spikes_1 = load_spikes(rat_name, mod_name, sess_name_0)
#    sspikes_1 = load_spikes_dec(rat_name, mod_name,sess_name_0, bSmooth = True, sig = sig1)
    sspikes_1 = load_spikes(rat_name, mod_name,sess_name_0, bSmooth = True, bSpeed = True)
#if sess_name_0[:3] in ('box', 'maz'):
#    xx,yy, speed = load_pos(rat_name, sess_name_0, bSpeed = True)
#    sspikes_1 = sspikes_1[speed>2.5,:] 

if (rat_name == 'roger') & (sess_name_1[:3] in ('rem', 'sws')):
    sess_name_1 += '_rec2'
if sess_name_1 == 'sws_c0_rec2':    
    spikes_2 = load_spikes(rat_name, mod_name, 'sws_rec2')
#    sspikes_2 = load_spikes_dec(rat_name, mod_name,'sws_rec2', bSmooth = True, sig = sig1)
    sspikes_2 = load_spikes(rat_name, mod_name,'sws_rec2', bSmooth = True, bSpeed = True)

else:
    spikes_2 = load_spikes(rat_name, mod_name, sess_name_1)
#    sspikes_2 = load_spikes_dec(rat_name, mod_name,sess_name_1, bSmooth = True, sig = sig1)
    sspikes_2 = load_spikes(rat_name, mod_name,sess_name_1, bSmooth = True, bSpeed = True)
    

file_name_1 = rat_name + '_' + mod_name + '_' +  sess_name_0
    
f = np.load('Data/' + file_name_1 + '_newdecoding.npz',  allow_pickle = True)
call = f['coordsnew']
if sess_name_0[:3] == 'box':
    callbox_1 = call.copy()
else:    
    callbox_1 = f['coordsbox']
c11all_1 = call[:,0]
c12all_1 = call[:,1]
f.close()    

if sess_name_0[:6] =='sws_c0':
    times_1 = f['times']
else:
    times_1 = np.where(np.sum(spikes_1>0, 1)>=1)[0]


spikes_1 = spikes_1[times_1,:]
sspikes_1 = sspikes_1[times_1,:]

############################## Load 2 ############################

file_name_2 = rat_name + '_' +  mod_name + '_' + sess_name_1    

f = np.load('Data/' + file_name_2 + '_newdecoding.npz',  allow_pickle = True)
call = f['coordsnew']
if sess_name_1[:3] == 'box':
    callbox_2 = call.copy()
else:    
    callbox_2 = f['coordsbox']
c11all_2 = call[:,0]
c12all_2 = call[:,1]
if sess_name_1[:6] =='sws_c0':
    times_2 = f['times']
else:
    times_2 = np.where(np.sum(spikes_2>0, 1)>=1)[0]
f.close()    

spikes_2 = spikes_2[times_2,:]
if sess_name_1[:3] in ('box'):
    xx,yy, speed = load_pos(rat_name, sess_name_1, bSpeed = True)
    xx = xx[speed>2.5]
    yy = yy[speed>2.5]
#    sspikes_2 = sspikes_2[speed>2.5,:]
    xx = xx[times_2]
    yy = yy[times_2]
sspikes_2 = sspikes_2[times_2,:]


def rot_coord(params1,params2, c1, c2, p):    
    rot_mat = np.zeros((2,2))
    if np.abs(np.cos(params1[0])) < np.abs(np.cos(params2[0])):        
        print('nonrot')
        cc1 = c2.copy()
        cc2 = c1.copy()
        y = params1.copy()
        x = params2.copy()
        p = np.flip(p)
    else:   
        print('rot')
        cc1 = c1.copy()
        cc2 = c2.copy()
        x = params1.copy()
        y = params2.copy()  
    print(p, x[1], y[1])
    if p[1] ==-1:
        cc2 = (2*np.pi-cc2)
    if p[0] ==-1:
        cc1 = (2*np.pi-cc1)
    alpha = (y[0]-x[0])
    if (alpha < 0) & (np.abs(alpha) > np.pi/2):
        print('1')
        cctmp = cc2.copy()
        cc2 = cc1.copy()
        cc1 = cctmp
    if (alpha < 0) & (np.abs(alpha) < np.pi/2):
        cc1 = (2*np.pi-cc1 +  np.pi/3*cc2)
        print('2')
    elif np.abs(alpha) > np.pi/2:
        cc2 = (cc2 + np.pi/3*cc1)
        print('3')

    return np.concatenate((cc1[:,np.newaxis], cc2[:,np.newaxis]),1)%_2_PI


f = np.load('Results/Orig/' + file_name_1 + '_para.npz', allow_pickle = True)
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
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p1 = -1

nnans = ~np.isnan(m2b_1)
mtot = m2b_1[nnans]%(2*np.pi)
p = p2b_1.copy()
x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
p2 = 1
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p2 = -1


cc1 = rot_coord(p1b_1,p2b_1, c11all_1, c12all_1, (p1,p2))
if sess_name_0[:3] not in ('box'):
    cbox1 = rot_coord(p1b_1,p2b_1, callbox_1[:,0], callbox_1[:,1], (p1,p2))
else:
    cbox1 = cc1.copy()

f = np.load('Results/Orig/' + file_name_2 + '_para.npz', allow_pickle = True)
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
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p1 = -1

nnans = ~np.isnan(m2b_2)
mtot = m2b_2[nnans]%(2*np.pi)
p = p2b_2.copy()
x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
p2 = 1
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p2 = -1
cc2 = rot_coord(p1b_2,p2b_2, c11all_2, c12all_2, (p1,p2))

if sess_name_0[:3] not in ('box'):
    cbox2 = rot_coord(p1b_2,p2b_2, callbox_2[:,0], callbox_2[:,1], (p1,p2))
else:
    cbox2 = cc2.copy()


pshift = np.arctan2(np.mean(np.sin(cbox1 - cbox2),0), np.mean(np.cos(cbox1 - cbox2),0))%(2*np.pi)
print(pshift)
cbox1 = (cbox1 - pshift)%(2*np.pi)
cc1 = (cc1 - pshift)%_2_PI
np.savez_compressed('Results/Orig/' + file_name + '_alignment_dec',
                    pshift = pshift,
                    cbox = cbox1,
                    csess = cc1                    
                   )

############### Toroidal alignment 2 ########################3
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

matplotlib.use('Agg')

args = sys.argv
rat_name = args[1].strip()
mod_name = args[2].strip()
sess_name_0 = args[3].strip()
sess_name_1 = args[4].strip()
_2_PI = 2*np.pi
num_shuffle = 100
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)
bins_torus = np.linspace(0,_2_PI, numangsint)
file_name = rat_name + '_' + mod_name  + '_' + sess_name_0 + '_' + sess_name_1
sig1 = 15
############################## Load 1 ############################
if (rat_name == 'roger') & (sess_name_0[:3] in ('rem', 'sws')):
    sess_name_0 += '_rec2'
if sess_name_0 == 'sws_c0_rec2':    
    spikes_1 = load_spikes(rat_name, mod_name, 'sws_rec2')
#    sspikes_1 = load_spikes_dec(rat_name, mod_name,'sws_rec2', bSmooth = True, sig = sig1)
    sspikes_1 = load_spikes(rat_name, mod_name,'sws_rec2', bSmooth = True, bSpeed = True)

else:
    spikes_1 = load_spikes(rat_name, mod_name, sess_name_0)
#    sspikes_1 = load_spikes_dec(rat_name, mod_name,sess_name_0, bSmooth = True, sig = sig1)
    sspikes_1 = load_spikes(rat_name, mod_name,sess_name_0, bSmooth = True, bSpeed = True)
#if sess_name_0[:3] in ('box', 'maz'):
#    xx,yy, speed = load_pos(rat_name, sess_name_0, bSpeed = True)
#    sspikes_1 = sspikes_1[speed>2.5,:] 

if (rat_name == 'roger') & (sess_name_1[:3] in ('rem', 'sws')):
    sess_name_1 += '_rec2'
if sess_name_1 == 'sws_c0_rec2':    
    spikes_2 = load_spikes(rat_name, mod_name, 'sws_rec2')
#    sspikes_2 = load_spikes_dec(rat_name, mod_name,'sws_rec2', bSmooth = True, sig = sig1)
    sspikes_2 = load_spikes(rat_name, mod_name,'sws_rec2', bSmooth = True, bSpeed = True)

else:
    spikes_2 = load_spikes(rat_name, mod_name, sess_name_1)
#    sspikes_2 = load_spikes_dec(rat_name, mod_name,sess_name_1, bSmooth = True, sig = sig1)
    sspikes_2 = load_spikes(rat_name, mod_name,sess_name_1, bSmooth = True, bSpeed = True)


file_name_1 = rat_name + '_' + mod_name + '_' +  sess_name_0
    
f = np.load('Data/' + file_name_1 + '_newdecoding.npz',  allow_pickle = True)
call = f['coordsnew']
if sess_name_0[:3] == 'box':
    callbox_1 = call.copy()
else:    
    callbox_1 = f['coordsbox']
c11all_1 = call[:,0]
c12all_1 = call[:,1]

if sess_name_0[:6] =='sws_c0':
    times_1 = f['times1']
else:
    times_1 = np.where(np.sum(spikes_1>0, 1)>=1)[0]
f.close()    

spikes_1 = spikes_1[times_1,:]
sspikes_1 = sspikes_1[times_1,:]

############################## Load 2 ############################

file_name_2 = rat_name + '_' +  mod_name + '_' + sess_name_1    

f = np.load('Data/' + file_name_2 + '_newdecoding.npz',  allow_pickle = True)
call = f['coordsnew']
if sess_name_1[:3] == 'box':
    callbox_2 = call.copy()
else:    
    callbox_2 = f['coordsbox']
c11all_2 = call[:,0]
c12all_2 = call[:,1]
if sess_name_1[:6] =='sws_c0':
    times_2 = f['times']
    print('t20', times_2)    
else:
    times_2 = np.where(np.sum(spikes_2>0, 1)>=1)[0]
    print('t21', times_2)    
f.close()    

spikes_2 = spikes_2[times_2,:]
if sess_name_1[:3] in ('box'):
    xx,yy, speed = load_pos(rat_name, sess_name_1, bSpeed = True)
    xx = xx[speed>2.5]
    yy = yy[speed>2.5]
#    sspikes_2 = sspikes_2[speed>2.5,:]
    xx = xx[times_2]
    yy = yy[times_2]
sspikes_2 = sspikes_2[times_2,:]

def rot_coord(params1,params2, c1, c2, p):    
    rot_mat = np.zeros((2,2))
    if np.abs(np.cos(params1[0])) < np.abs(np.cos(params2[0])):        
        print('nonrot')
        cc1 = c2.copy()
        cc2 = c1.copy()
        y = params1.copy()
        x = params2.copy()
        p = np.flip(p)
    else:   
        print('rot')
        cc1 = c1.copy()
        cc2 = c2.copy()
        x = params1.copy()
        y = params2.copy()  
    print(p, x[1], y[1])
    if p[1] ==-1:
        cc2 = (2*np.pi-cc2)
    if p[0] ==-1:
        cc1 = (2*np.pi-cc1)
    alpha = (y[0]-x[0])
    if (alpha < 0) & (np.abs(alpha) > np.pi/2):
        print('1')
        cctmp = cc2.copy()
        cc2 = cc1.copy()
        cc1 = cctmp
    if (alpha < 0) & (np.abs(alpha) < np.pi/2):
        cc1 = (2*np.pi-cc1 +  np.pi/3*cc2)
        print('2')
    elif np.abs(alpha) > np.pi/2:
        cc2 = (cc2 + np.pi/3*cc1)
        print('3')

    return np.concatenate((cc1[:,np.newaxis], cc2[:,np.newaxis]),1)%_2_PI


f = np.load('Results/Orig/' + file_name_1 + '_para.npz', allow_pickle = True)
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
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p1 = -1

nnans = ~np.isnan(m2b_1)
mtot = m2b_1[nnans]%(2*np.pi)
p = p2b_1.copy()
x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
p2 = 1
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p2 = -1


cc1 = rot_coord(p1b_1,p2b_1, c11all_1, c12all_1, (p1,p2))
if sess_name_0[:3] not in ('box'):
    cbox1 = rot_coord(p1b_1,p2b_1, callbox_1[:,0], callbox_1[:,1], (p1,p2))
else:
    cbox1 = cc1.copy()

f = np.load('Results/Orig/' + file_name_2 + '_para.npz', allow_pickle = True)
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
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p1 = -1

nnans = ~np.isnan(m2b_2)
mtot = m2b_2[nnans]%(2*np.pi)
p = p2b_2.copy()
x1 = rotate(x, p[0]*360/(2*np.pi), reshape= False)
pm1 = ((p[2]*x1[nb:-nb,nb:-nb]+p[1])%(2*np.pi))[nnans]-mtot
pm2 = ((2*np.pi-(p[2]*x1[nb:-nb,nb:-nb]+p[1]))%(2*np.pi))[nnans]-mtot
p2 = 1
if np.sum(np.abs(pm1))>np.sum(np.abs(pm2)):
    p2 = -1
cc2 = rot_coord(p1b_2,p2b_2, c11all_2, c12all_2, (p1,p2))

if sess_name_0[:3] not in ('box'):
    cbox2 = rot_coord(p1b_2,p2b_2, callbox_2[:,0], callbox_2[:,1], (p1,p2))
else:
    cbox2 = cc2.copy()


pshift = np.arctan2(np.mean(np.sin(cbox1 - cbox2),0), np.mean(np.cos(cbox1 - cbox2),0))%(2*np.pi)
cbox1 = (cbox1 - pshift)%(2*np.pi)
cc1 = (cc1 - pshift)%_2_PI
np.savez_compressed('Results/Orig/' + file_name + '_alignment_dec',
                    pshift = pshift,
                    cbox = cbox1,
                    csess = cc1                    
                   )



from scipy.stats import binned_statistic
num_neurons = len(sspikes_1[0,:])
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)
masscenters_1 = np.zeros((num_neurons,2))
masscenters_2 = np.zeros((num_neurons,2))

mtot_2 = np.zeros((num_neurons, numangsint_1, numangsint_1))
mtot_1 = np.zeros((num_neurons, numangsint_1, numangsint_1))

xv, yv = np.meshgrid(bins[0:-1] + (bins[1:] -bins[:-1])/2, 
                 bins[0:-1] + (bins[1:] -bins[:-1])/2)
pos  = np.concatenate((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis]),1)
ccos = np.cos(pos)
csin = np.sin(pos)
sig = 2.75
corr = np.zeros(num_neurons)
for n in np.arange(num_neurons):
    mtot_1[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc1[:,0],cc1[:,1], 
                                                              sspikes_1[:,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)

    
    mtot0 = np.rot90(mtot_1[n, :, :].copy(), 1).T
    m0 = mtot0.copy()
    m0[np.isnan(m0)] = np.mean(m0[~np.isnan(m0)])
    m0 = smooth_tuning_map(m0, numangsint, sig, bClose = False)
    mtot0 = mtot0.flatten()
    nnans = ~np.isnan(mtot0)
    centcos = np.sum(np.multiply(ccos[nnans],mtot0[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot0[nnans,np.newaxis]),0)
    masscenters_1[n,:] = np.arctan2(centsin,centcos)%_2_PI

    mtot_2[n,:,:], x_edge, y_edge,circ = binned_statistic_2d(cc2[:,0],cc2[:,1], 
                                                              sspikes_2[:,n], statistic='mean', 
                                                              bins=bins, range=None, expand_binnumbers=True)
    
    mtot1 = np.rot90(mtot_2[n, :, :].copy(), 1).T
    m1 = mtot1.copy()
    m1[np.isnan(m1)] = np.mean(m1[~np.isnan(m1)])
    m1 = smooth_tuning_map(m1, numangsint, sig, bClose = False)
    mtot1 = mtot1.flatten()
    nnans = ~np.isnan(mtot1)
    centcos = np.sum(np.multiply(ccos[nnans],mtot1[nnans,np.newaxis]),0)
    centsin = np.sum(np.multiply(csin[nnans],mtot1[nnans,np.newaxis]),0)
    masscenters_2[n,:] = np.arctan2(centsin,centcos)%_2_PI
    corr[n] = pearsonr(m0.flatten(), m1.flatten())[0]
dist = np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2), np.cos(masscenters_1 - masscenters_2))))


print(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2), np.cos(masscenters_1 - masscenters_2)))))

np.savez_compressed('Results/Orig/' + file_name + '_alignment3',
                    mtot_1 = mtot_1,
                    mtot_2 = mtot_2,
                    masscenters_1 = masscenters_1,
                    masscenters_2 = masscenters_2,
                    corr = corr,
                    dist = dist
                   )


# In[ ]:


########## torus classifier ###############

import numpy as np
from matplotlib import pyplot as plt
from math import pi, cos, sin
from utils import *
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
from scipy.ndimage import gaussian_filter1d
matplotlib.use('Agg')

alpha_all = np.zeros((len([10,20,30,40,50,60,70,80,90,100,110,120,130,140]), 100))

rat_name = 'roger'
mod_name = 'mod3'
sess_name = 'box'


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
xx = xx[speed>=2.5]
yy = yy[speed>=2.5]

num_decode = np.arange(0,len(xx[:]), 10)
xx = xx[num_decode]
yy = yy[num_decode]
import sys
tens = [int(sys.argv[1].strip())]
angs = np.zeros((len(tens), 1000))
xlen = np.zeros((len(tens), 1000))
ylen = np.zeros((len(tens), 1000))
score1 = np.zeros((len(tens), 1000))
score2 = np.zeros((len(tens), 1000))
names = []
nn = -1
for n in tens:
    tennames = []
    nn += 1
    file_name = 'roger_mod3_box_N' + str(n) + '_'
    filenames = glob.glob('Figs_N/' + file_name + '*.npz') 
    #filenames = glob.glob('Figs_N1/NPZ_files/' + file_name + '*.npz') 
    for j in range(len(filenames)):
        tennames.append(filenames[j])
        f = np.load(filenames[j], allow_pickle=True)
        c11all, c12all = f['call_box']
        f.close()
        m1b_1, m2b_1, xedge,yedge = get_ang_hist(c11all, 
            c12all, xx,yy)
        p1b_1, f1 = fit_sine_wave(m1b_1)
        p2b_1, f2 = fit_sine_wave(m2b_1)

        x,y = rot_para(p1b_1,p2b_1)
        xmin = xedge.min()
        xedge -= xmin
        xmax = xedge.max()
        ymin = yedge.min()
        yedge -= ymin
        ymax = yedge.max()

        angs[nn,j] = ((y[0]-x[0])/(2*np.pi)*360)%360
        xlen[nn,j] = 1/x[2]*xmax
        ylen[nn,j] = 1/y[2]*ymax
        score1[nn,j] = f1
        score2[nn,j] = f2
    names.append(tennames)
    
np.savez('toroidal_classification', angs=angs, xlen=xlen, ylen=ylen, score1=score1, score2=score2, names = names)


import numpy as np
from math import pi, cos, sin
from utils import *

rat_name = 'roger'
mod_name = 'mod3'
sess_name = 'box'

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
xx = xx[speed>=2.5]
yy = yy[speed>=2.5]

num_decode = np.arange(0,len(xx[:]), 10)
xx = xx[num_decode ]
yy = yy[num_decode]
import sys
tens = [int(sys.argv[1].strip())]
angs = np.zeros((len(tens), 1000))
xlen = np.zeros((len(tens), 1000))
ylen = np.zeros((len(tens), 1000))
score1 = np.zeros((len(tens), 1000))
score2 = np.zeros((len(tens), 1000))
nn = 0
f = np.load('toroidal_classification' + str(50) + '.npz', allow_pickle = True)
angs[nn,:] = f['angs'][0,:]
xlen[nn,:] = f['xlen'][0,:]
ylen[nn,:] = f['ylen'][0,:]
score1[nn,:] = f['score1'][0,:]
score2[nn,:] = f['score2'][0,:]
namestemp = f['names']
f.close()


#file_name = 'roger_mod3_box_N' + str(50) + '_'
#filenames = ['Figs_N/' + file_name + '4914.npz', 'Figs_N/' + file_name + '4915.npz']
names = []
for n in tens:
    tennames = []
    for nms in namestemp:
        tennames.append(nms)
    filenames = glob.glob('Figs_N1/NPZ_files/' + file_name + '*.npz') 
    for j in range(len(filenames)):

        tennames.append(filenames[j])
        f = np.load(filenames[j], allow_pickle=True)
        c11all, c12all = f['call_box']
        f.close()
        m1b_1, m2b_1, xedge,yedge = get_ang_hist(c11all, 
            c12all, xx,yy)
        p1b_1, f1 = fit_sine_wave(m1b_1)
        p2b_1, f2 = fit_sine_wave(m2b_1)

        x,y = rot_para(p1b_1,p2b_1)
        xmin = xedge.min()
        xedge -= xmin
        xmax = xedge.max()
        ymin = yedge.min()
        yedge -= ymin
        ymax = yedge.max()

        angs[nn,j] = ((y[0]-x[0])/(2*np.pi)*360)%360
        xlen[nn,j] = 1/x[2]*xmax
        ylen[nn,j] = 1/y[2]*ymax
        score1[nn,j] = f1
        score2[nn,j] = f2
    names.append(tennames)
    nn += 1
np.savez('toroidal_classification' + str(tens[0]), angs=angs, xlen=xlen, ylen=ylen, score1=score1, score2=score2, names = names)


import numpy as np
from matplotlib import pyplot as plt
from math import pi, cos, sin
from utils import *

SIN_PI_3 = np.sin(np.pi/3)
_2_PI = 2*np.pi
rot_mat = np.array([[1, -SIN_PI_3], [0, 1]])
num_shuffle = 100
numangsint = 51
numangsint_1 = numangsint-1
bins = np.linspace(0,_2_PI, numangsint)
sig = 2
PI_2 = np.pi/2
bounds = np.array([[0, _2_PI], [0, _2_PI], [-np.pi, np.pi]]).T
npoints = 1000
bins_torus = np.linspace(0,_2_PI, numangsint)

nn = -1
alpha_all = np.zeros((len([10,20,30,40,50,60,70,80,90,100,110,120,130,140]), 100))

from utils import *
from matplotlib import animation, cm, transforms, pyplot as plt
from matplotlib.collections import PathCollection
from scipy.ndimage import gaussian_filter1d
rat_name = 'roger'
mod_name = 'mod3'
sess_name = 'box'


bSmooth = False    
if bSmooth:
    spkname = 'sspikes'
else:
    spkname = 'spikes'
    
f = np.load('Data/' + rat_name + '_' + mod_name + '_' + sess_name + '_spikes_03_5.npz', allow_pickle = True)
spk = f[spkname]
f.close()

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
xx = xx[speed>=2.5]
yy = yy[speed>=2.5]


for n in [10,20,30,40,50,60,70,80,90,100,110,120,130,140]:
    nn += 1
    file_name = 'roger_mod3_box_pers_analysis_N' + str(n) + '_'
    filenames = glob.glob('num_neurons_files/' + file_name + '*.npz') 
    for j in range(len(filenames)):
        f = np.load(filenames[j], allow_pickle=True)
        c11all, c12all = f['call_box']
        f.close()


        num_neurons = len(spikes[0,:])
        acorr1 = np.zeros((numangsint_1, numangsint_1))
        nnans1 = ~np.isnan(c11all)
        for neurid in range(num_neurons):
            mtot_all, x_edge, y_edge, circ = binned_statistic_2d(c11all[nnans1],c12all[nnans1], 
                spikes[nnans1,neurid], statistic='mean', bins=bins_torus, range=None, expand_binnumbers=True)
            mtot_all[np.isnan(mtot_all)] = 0
            mtot1 =  normscale(mtot_all.copy())
            acorr1 += pearson_correlate2d(mtot1, mtot1, fft = True, mode = 'same')

        ############################## Fit Gaussian to autocorrelation ############################
        cindex = np.unravel_index(acorr1.argmax(), acorr1.shape)
        acorr1[cindex[0], cindex[1]] = 0
        acorr1[cindex[0], cindex[1]] = acorr1.max()+1e-2
        acorr1 = normscale(acorr1)

        pos = np.indices((numangsint-1, numangsint-1)).T/(numangsint-1)*_2_PI
        datatmp = acorr1.flatten()/acorr1.sum()*npoints
        datatmp = np.round(datatmp,0)
        datatmp = datatmp.astype(int)
        n_points = datatmp.sum()
        X = np.zeros((n_points,2))
        num = np.zeros(n_points)
        prev = 0
        for i in range(datatmp.shape[0]):
            if datatmp[i]> 0 :
                X[prev:prev+ datatmp[i],:] = np.unravel_index(i, acorr1.shape)
                num[prev:prev+ datatmp[i]]  = datatmp[i]
                prev += datatmp[i]
        x,y = X.T/acorr1.shape[0]*_2_PI


        params = fit_bivariate_normal(x,y)
        params = np.max(np.concatenate((np.array(params)[np.newaxis,:], bounds[0:1,:]+1e-20),0),0)
        params = np.min(np.concatenate((np.array(params)[np.newaxis,:], bounds[1:2,:]-1e-20),0),0)
        errorfunction = lambda p: np.ravel(gaussian_2d(*p)(pos) - acorr1.T)
        if np.isnan(sum(params)):
            params = np.zeros(3)
            
        (sigma2, sigma1, alpha)  = optimize.least_squares(errorfunction, params, bounds = bounds)['x']
        alpha = (alpha)%_2_PI
        if alpha>np.pi:
            alpha = -(_2_PI - alpha)
        deg = alpha/(2*np.pi)*360
        alpha_all[nn,j] = deg


        #deg_true0 = ~((deg >0) & (deg <=90)) | ((deg >-180) & (deg <=-90))


        #if deg_true0:
        #    deg += 90
        #    deg = deg%360    


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(acorr1, origin = 'lower', extent = [0,2*np.pi, 0, 2*np.pi])
        #ax.imshow(acorr1, extent = [0,2*np.pi, 0, 2*np.pi])


        u=np.pi       #x-position of the center
        v=np.pi      #y-position of the center
        a=sigma2*2       #radius on the x-axis
        b=sigma1*2      #radius on the y-axis
        t_rot=deg/(2*np.pi)+np.pi/2 #rotation angle

        t = np.linspace(0, 2*pi, 100)
        Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
             #u,v removed to keep the same center location
        R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])  
             #2-D rotation matrix

        Ell_rot = np.zeros((2,Ell.shape[1]))
        for i in range(Ell.shape[1]):
            Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

        #ax.plot( u+Ell[0,:] , v+Ell[1,:] )     #initial ellipse
        ax.plot( u+Ell_rot[0,:] , v+Ell_rot[1,:],'red', lw = 2 )    #rotated ellipse

        ax.axis('off')
        fig.savefig(filenames[j][18:-4])
        plt.close()
        

m1b_1, m2b_1, xedge,yedge = get_ang_hist(c11all, 
    c12all, xx,yy)
p1b_1 = fit_sine_wave(m1b_1)
p2b_1 = fit_sine_wave(m2b_1)

x,y = rot_para(p1b_1,p2b_1)
xmin = xedge.min()
xedge -= xmin
xmax = xedge.max()
ymin = yedge.min()
yedge -= ymin
ymax = yedge.max()
print(((y[0]-x[0])/(2*np.pi)*360)%360)
print(1/x[2]*xmax)
print(1/y[2]*ymax)
print('')
#            plot_para(ax,xedge,yedge, x, y, colors[sess_name[:3]])
plot_stripes(xedge,yedge, p2b_1, m2b_1, file_name + '_1')
plot_stripes(xedge,yedge, p1b_1, m1b_1, file_name + '_2')


# In[ ]:



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
            Lvals[~fg] = -np.sum( (ys[~fg]-yt[~fg])**2 )             
        else:
            H = np.dot(P[:, i], X_test)
            expH = np.exp(H)
            yt[fg] = expH
            finthechat = (np.ravel(np.log(factorial(ys[fg]))))
            Lvals[fg] = (np.ravel(ys[fg]*H - expH)) - finthechat

    leastsq = np.sum( (ys-yt)**2 )
    #print('LEAST SQ', leastsq)
    ym = np.mean(ys)
    #return (np.sum((yt-ym)**2) / np.sum((ys-ym)**2))
    return yt, (1. - leastsq/np.sum((ys-ym)**2)), P, Lvals

for rat_name, mod_name, sess_name in (('shane', 'mod1', 'box'),
                                      ('quentin', 'mod1', 'box'),
                                     ('quentin', 'mod2', 'box'),
                                     ('roger', 'mod1', 'box'),
                                     ('roger', 'mod3', 'box'),
                                     ('roger', 'mod4', 'box')): 
    sspikes_mod1 = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bSpeed = True)       
    xx,yy,speed = load_pos(rat_name, sess_name, bSpeed = True)       
    xx = xx[speed>2.5]
    yy = yy[speed>2.5]


    dim = len(sspikes_mod1[0,:])
    ph_classes = [0,1] 
    num_circ = len(ph_classes)
    dec_tresh = 0.99
    metric = 'cosine'
    maxdim = 1
    coeff = 47
    num_neurons = len(sspikes_mod1[0,:])
    active_times = 15000
    num_times = 5
    times_cube = np.arange(0,len(sspikes_mod1[:,0]),num_times)
    movetimes = np.sort(np.argsort(np.sum(sspikes_mod1[times_cube,:],1))[-active_times:])
    movetimes = times_cube[movetimes]

    pca_mod1,var_exp_mod1, eigval_mod1  = pca(preprocessing.scale(sspikes_mod1[movetimes,:]), dim = dim)

    nF2 = 3
    num_bins = 15

    P_space_all_mod1 = np.zeros((num_neurons, num_bins**2, nF2) )
    spacescores_mod1 = np.zeros((num_neurons))

    __, num_neurons = np.shape(pca_mod1)
    xxyy = np.zeros((len(xx[movetimes]),2))
    xxyy[:,0] = xx[movetimes]+0
    xxyy[:,1] = yy[movetimes]+0

    for lamtemp in np.array([1,10,100,1000]):
        LAM = np.sqrt(lamtemp)
        GoGaussian = True
        yps_all_mod1 = []
        Lvals_space_mod1 = []
        for n in np.arange(0, num_neurons, 1): 
            yps_all_mod1.append([])
            Lvals_space_mod1.append([])

            yps_all_mod1[n], spacescores_mod1[n], P_space_all_mod1[n,:, :], Lvals_space_mod1[n] = dirtyglm(xxyy[:,:],
                                                                                                           pca_mod1[:,n],
                                                                                                           num_bins, False, LAM,
                                                                                                           GoGaussian, nF2)
        file_name = rat_name + mod_name + sess_name
        np.savez('PCA_dev_'+ file_name + '_' + str(int(lamtemp)),  spacescores_mod1 = spacescores_mod1, P_space_all_mod1 = P_space_all_mod1)


# In[ ]:


########## Cup product simplex #################
import gudhi
import numpy as np

rat_name = 'roger'
mod_name = 'mod3'
sess_name = 'box'
if (rat_name == 'roger') & ((sess_name=='rem') | (sess_name[:3] == 'sws')):
    sess_name += '_rec2'
file_name = rat_name + '_' + mod_name + '_' + sess_name


f = np.load('' + file_name + '_d.npz', allow_pickle=True)
d = f['d']
f.close()

maxlen = 4.6 
dim = 2
rips = gudhi.RipsComplex(distance_matrix = d, max_edge_length= maxlen)
tree = rips.create_simplex_tree(max_dimension = dim)
file = open(file_name + '.sim', "w")
g = tree.get_skeleton(dim)
for e in g:
    s = e[0]
    ss = '('
    for i in range(len(s)-1):
        ss += str(e[0][i]) + ','
    ss += str(e[0][-1]) + ')\n'
    
    file.write(ss)
file.close()


# In[ ]:


import os
import gudhi
import np

rat_name = sys.argv[0].strip()
mod_name = sys.argv[1].strip()
sess_name = sys.argv[2].strip()
sz = sys.argv[3].strip()

file_name = mod_name + '_' + rat_name + '_' + sess_name + sz

f = np.load('Results/pers_analysis_' + file_name + '.npz')
d = f['dist']
f.close()

dim = 2
maxlen = 4.84
rips = gudhi.RipsComplex(distance_matrix = d, max_edge_length= maxlen)
tree = rips.create_simplex_tree(max_dimension = dim)
file = open('Data/cup_' + file_name + '.sim', "w")
g = tree.get_skeleton(dim)
for e in g:
    s = e[0]
    ss = '('
    for i in range(len(s)-1):
        ss += str(e[0][i]) + ','
    ss += str(e[0][-1]) + ')\n'
    
    file.write(ss)
file.close()

os.system('cmd /k "cringsim Data/cup_' + filename + '.sim > Results/Orig/cup_' + filename + '.txt"')


# In[ ]:


######### fit glm #######################

import sys, glob
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import binned_statistic_2d
import numpy as np
from sklearn import preprocessing
from scipy.special import factorial
from sklearn.metrics import explained_variance_score
from utils import *

rat_name = sys.argv[1]
mod_name = sys.argv[2]
sess_name = sys.argv[3]
LAM = float(sys.argv[4])
GoGaussian = sys.argv[5]=='1'

############ Load ################
if (sess_name[:3] in ('rem', 'sws')) & (rat_name == 'roger'):    
    sess_name += '_rec2'

file_name = rat_name + '_' + mod_name + '_' + sess_name 
if GoGaussian:
    NAME = '%s_Gaussian_LAM%08d_FULL'%(file_name,int(float(LAM*100.)))
    spk = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bBox = False)
else:
    NAME = '%s_Poisson__LAM%08d_FULL'%(file_name,int(float(LAM*100.)))
    spk = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bBox = False)
T, num_neurons = np.shape(spk)
decs = np.zeros((T, 2))
file_name = rat_name + '_' + mod_name + '_' + sess_name
f = np.load('Results/Orig/' + file_name + '_decoding.npz', allow_pickle = True)
decs[:,0] = f['c11all']
decs[:,1] = f['c12all']
f.close()

nF = 3
num_binsDEC = 10
nnans = ~np.isnan(decs[:,0])
T = sum(nnans)
decs = decs[nnans,:]
decs[:,0] = normit(decs[:,0])
decs[:,1] = normit(decs[:,1])
spk = spk[nnans,:]
crossvalidDEC = np.zeros((num_neurons, nF))
LvalsDEC = np.zeros((T, num_neurons))
EvDEC = np.zeros((num_neurons))
yDEC = np.zeros((T, num_neurons))
for whichneuron in range(num_neurons):
    ys = spk[:,whichneuron]
    LvalsDEC[:, whichneuron], crossvalidDEC[whichneuron,:], yDEC[:,whichneuron] = goforgo(decs, ys, LAM, True, num_binsDEC, nF, GoGaussian)
    EvDEC[whichneuron] = explained_variance_score(ys, yDEC[:,whichneuron]) 

bspace = sess_name[:3] not in ('rem', 'sws')
if bspace:
    num_binsX = 30
    xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
    xx = xx[speed>2.5]
    yy = yy[speed>2.5]
    xs = np.zeros((T, 2))
    xs[:,0] = normit(xx)
    xs[:,1] = normit(yy)
    xs = xs[nnans]
    crossvalidX = np.zeros((num_neurons, nF))
    LvalsX = np.zeros((T, num_neurons))
    EvX = np.zeros((num_neurons))
    yX = np.zeros((T, num_neurons))
    for whichneuron in range(num_neurons):
        ys = spk[:,whichneuron]
        LvalsX[:, whichneuron], crossvalidX[whichneuron,:], yX[:,whichneuron] = goforgo(xs, ys, LAM, True, num_binsX, nF, GoGaussian)
        EvX[whichneuron] = explained_variance_score(ys, yX[:,whichneuron]) 

    np.savez_compressed('Results/GLM/' + NAME, 
        crossvalidX = crossvalidX,
        crossvalidDEC = crossvalidDEC,
        LAM = LAM,
        LvalsDEC = LvalsDEC,
        LvalsX = LvalsX,
        EvX = EvX,
        EvDec = EvDEC,
        yX = yX,
        yDEC = yDEC)
else:
    np.savez_compressed('Results/GLM/' + NAME, 
    crossvalidDEC = crossvalidDEC,
    LAM = LAM,
    LvalsDEC = LvalsDEC,
    EvDec = EvDEC,
    yDEC = yDEC)


# In[ ]:


############ Fit GLM Physical space ################

import sys, glob
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import binned_statistic_2d
import numpy as np
from sklearn import preprocessing
from scipy.special import factorial
from sklearn.metrics import explained_variance_score
from utils import *

rat_name = sys.argv[1]
mod_name = sys.argv[2]
sess_name = sys.argv[3]
LAM = np.sqrt(float(sys.argv[4]))
GoGaussian = sys.argv[5]=='1'

############ Load ################
if (sess_name[:3] in ('rem', 'sws')) & (rat_name == 'roger'):    
    sess_name += '_rec2'

file_name = rat_name + '_' + mod_name + '_' + sess_name 
if GoGaussian:
    NAME = '%s_Gaussian_LAM%08d'%(file_name,int(float(LAM*100.)))
    spk = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bBox = False)
else:
    NAME = '%s_Poisson__LAM%08d'%(file_name,int(float(LAM*100.)))
    spk = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bBox = False)
T, num_neurons = np.shape(spk)
decs = np.zeros((T, 2))
file_name = rat_name + '_' + mod_name + '_' + sess_name
f = np.load('Results/Orig/' + file_name + '_decoding.npz', allow_pickle = True)
decs[:,0] = f['c11all']
decs[:,1] = f['c12all']
f.close()
nnans = ~np.isnan(decs[:,0])
T = sum(nnans)
decs = decs[nnans,:]
decs[:,0] = normit(decs[:,0])
decs[:,1] = normit(decs[:,1])
spk = spk[nnans,:]

nF = 3
num_binsX = 30
xx,yy, speed = load_pos(rat_name, sess_name, bSpeed = True)
xx = xx[speed>2.5]
yy = yy[speed>2.5]
xs = np.zeros((T, 2))
xs[:,0] = normit(xx[nnans])
xs[:,1] = normit(yy[nnans])
P = np.zeros((num_neurons, num_binsX**2))
for whichneuron in range(num_neurons):
    P[whichneuron,:] = goforgo(xs, spk[:,whichneuron], LAM, False, num_binsX, nF, GoGaussian)
np.savez_compressed('Results/GLM/' + NAME + '_space_P', P = P)

tmp = np.floor(T/3)
yX = np.zeros_like(spk)
LvalsX = np.zeros_like(spk)
EVX = np.zeros(num_neurons)
for whichneuron in range(num_neurons):
    fg = np.zeros(T)
    P_tmp = P[whichneuron,:]
    for i in range(3):
        if(i==3-1):
            fg[(int(tmp*i)):] = 1
        else:
            fg[(int(tmp*i)):(int(tmp*(i+1)))] = 1
        fg = fg<0.5

        X_test = preprocess_dataX2(xs[~fg,:], num_binsX)     
        if(GoGaussian):
            yX[~fg, whichneuron] = np.dot(P_tmp, X_test)
            LvalsX[~fg,whichneuron] = (spk[:,whichneuron][~fg]-yX[~fg,whichneuron])**2
        else:
            H = np.dot(P_tmp, X_test)
            expH = np.exp(H)
            yX[~fg, whichneuron] = expH 
            finthechat = (np.ravel(np.log(factorial(spk[:,whichneuron][~fg]))))
            LvalsX[~fg, whichneuron] = -(np.ravel(spk[:,whichneuron][~fg]*H - expH)) + finthechat
    EVX[whichneuron] = explained_variance_score(spk[:,whichneuron], yX[:,whichneuron]) 
cov1 = np.cov(spk.T)
cov2 = np.cov(yX.T)
cov5 = np.cov(np.transpose(spk-yX))
np.savez('Results/GLM/' + NAME + 'space_stats', 
	EVX = EVX, covs = (cov1,cov2,cov5), LvalsX = np.mean(LvalsX,0)) 
np.savez('Results/GLM/' + NAME + 'space_pred', yX = yX, LvalsX = LvalsX)


# In[ ]:


############ Fit GLM toroidal ################

import sys, glob
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import binned_statistic_2d
import numpy as np
from sklearn import preprocessing
from scipy.special import factorial
from sklearn.metrics import explained_variance_score
from utils import *

rat_name = sys.argv[1]
mod_name = sys.argv[2]
sess_name = sys.argv[3]
LAM = np.sqrt(float(sys.argv[4]))
GoGaussian = sys.argv[5]=='1'


if (sess_name[:3] in ('rem', 'sws')) & (rat_name == 'roger'):    
    sess_name += '_rec2'

file_name = rat_name + '_' + mod_name + '_' + sess_name 
if GoGaussian:
    NAME = '%s_Gaussian_LAM%08d_FULL'%(file_name,int(float(LAM*100.)))
    spk = load_spikes(rat_name, mod_name, sess_name, bSmooth = True, bBox = False)
else:
    NAME = '%s_Poisson__LAM%08d_FULL'%(file_name,int(float(LAM*100.)))
    spk = load_spikes(rat_name, mod_name, sess_name, bSmooth = False, bBox = False)
T, num_neurons = np.shape(spk)
decs = np.zeros((T, 2))
file_name = rat_name + '_' + mod_name + '_' + sess_name
f = np.load('Results/Orig/' + file_name + '_decoding.npz', allow_pickle = True)
decs[:,0] = f['c11all']
decs[:,1] = f['c12all']
f.close()
nF = 3
num_binsDEC = 10
nnans = ~np.isnan(decs[:,0])
T = sum(nnans)
decs = decs[nnans,:]
decs[:,0] = normit(decs[:,0])
decs[:,1] = normit(decs[:,1])
spk = spk[nnans,:]
P = np.zeros((num_neurons, num_binsDEC**2))
for whichneuron in range(num_neurons):
    P[whichneuron,:] = goforgo(decs, spk[:,whichneuron], LAM, True, num_binsDEC, nF, GoGaussian)
np.savez_compressed('Results/GLM/' + NAME + '_tor_P', P = P)

tmp = np.floor(T/3)
yDEC = np.zeros_like(spk)
LvalsDEC = np.zeros_like(spk)
EVDEC = np.zeros(num_neurons)
for whichneuron in range(num_neurons):
    fg = np.zeros(T)
    P_tmp = P[whichneuron,:]
    for i in range(3):
        if(i==3-1):
            fg[(int(tmp*i)):] = 1
        else:
            fg[(int(tmp*i)):(int(tmp*(i+1)))] = 1
        fg = fg<0.5

        X_test = preprocess_dataX2(decs[~fg,:], num_binsDEC)     
        if(GoGaussian):
            yDEC[~fg, whichneuron] = np.dot(P_tmp, X_test)
            LvalsDEC[~fg,whichneuron] = (spk[:,whichneuron][~fg]-yDEC[~fg,whichneuron])**2
        else:
            H = np.dot(P_tmp, X_test)
            expH = np.exp(H)
            yDEC[~fg, whichneuron] = expH 
            finthechat = (np.ravel(np.log(factorial(spk[:,whichneuron][~fg]))))
            LvalsDEC[~fg, whichneuron] = -(np.ravel(spk[:,whichneuron][~fg]*H - expH)) + finthechat

    EVDEC[whichneuron] = explained_variance_score(spk[:,whichneuron], yDEC[:,whichneuron]) 
    cov3 = np.cov(yDEC.T)
    cov4 = np.cov(np.transpose(spk-yDEC))
    np.savez('Results/GLM/' + NAME + 'tor_stats', 
    	EVDEC = EVDEC, covs = (cov3,cov4), LvalsDEC = np.mean(LvalsDEC,0)) 
    np.savez('Results/GLM/' + NAME + 'tor_pred', yDEC = yDEC, LvalsDEC = LvalsDEC)


# In[ ]:


import sys
import numpy as np
from scipy.stats import binned_statistic_2d
from utils import *

############################## Fit parallelogram to space ############################

rat_name = ''
mod_name = ''
sess_name = ''

t = time.time()    
if (rat_name == 'roger') & (sess_name[:3] in ('rem', 'sws')):    
    sess_name += '_rec2'
boxname = 'box'
if sess_name[-5:] == '_rec2':
    boxname += '_rec2'

t1 = time.time()
t = time.time()    

file_name = rat_name + '_' + mod_name + '_' + sess_name
xx,yy, speed = load_pos(rat_name, boxname, bSpeed = True)

f = np.load('Data/' + file_name + '_decoding.npz', allow_pickle = True)
c11all_orig1 = f['coordsbox'][:,0]
c12all_orig1 = f['coordsbox'][:,1]
if sess_name[:6] =='sws_c0':
    times = f['times']
else:
    spikes = load_spikes(rat_name, mod_name, boxname, bSmooth = False, bSpeed =False, bBox = False)
    spikes = spikes[speed>2.5,:]    
    times = np.where(np.sum(spikes>0, 1)>=1)[0]
    spikes = spikes[times,:]
f.close()

xx = xx[speed>2.5]
yy = yy[speed>2.5]
xx = xx[times]
yy = yy[times]

print(yy.shape, c11all_orig1.shape)

t1 = time.time()
t = time.time()    
m1b_1, m2b_1, xedge,yedge = get_ang_hist(c11all_orig1, 
    c12all_orig1, xx,yy)
t1 = time.time()

p1b_1, f1 = fit_sine_wave(m1b_1)
p2b_1, f2 = fit_sine_wave(m2b_1)

print(p1b_1, p2b_1)

np.savez_compressed('Results/Orig/' + file_name + '_para', 
    p1b_1 = p1b_1, 
    p2b_1 = p2b_1,
    m1b_1 = m1b_1, 
    m2b_1 = m2b_1, 
    xedge = xedge,
    yedge = yedge,
    fun = (f1,f2))

