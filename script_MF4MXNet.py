#! usr/bin/python
# coding=utf-8

# Convolution using mxnet  ### x w
from __future__ import print_function
import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import nd, autograd, gluon
from mxnet.gluon.nn import Dense, ELU, LeakyReLU, LayerNorm, Conv2D, MaxPool2D, Flatten, Activation
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
from mxnet.image import Augmenter
import matplotlib.mlab as mlab
from scipy.signal import tukey

mx.random.seed(1)                      # Set seed for reproducable results

# system
import os, sys, time, datetime, copy
from loguru import logger
config = {
    "handlers": [
        {"sink": "MF4MXNet_{}.log".format(datetime.date.today()), "level":"DEBUG" ,"format": '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>'},
        # {"sink": "Solver_cnn.log",},
        {"sink": sys.stdout, "format": '<green>{time:YYYY-MM-DD}</green> <cyan>{time:HH:mm:ss}</cyan> | <level>{level: <7}</level> | <level>{message}</level>',
        "level": "INFO"},
    ],
    # "extra": {"user": "someone"}
}
#### REF #### https://loguru.readthedocs.io/en/stable/api/logger.html
# DEBUG 10  # INFO 20  # WARNING 30  # ERROR 40  # CRITICAL 50
logger.configure(**config)
logger.debug('#'*40)
from pyinstrument import Profiler  # https://github.com/joerick/pyinstrument
from tqdm import tnrange, tqdm_notebook, tqdm
########## RAY ################
# import ray
# # CPU_COUNT = 40  # cpu_count()
# CPU_COUNT = 2
# logger.info("#" * 30)
# logger.info("CPU_COUNT: {}", CPU_COUNT)
# logger.info("#" * 30)
# ray.init(num_cpus=CPU_COUNT, num_gpus = 0, include_webui=False, ignore_reinit_error=True)
########## RAY ################

def mkdir(path):
    isExists=os.path.exists(path)
 
    if not isExists:
        os.makedirs(path) 
        logger.success(path+' 创建成功')
    else:
        logger.success(path+' 目录已存在')

def EquapEvent(fs, data):
    # Window function
    dwin = tukey(data.size, alpha=1./8)
    sample = data.astype('float32') # (1,fs) ndarray cpu
    psd = np.real(np.fft.ifft(1/np.sqrt(power_vec(sample[0].asnumpy(), fs)))).reshape(1,-1) # (1,fs) np.array
    sample_block = (sample* nd.array(dwin)).expand_dims(0).expand_dims(0)  #(1,1,1,fs) ndarray cup
    sample_psd_block = nd.concat(sample_block, nd.array(psd).expand_dims(0).expand_dims(0), dim=1)    
    return sample_psd_block # (1, 2, 1, 4096) ndarray cpu

def pred_O1Events(deltat, fs, T, C, frac):
    onesecslice = [(65232, 69327) , (65178, 69273), 
     (66142, 70237), (66134, 70229),
     (65902, 69997), (65928, 70023), 
     (65281, 69376), (65294, 69389)]
    llLIGOevents = [file for file in os.listdir('Data_LIGO_Totural') if 'strain' in file]
    llLIGOevents.sort()
    aroundEvents = np.concatenate([np.load('./Data_LIGO_Totural/'+file).reshape(1,-1)[:,onesecslice[index][0]-int((deltat-0.5)*fs):onesecslice[index][1]+int((deltat-0.5)*fs)+1] \
                                     for index, file in enumerate(llLIGOevents)])
    logger.info('data_block: {} | {}', aroundEvents.shape, np.array(llLIGOevents))
    aroundEvents = nd.array(aroundEvents).expand_dims(1)
    logger.info('aroundEvents: {} [cpu ndarray]', aroundEvents.shape)

    bias = 0#fs//2
    # frac = 40
    moving_slide = {}
    spsd_block = {}
    for index, filename in tqdm(enumerate(llLIGOevents), disable=True):
        moving_slide[filename] = np.concatenate([ aroundEvents[index:index+1, 0, i*int(fs*(T/frac))+bias : i*int(fs*(T/frac))+T*fs+bias].asnumpy() for i in range(aroundEvents.shape[-1]) if i*int(fs*(T/frac))+T*fs+bias <=aroundEvents.shape[-1] ], axis=0)#[:160]#[:64 if T == 2 else 128]
        spsd_block[filename] = np.concatenate([np.real(np.fft.ifft(1/np.sqrt(power_vec(i, fs)))).reshape(1,-1) for i in moving_slide[filename]])
        # (64, fs*T)
    logger.info('moving_slide: {} [np.array]', moving_slide[filename].shape)
    logger.info('spsd_block: {} [np.array]', spsd_block[filename].shape)
    time_range = [(i*int(fs*(T/frac))+bias + 20480//2)/fs for i in range(aroundEvents.shape[-1]) if i*int(fs*(T/frac))+T*fs+bias <=aroundEvents.shape[-1] ]

    dwin = tukey(T*fs, alpha=1./8)
    iterator_events, data_psd_events = {}, {}
    for index, (filename_H1, filename_L1) in enumerate(zip(llLIGOevents[::2], llLIGOevents[1::2])):
        data_block_nd = nd.concat(nd.array(moving_slide[filename_H1] * dwin).expand_dims(1), 
                                 nd.array(moving_slide[filename_L1] * dwin).expand_dims(1), dim=1) # (161, C, T*fs)
        psd_block_nd = nd.concat(nd.array(spsd_block[filename_H1]).expand_dims(1), 
                                 nd.array(spsd_block[filename_L1]).expand_dims(1), dim=1) # (161, C, T*fs)

    #      (161, 2, 2, 1, 20480)
        data_psd_events[filename_H1.split('_')[0]] = nd.concat(data_block_nd.expand_dims(1), 
                                                               psd_block_nd.expand_dims(1), dim=1).expand_dims(3)
        events_dataset = gluon.data.ArrayDataset(data_psd_events[filename_H1.split('_')[0]])
        iterator_events[filename_H1.split('_')[0]] = gdata.DataLoader(events_dataset, 8, shuffle=False, last_batch = 'keep', num_workers=0)

    logger.info('data_psd_events: {} | {}', data_psd_events['GW150914'].shape, data_psd_events.keys())

    return iterator_events, time_range


# 计算 PSD
def power_vec(x, fs):
    """
    Input 1-D np.array
    """
    # fs = 4096
    # NFFT = T*fs//8
    # We have assumed it as 1/8.
    NFFT = int((x.size/fs/8.0)*fs)
    # with Blackman window function
    psd_window = np.blackman(NFFT)
    # and a 50% overlap:
    NOVL = NFFT/2

    # -- Calculate the PSD of the data.  Also use an overlap, and window:
    data_psd, freqs = mlab.psd(x, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)

    datafreq = np.fft.fftfreq(x.shape[-1])*fs
    # -- Interpolate to get the PSD values at the needed frequencies
    return np.interp(np.abs(datafreq), freqs, data_psd)

# 计算标准差
def nd_std(x, axis=-1):
    """ Standard Deviation (SD) 
        Note: Do not try 'axis=0'
    """
    return nd.sqrt(nd.square(nd.abs(x - x.mean(axis=axis).expand_dims(axis=axis) )).mean(axis=axis))

class RandomPeakAug(Augmenter):
    """Make RandomPeakAug.
    Parameters
    ----------
    percet : float [0,1]
    p : the possibility the img be rotated
    """
    __slots__ = ['fs', 'T', 'C', 'N', 'margin', 'ori_peak', 'shape_aug']
    def __init__(self, margin, fs, C, ori_peak=None, T=1, rand_jitter = 1):
        super(RandomPeakAug, self).__init__(margin=margin, ori_peak=ori_peak, fs=fs, T=T, C=C)
        self.fs = fs
        self.T = T          # [s]
        self.N = int(fs*T)  # [n]
        self.C = C
        self.margin = int(margin * fs * T ) #[n]
        self.ori_peak = int(ori_peak * fs * T)  if ori_peak else None
        # self.shape_aug = mx.image.RandomCropAug(size=(fs, 1))
        self.rand_jitter = rand_jitter
        # print(C, fs, self.margin, self.ori_peak)

    def __call__(self, src):
        """Augmenter body"""
        assert src.shape[-2:] == (self.C, self.N)  # (nsample, C, N)
        if self.ori_peak is None:
            self.ori_peak = int(src.argmax(axis=2)[0,0].asscalar()) # first+H1 as bench
            logger.debug('self.ori_peak: {}', self.ori_peak)

        # myrelu = lambda x: x if (x>0) and (x<=self.ori_peak*2) else None
        # (nsample, C, 2*(N-margin))
        # full = nd.concatenate([src, nd.zeros(shape=src.shape[:2]+(self.ori_peak*2-self.N,))], axis=2)[:,:,myrelu(self.ori_peak-(self.N-self.margin)):myrelu(self.ori_peak+(self.N-self.margin))]
        full = nd.concat(src, nd.zeros(shape=src.shape[:2]+(self.ori_peak-self.margin,)) , dim=2)[:,:,self.ori_peak-(self.N-self.margin):]
        assert (nd.sum( full[:,:1].argmax(-1) / full[:,:1].shape[-1] )/full[:,:1].shape[0]).asscalar()  == 0.5

        if self.margin == (self.T*self.fs)//2:
            return full

        if self.rand_jitter: # for every sample
            """
            RP = RandomPeakAug(margin=0.1, fs = fs, C = 2, ori_peak=0.9, rand_jitter=0)
            %timeit _ = RP(dataset_GW[pre])
            # 505 ms ± 30.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            """
            randlist= [ (i , i+fs) for i in np.random.randint(low=1,high=(fs-2*self.margin), size= full.shape[0]) if i+fs <= full.shape[-1]]
            assert len(randlist) == full.shape[0]
            return nd.concatenate([ sample.expand_dims(axis=0)[:,:,i:j] for sample, (i, j) in zip(full, randlist) ], axis=0) # (nsample, C, N)

            # full = nd.concatenate([self.shape_aug(sample.swapaxes(0,1).expand_dims(axis=0)) for sample in full ], axis=0) # (nsample, N, C)
            # return full.swapaxes(1,2) # (nsample, C, N)
        else:
            """
            RP = RandomPeakAug(margin=0.1, fs = fs, C = 2, ori_peak=0.9, rand_jitter=1)
            %timeit _ = RP(dataset_GW[pre])
            # 808 µs ± 37.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
            """
            full = full.swapaxes(0,2).expand_dims(axis=0) # (1, 2*(N-margin), C, nsample)
            return self.shape_aug(full.reshape(1,0,-3)).reshape(1,0,self.C,-1).swapaxes(1,3)[0]  # where swapaxes from (1, 2*(N-margin), C, nsample) to (nsample, C, N)

class MatchedFilteringLayer(gluon.HybridBlock):
    def __init__(self, mod, fs,
                 template_H1, 
                 template_L1, 
                 differentiable = False):
        super(MatchedFilteringLayer, self).__init__()
        self.mod = int(mod)
        self.fs = int(fs)

        with self.name_scope():
            # self.weights = self.params.get('weights',
            #                                shape=(hidden_units, 0),
            #                                allow_deferred_init=True)

            self.template_H1 = self.params.get('template_H1',
                                      shape=template_H1.shape,
                                      init=mx.init.Constant(template_H1.asnumpy().tolist()), # Convert to regular list to make this object serializable
                                      differentiable=differentiable)
            self.template_L1 = self.params.get('template_L1',
                                      shape=template_L1.shape,
                                      init=mx.init.Constant(template_L1.asnumpy().tolist()), # Convert to regular list to make this object serializable
                                      differentiable=differentiable)

        self.num_filter_template = self.template_H1.shape[0]
        self.kernel_size = self.template_H1.shape[-1]
        ## Global fs/ctx

        
    def get_module(self, F, data, mod):
        ctx = data.context
        return F.concatenate([data, F.zeros(data.shape[:-1]+(mod - data.shape[-1]%mod, ), ctx=ctx)], axis=len(data.shape)-1).reshape(0,0,-1,mod).sum(axis=-2).expand_dims(2)[:,:,:,::-1]
        # something wrong here for pad??
        # data = F.reshape(F.pad(data, mode="constant", constant_value=0, pad_width=(0,0, 0,0, 0,0, 0,1)), shape=(0,0,-1,mod))
        # return F.reverse(F.expand_dims(F.sum(data, axis=-2), 2), axis=3)        
        
    def hybrid_forward(self, F, data, template_H1, template_L1):
         # data (nsmaple, 2, C, 1, T*fs) gpu nd.array
        data_H1, data_L1 = F.split(data = data, axis=2, num_outputs=2)
        data_H1 = data_H1[:,:,0] # (nsample, 2, 1, T*fs)
        data_L1 = data_L1[:,:,0]
        MF_H1 = self.onedetector_forward(F, data_H1, template_H1)
        MF_L1 = self.onedetector_forward(F, data_L1, template_L1)
        # (nsample, num_filter_template, 1, T*fs)
        return nd.concat(MF_H1.expand_dims(0), MF_L1.expand_dims(0), dim=0)
        
    def onedetector_forward(self, F, data, template):
        # Note: Not working for hybrid blocks/mx.symbol!
        # (8, 1, 1, T*fs), (8, 1, 1, T*fs) <= (8, 2, 1, T*fs)
        data_block_nd, ts_block_nd = F.split(data = data, axis=1, num_outputs=2) 
        # assert F.shape_array(data).size_array().asscalar() == 4 # (8, 1, 1, T*fs)
        # assert F.shape_array(self.weight).size_array().asscalar() == 4
        batch_size = F.slice_axis(F.shape_array(ts_block_nd), axis=0, begin=0, end=1).asscalar()  # 8

        # Whiten data ===========================================================
        data_whiten = F.concatenate( [F.Convolution(data=data_block_nd[i:i+1],   # (8, 1, 1, T*fs)
                                                     weight=ts_block_nd[i:i+1],    # (8, 1, 1, T*fs)
                                                     no_bias=True,
                                                     kernel=(1, self.mod),
                                                     stride=(1,1),
                                                     num_filter=1, 
                                                     pad=(0,self.mod -1),) for i in range(batch_size) ],
                                    axis=0)
        data_whiten = self.get_module(F, data_whiten, self.mod) # (8, 1, 1, T*fs)

        # Whiten template =======================================================
        template_whiten = F.Convolution(data=template,   # (8, 1, 1, T*fs)
                             weight=ts_block_nd,  # (8, 1, 1, T*fs)
                             no_bias=True,
                             kernel=(1, self.mod),
                             stride=(1,1),
                             num_filter=batch_size, 
                             pad=(0,self.mod -1),)
        template_whiten = self.get_module(F, template_whiten, self.kernel_size)
        # template_whiten (8, 8, 1, T*fs)

        # == Calculate the matched filter output in the time domain: ============
        optimal = F.concatenate([ F.Convolution(data=data_whiten[i:i+1],  # (8, 8, 1, T*fs)
                                                 weight=template_whiten[:,i:i+1],  # (8, 8, 1, T*fs)
                                                 no_bias=True,
                                                 kernel=(1, self.kernel_size),
                                                 stride=(1,1),
                                                 num_filter=self.num_filter_template, 
                                                 pad=(0, self.kernel_size -1),) for i in range(batch_size)],
                               axis=0)
        
        optimal = self.get_module(F, optimal, self.mod)
        optimal_time = F.abs(optimal*2/self.fs)
        # optimal_time (8, 8, 1, T*fs)

        # == Normalize the matched filter output: ===============================
        sigmasq = F.concatenate([ F.Convolution(data=template_whiten.swapaxes(0,1)[j:j+1:,i:i+1], # (8, 8, 1, T*fs)
                                                 weight=template_whiten.swapaxes(0,1)[j:j+1:,i:i+1],   # (8, 8, 1, T*fs)
                                                 no_bias=True,
                                                 kernel=(1, self.kernel_size),
                                                 stride=(1,1),
                                                 num_filter=1, 
                                                 pad=(0, self.kernel_size -1),) for j in range(batch_size) for i in range(self.num_filter_template) ],
                                axis=0)
        sigmasq = self.get_module(F, sigmasq, self.kernel_size)[:,:,:,0].reshape(optimal_time.shape[:2])
        sigma = F.sqrt(F.abs( sigmasq/self.fs )).expand_dims(2).expand_dims(2)
        # sigma  (8, 8, 1, 1)
        return F.broadcast_div(optimal_time, sigma) # (8, 8, 1, T*fs)  SNR_MF        
        

class CutHybridLayer(gluon.HybridBlock):
    def __init__(self, margin):
        super(CutHybridLayer, self).__init__()
        extra_range = 0.0
        self.around_range = (1-margin*2)/2
        # self.left = int(fs- np.around(self.around_range + extra_range, 2) * fs)
        # self.right = int(fs+ np.around(self.around_range + extra_range, 2) * fs)+1
        
    def hybrid_forward(self, F, x):
        # (C, nsample, num_filter_template, 1, T*fs)
        return F.max(x, axis=-1).swapaxes(1,0).swapaxes(3,2)
        # if self.around_range == 0:
        #     return F.slice_axis(x, begin=0, end=1, axis=3).swapaxes(1,3)
        # else:
        #     return F.slice_axis(F.Concat(x,x, dim=3), axis=-1, begin=self.left, end=self.right)

def preTemplateFloyd(fs, T, C, shift_size, wind_size, margin,debug = True):
    temp_window = tukey(fs*wind_size, alpha=1./8)
    
    dataset_GW = {}
    keys = {}
    pre = 'train'
    data = np.load('/floyd/input/templates/data_T{}_fs4096_{}{}_{}.npy'.format(T,T*0.9,T*0.9, pre))[:,1] # drop GPS
    dataset_GW[pre] = nd.array(data)[:,:C]  # (1610,C,T*fs) cpu nd.ndarray
    keys[pre] = np.load('/floyd/input/templates/data_T{}_fs4096_{}{}_keys_{}.npy'.format(T, T*0.9,T*0.9,pre))    

    logger.debug('Loading {} data: {}', pre, dataset_GW[pre].shape)
    keys[pre] = pd.Series(keys[pre][:,0])  # pd.DataFrame

    # use equal training masses as template
    equalmass_index = keys['train'][keys['train'].map(lambda x: x.split('|')[0]==x.split('|')[1])].index.tolist()
    nonequalmass_index = keys['train'][keys['train'].map(lambda x: x.split('|')[0]!=x.split('|')[1])].index.tolist()
    template_block = dataset_GW['train'][equalmass_index]  # 35x1x4096 cpu nd.ndarray

    # Move the template peak to center corresponding H1
    d = int(template_block[:,0].argmax(-1)[0].asscalar() - fs*T//2)
    template_block = nd.concat(template_block[:,:,d:], nd.zeros(template_block.shape[:2]+(d,)) , dim=2)[:,:,fs*T//2-wind_size*fs//2-int(shift_size*fs) : fs*T//2+wind_size*fs//2-int(shift_size*fs)] * nd.array(temp_window)
    template_block = template_block.expand_dims(2)
    if debug:
        logger.debug('Template_block loaded: {}', template_block.shape)  # (35, C, 1, wind_size*fs) cpu nd.ndarray
    if shift_size:
        assert nd.sum(template_block[:,0,0].argmax(-1)).asscalar() / template_block.shape[0]  == int(wind_size*fs*0.8) # Check H1's peak position
    else:
        assert nd.sum(template_block[:,0,0].argmax(-1)).asscalar() / template_block.shape[0]  == wind_size*fs//2 # Check H1's peak position

    RP = RandomPeakAug(margin=margin, T=T, fs = fs, C = C, ori_peak=None, rand_jitter=1)
    
    return dataset_GW, template_block, RP, keys, fs, T, C, margin, wind_size
    


def preDataset1(fs, T, C, shift_size, wind_size, margin,debug = True,TemplateOnly=False):
    temp_window = tukey(fs*wind_size, alpha=1./8)
    mark = 1 if TemplateOnly else 2
    
    dataset_GW = {}
    keys = {}
    for pre in ['train', 'test'][:mark]:
        if T == 1:
            data = np.load('data/GWaveform/data0.90.9_{}.npy'.format(pre))[:,1]
            dataset_GW[pre] = nd.array(data)[:,:C,::4]  # (1610,C,T*fs) cpu nd.ndarray
            keys[pre] = np.load('data/GWaveform/data0.90.9_keys_{}.npy'.format(pre))
        else:
            data = np.load('data/GWaveform/data_T{}_fs4096_{}{}_{}.npy'.format(T,T*0.9,T*0.9, pre))[:,1] # drop GPS
            dataset_GW[pre] = nd.array(data)[:,:C]  # (1610,C,T*fs) cpu nd.ndarray
            keys[pre] = np.load('data/GWaveform/data_T{}_fs4096_{}{}_keys_{}.npy'.format(T, T*0.9,T*0.9,pre))
        if debug:
            logger.debug('Loading {} data: {}', pre, dataset_GW[pre].shape)
        keys[pre] = pd.Series(keys[pre][:,0])  # pd.DataFrame
    assert dataset_GW['train'].shape[0] == dataset_GW['test'].shape[0]
    assert not np.allclose(dataset_GW['train'].asnumpy(), dataset_GW['test'].asnumpy(), atol=1e-21)

    # use equal training masses as template
    equalmass_index = keys['train'][keys['train'].map(lambda x: x.split('|')[0]==x.split('|')[1])].index.tolist()
    nonequalmass_index = keys['train'][keys['train'].map(lambda x: x.split('|')[0]!=x.split('|')[1])].index.tolist()
    template_block = dataset_GW['train'][equalmass_index]  # 35x1x4096 cpu nd.ndarray

    # use equal chi masses as template
    # template_block = np.load('data/GWaveform/template_data_T{}_fs4096_{}{}_train.npy'.format(T, T*0.9,T*0.9))[:,1]
    # template_block = nd.array(template_block)[:,:1]
    # keys_template = np.load('data/GWaveform/template_data_T{}_fs4096_{}{}_keys_train.npy'.format(T, T*0.9,T*0.9))

    # Move the template peak to center corresponding H1
    d = int(template_block[:,0].argmax(-1)[0].asscalar() - fs*T//2)
    template_block = nd.concat(template_block[:,:,d:], nd.zeros(template_block.shape[:2]+(d,)) , dim=2)[:,:,fs*T//2-wind_size*fs//2-int(shift_size*fs) : fs*T//2+wind_size*fs//2-int(shift_size*fs)] * nd.array(temp_window)
    template_block = template_block.expand_dims(2)
    if debug:
        logger.debug('Template_block loaded: {}', template_block.shape)  # (35, C, 1, wind_size*fs) cpu nd.ndarray
    if shift_size:
        assert nd.sum(template_block[:,0,0].argmax(-1)).asscalar() / template_block.shape[0]  == int(wind_size*fs*0.8) # Check H1's peak position
    else:
        assert nd.sum(template_block[:,0,0].argmax(-1)).asscalar() / template_block.shape[0]  == wind_size*fs//2 # Check H1's peak position

    RP = RandomPeakAug(margin=margin, T=T, fs = fs, C = C, ori_peak=None, rand_jitter=1)
    
    if debug and (not TemplateOnly):
        noise, _ = Gen_noise(fs, T, C)   # (4096, C, fs*T)  cpu ndarray
        logger.debug('Noise from [Gen_noise()]: {}', noise.shape)
    
    return dataset_GW, template_block, RP, keys, fs, T, C, margin, wind_size
    
def Gen_noise(fs, T, C, fixed=None):
    tNoiseKEY = 'Event'
    noise_address = os.path.join('./', 'data', 'LIGO_O1_noise_ndarray')
    root = os.path.expanduser(noise_address)
    ll = [ file for file in os.listdir(root) if '_bug' not in file if tNoiseKEY in file]
    
    if fixed:
        r = 2
        noise = nd.concatenate( [readnpy(noise_address, file)[1][:,:C,::4] for file in ll[r:r+T] ] , axis=0 ).astype('float32') # (4096*T, C, 4096)
        noise_gps = nd.concatenate( [readnpy(noise_address, file)[0,:,:1,::4] for file in ll[r:r+T] ] , axis=0 ).astype('float32') # (4096*T, C, 4096)
        noise = noise.swapaxes(1,0).reshape(0,-1,fs*T).swapaxes(1,0)
        noise_gps = noise_gps.swapaxes(1,0).reshape(0,-1,fs*T).swapaxes(1,0)
        noise = nd.concatenate( [noise, noise_gps] , axis=1)
        return noise[:,:2], (ll[r:r+T], noise_gps.asnumpy())
    
    r = np.random.randint(len(ll)-T)
    noise = nd.concatenate( [readnpy(noise_address, file)[1][:,:C,::4] for file in ll[r:r+T] ] , axis=0 ).astype('float32') # (4096*T, C, 4096)
    noise_gps = nd.concatenate( [readnpy(noise_address, file)[0,:,:1,::4] for file in ll[r:r+T] ] , axis=0 ).astype('float32') # (4096*T, C, 4096)
    noise = noise.swapaxes(1,0).reshape(0,-1,fs*T).swapaxes(1,0)
    noise_gps = noise_gps.swapaxes(1,0).reshape(0,-1,fs*T).swapaxes(1,0)
    noise = nd.concatenate( [noise, noise_gps] , axis=1)
    noise = nd.shuffle(noise)
    if T != 1:
        return noise[:,:2], (ll[r:r+T], noise[:,2].asnumpy())
    return noise, (ll[r:r+T], noise[:,2,0].asnumpy())  #  (nsample, C, N)

    
def readnpy(address, file):
    address = os.path.join(address, file)
    address = os.path.expanduser(address)
    return nd.load(address)[0]

def getchiM(keys_template):
    m1_list = np.array(keys_template.map(lambda x: float(x.split('|')[0]) ))
    m2_list = np.array(keys_template.map(lambda x: float(x.split('|')[1]) ))

    M_list = m1_list+m2_list
    return np.power( np.divide( np.power(m1_list * m2_list, 3) , M_list) , 1/5)

def getMratio(keys_template):
    m1_list = np.array(keys_template.map(lambda x: float(x.split('|')[0]) ))
    m2_list = np.array(keys_template.map(lambda x: float(x.split('|')[1]) ))

    return m2_list/m1_list

def preDataset2(SNR, data, batch_size, shuffle=True, fixed = None, debug = True):

    dataset, _, RP, keys, fs, T, C, margin, _ = data
    # Window function
    dwindow = tukey(fs*T, alpha=1./8)
    
    data_block, label_block, chiMkeys_block, Mratiokeys_block, datasets, iterator = {}, {}, {}, {}, {}, {}

    for pre in ['train', 'test']:
        data_block[pre] = RP(dataset[pre]) # (nsample, C, T*fs)
        # data_block[pre] = nd.concat(RP(dataset[pre]), RP(dataset[pre]), dim=0)  # 3150x1x4096  cpu nd.array
        if margin != 0.5:  # global
            assert nd.sum(nd.abs(data_block[pre].argmax(-1) - fs//2) > fs/10).asscalar() == 0  # Check the peaks
        nsample = data_block[pre].shape[0]

        noise, noise_m_gps = Gen_noise(fs, T, C, fixed=fixed)   # (4096, C, fs*T)  cpu ndarray

        sigma = data_block[pre].max(axis=-1) / SNR / nd_std(noise[:nsample], axis=-1)
        signal = nd.divide( data_block[pre] , sigma[:,0].reshape((nsample, 1 ,1))) # taking H1 as leading 
        data_block[pre] = signal + noise[:nsample] # (nsample, C, T*fs)
        if fixed:
            noise_m, noise_p_gps = noise, noise_m_gps
        else:
            noise_m, noise_p_gps = Gen_noise(fs, T, C, fixed=fixed)   # (4096, C, fs*T)  cpu ndarray
        data_block[pre] = nd.concat(data_block[pre], noise_m[:nsample], dim=0)  # (nsample, 1, T*fs) cpu nd.array
        # (nsample, C, T*fs)

        # Note: use mixed data to gen PSD 
        spsd_block_channel = []
        for c in range(C):
            spsd_block = np.concatenate([np.real(np.fft.ifft(1/np.sqrt(power_vec(i[c].asnumpy(), fs)))).reshape(1,-1) for i in data_block[pre]])
            # (nsample, T*fs)  np.array
            spsd_block_channel.append(nd.array(spsd_block).expand_dims(1).expand_dims(1) )# (nsample, 1, 1, T*fs) nd.array cpu
        spsd_block = nd.concatenate(spsd_block_channel, axis=1) # (nsample, C, 1, T*fs)
        if debug:
            logger.debug('spsd_block for {}: {}', pre,spsd_block.shape)

        # data * dwindow
        data_block[pre] = (data_block[pre]* nd.array(dwindow)).expand_dims(2) # (nsample, C, 1, T*fs) nd.array cpu
        if debug:
            logger.debug('data_block for {}: {}', pre, data_block[pre].shape)

        data_block[pre] = nd.concat(data_block[pre].expand_dims(1), spsd_block.expand_dims(1), dim=1)
        if debug:
            logger.debug('data_block(psd,nd) for {}: {}', pre, data_block[pre].shape)  # (nsmaple, 2, C, 1, T*fs) cpu nd.array

        label_block[pre] = nd.array([1]*nsample + [0]*nsample)

        chiMkeys_block[pre] = nd.array(getchiM(keys[pre]).tolist() + [0]*nsample)
        Mratiokeys_block[pre] = nd.array(getMratio(keys[pre]).tolist() + [0]*nsample)
    
        datasets[pre] = gluon.data.ArrayDataset(data_block[pre], label_block[pre], chiMkeys_block[pre], Mratiokeys_block[pre])
        iterator[pre] = gdata.DataLoader(datasets[pre], batch_size, shuffle=shuffle, last_batch = 'keep', num_workers=0)
        if debug:
            logger.debug('\nNoise from: {} | {}', noise_m_gps[0], noise_p_gps[0])

    return dataset, iterator, (noise_m_gps[0]+noise_p_gps[0], np.concatenate((noise_m_gps[1][:nsample], noise_p_gps[1][:nsample]),axis=0))


def preEventsDataset(fs, T, C):
    deltat = T/2#1#2.5#3#5

    onesecslice = [(65232, 69327) , (65178, 69273), 
    (66142, 70237), (66134, 70229),
    (65902, 69997), (65928, 70023), 
    (65281, 69376), (65294, 69389)]
    llLIGOevents = [file for file in os.listdir('Data_LIGO_Totural') if 'strain' in file]
    llLIGOevents.sort()
    aroundEvents = np.concatenate([np.load('./Data_LIGO_Totural/'+file).reshape(1,-1)[:,onesecslice[index][0]-int((deltat-0.5)*fs):onesecslice[index][1]+int((deltat-0.5)*fs)+1] \
                                    for index, file in enumerate(llLIGOevents)])
    logger.debug('Loaded aroundEvents: {}', aroundEvents.shape)  # (8, T*fs)
    logger.debug('Loaded list of Events: \n{}', np.array(llLIGOevents))

    aroundEvents = nd.array(aroundEvents).expand_dims(1)

    if C == 1:    
        aroundEvent_psd_block = nd.concatenate([EquapEvent(fs, data) for data in aroundEvents], axis=0)
    elif C == 2:
        aroundEvent_psd_block_H1 = nd.concatenate([EquapEvent(fs, data) for data in aroundEvents[::2]], axis=0) # (4, 2, 1, 102400)
        aroundEvent_psd_block_L1 = nd.concatenate([EquapEvent(fs, data) for data in aroundEvents[1::2]], axis=0) # (4, 2, 1, 102400)
        aroundEvent_psd_block = nd.concat(aroundEvent_psd_block_H1.swapaxes(1,0).expand_dims(2), 
                                        aroundEvent_psd_block_L1.swapaxes(1,0).expand_dims(2), dim=2 ).swapaxes(1,0)
    logger.debug('aroundEvent_psd_block: {}', aroundEvent_psd_block.shape)
    return aroundEvent_psd_block


def preNeuralNet(fs, T, ctx, template_block, margin, learning_rate=0.003):
    net = gluon.nn.Sequential()         
    with net.name_scope():           # Used to disambiguate saving and loading net parameters
        net.add(MatchedFilteringLayer(mod=fs*T, fs=fs,
                                    template_H1=template_block[:,:1],#.as_in_context(ctx),
                                    template_L1=template_block[:,-1:]#.as_in_context(ctx) 
                                    ))
        net.add(CutHybridLayer(margin = margin))
        net.add(Conv2D(channels=16, kernel_size=(1, 3), activation='relu'))
        net.add(MaxPool2D(pool_size=(1, 4), strides=2))
        net.add(Conv2D(channels=32, kernel_size=(1, 3), activation='relu'))    
        net.add(MaxPool2D(pool_size=(1, 4), strides=2))
        net.add(Flatten())
        net.add(Dense(32))
        net.add(Activation('relu'))
        net.add(Dense(2))

    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx[-1], force_reinit=True)     # Initialize parameters of all layers
    net.summary(nd.random.randn(1,2,2,1,fs*T, ctx=ctx[-1]))
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)     # Initialize parameters of all layers
    # 交叉熵损失函数 
    # loss = gloss.SoftmaxCrossEntropyLoss()
    # The cross-entropy loss for binary classification.
    bloss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})
    return net, bloss, trainer

def Batch_Training(net, bloss, trainer, iterator, aroundEvent_psd_block, calEvents, checkpoint_everybatch = False, Deadtime = False):

    # Set schedule.
    if time.strftime('%H:%M:%S') >= Deadtime:
        logger.warning('> Deadtime ({})'.format(Deadtime))
        raise KeyboardInterrupt

    curr_loss, curr_loss_list = 0, []
    curr_loss_v, curr_loss_list_v = 0, []
    train_l_sum, train_acc_sum, n = .0, .0, .0
    test_l_sum, test_acc_sum = .0, .0
    predEvents_list = []

    assert len(iterator['train']) == len(iterator['test'])
    num = len(iterator['train'])

    for index, [(data, label, _, _), (data_v, label_v, _, _)] in enumerate(zip(iterator['train'], iterator['test'])):
        gpu_Xs, gpu_ys = gutils.split_and_load(data, ctx), gutils.split_and_load(label, ctx)
        gpu_Xs_v, gpu_ys_v = gutils.split_and_load(data_v, ctx), gutils.split_and_load(label_v, ctx)
        # assert not np.allclose(label.asnumpy() , label_v.asnumpy())

        ######################## Training ###########################
        with autograd.record():  # 在各块GPU上分别计算损失
            gpu_y_hats = [net(gpu_X) for gpu_X in gpu_Xs]
            ls = [bloss(gpu_y_hat, nd.one_hot(gpu_y, 2)) for gpu_y_hat, gpu_y in zip(gpu_y_hats, gpu_ys) ]
        for l in ls:  # 在各块GPU上分别反向传播
            l.backward()
            curr_loss_list.append(nd.mean(l).asscalar())
        trainer.step(batch_size)
        curr_loss = np.mean(curr_loss_list)

        # # Training moving loss/acc.
        train_acc_sum += sum([(gpu_y_hat.argmax(axis=1) == y).sum().asscalar() for gpu_y_hat, y in zip(gpu_y_hats, gpu_ys)])
        train_l_sum += sum([l.sum().asscalar() for l in ls])        
        n += label.size

        # Validation
        gpu_y_hats_v = [net(gpu_X_v) for gpu_X_v in gpu_Xs_v]
        ls_v = [bloss(gpu_y_hat_v, nd.one_hot(gpu_y_v, 2)) for gpu_y_hat_v, gpu_y_v in zip(gpu_y_hats_v, gpu_ys_v) ]
        for l in ls_v:
            curr_loss_list_v.append(nd.mean(l).asscalar())
        curr_loss_v = np.mean(curr_loss_list)

        # # Training moving loss/acc.
        test_acc_sum += sum([(gpu_y_hat_v.argmax(axis=1) == y_v).sum().asscalar() for gpu_y_hat_v, y_v in zip(gpu_y_hats_v, gpu_ys_v)])
        test_l_sum += sum([l.sum().asscalar() for l in ls_v])        

        # Calculate at Events
        if calEvents:
            predEvents = net(aroundEvent_psd_block.as_in_context(ctx[-1])) # (4,2,1,35)
            predEvents = nd.softmax(predEvents)[:,1].asnumpy().tolist()

            logger.debug('ls:{:.5f}, lsv:{:.5f}, ({:.3f}|{:.3f}|{:.3f}|{:.3f}), ({:.2f}%)', 
                                            curr_loss, curr_loss_v, 
                                            predEvents[0],predEvents[1],
                                            predEvents[2],predEvents[3], (index+1)/num*100 )
            predEvents_list.append(predEvents)
        else:
            logger.debug('ls:{:.5f}, lsv:{:.5f}, ({:.2f}%)', curr_loss, curr_loss_v, (index+1)/num*100 )            
        ######################## Training ###########################

        #### checkpoint_everybatch ###############################
        if checkpoint_everybatch:
            try:
                assert type(checkpoint_everybatch) == type('')
            except:
                checkpoint_everybatch = input('Input address for checkpoint_everybatch (with / at final):') # str
                mkdir(checkpoint_everybatch)
            net.save_parameters(checkpoint_everybatch + 'Everybatch{}_lsv{:.3f}.params'.format(index+1, curr_loss_v))

    nd.waitall()
    if not calEvents:
        predEvents = net(aroundEvent_psd_block.as_in_context(ctx[-1])) # (4,2,1,35)
        predEvents = nd.softmax(predEvents)[:,1].asnumpy().tolist()
        predEvents_list.append(predEvents)  # (~, 4)
    return train_l_sum/n, train_acc_sum/n, test_l_sum/n, test_acc_sum/n, (curr_loss_list, curr_loss_list_v, predEvents_list)


def Epoch_Training(SNR, predata, net, batch_size, Epoch, bloss, trainer, aroundEvent_psd_block, calEvents = None, checkpoint_everybatch = False, Deadtime = False, tf_epoch = 0):
    
    _, _, _, _, fs, T, C, _, wind_size = predata
    
    Eacc_list, Ecurr_loss_list, Ecurr_loss_list_v, EpredEvents_list = [], [], [], []
    best_test_acc = 0.

    for index in range(tf_epoch, Epoch): # global Epoch
        start = time.time()
        _, iterator, _ = preDataset2(SNR = SNR, data=predata, batch_size=batch_size, shuffle=True, fixed=False, debug = True)
        # in which #  _, _ = dataset, cache_noise_dataset        

        # Batch_Training
        try:
            history = Batch_Training(net, bloss, trainer, iterator, aroundEvent_psd_block, calEvents=calEvents, checkpoint_everybatch = checkpoint_everybatch, Deadtime = Deadtime)
        except KeyboardInterrupt as e:
            logger.warning('KeyboardInterrupt for SNR{}'.format(SNR))
            break
        train_l, train_acc, test_l, test_acc, (curr_loss_list, curr_loss_list_v, predEvents_list) = history

        # Collect the history
        EpredEvents_list.extend(predEvents_list) # (~+~, 4)
        Ecurr_loss_list.append(curr_loss_list) # (~, 202)
        Ecurr_loss_list_v.append(curr_loss_list_v) # (~, 202)
        Eacc_list.append([train_acc, test_acc]) # (~, 2)
        logger.debug('Shape of history: {}|{}|{}|{}', np.array(EpredEvents_list).shape, 
                                                      np.array(Ecurr_loss_list).shape,
                                                      np.array(Ecurr_loss_list_v).shape,
                                                      np.array(Eacc_list).shape )

        # Cache the best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = index+1
            best_model_name = checkpoint_name.format(SNR, fs, T, wind_size, best_epoch, best_test_acc)

            # at last: save the best model to HDD
            net.save_parameters(checkpoint_address+best_model_name)

            # Stdout
            logger.success('[E.{}|SNR.{}] train ls. {:.4f}, acc. {:.3f}, test ls. {:.4f}, acc. {:.3f}, ({:.3f}|{:.3f}|{:.3f}|{:.3f})({:.1f}s)', 
                            index+1, SNR, train_l, train_acc, test_l, test_acc, 
                            EpredEvents_list[-1][0],EpredEvents_list[-1][1],
                            EpredEvents_list[-1][2],EpredEvents_list[-1][3], time.time()-start)
        else:
            # Stdout
            logger.info('[E.{}|SNR.{}] train ls. {:.4f}, acc. {:.3f}, test ls. {:.4f}, acc. {:.3f}, ({:.3f}|{:.3f}|{:.3f}|{:.3f})({:.1f}s)', 
                            index+1, SNR, train_l, train_acc, test_l, test_acc, 
                            predEvents_list[-1][0],predEvents_list[-1][1],
                            predEvents_list[-1][2],predEvents_list[-1][3], time.time()-start)            

        # Terminate the epoch loop
        if (best_epoch >= 10) and (best_epoch < (index +1 )/2):
            logger.warning('Terminate the epoch loop!')
            return Eacc_list, Ecurr_loss_list, Ecurr_loss_list_v, EpredEvents_list, best_model_name

    return Eacc_list, Ecurr_loss_list, Ecurr_loss_list_v, EpredEvents_list, best_model_name

def SNR_Training(SNR_list, predata, net, batch_size, Epoch, bloss, trainer, aroundEvent_psd_block, calEvents, checkpoint_address, checkpoint_everybatch = False, Deadtime = False, tf_epoch=0):
    _, _, _, _, fs, T, C, _, wind_size = predata
    # Training
    for SNR in SNR_list:
        logger.success('SNR: {}', SNR)        
        
        history = Epoch_Training(SNR, predata, net, batch_size, Epoch, bloss, trainer, aroundEvent_psd_block, calEvents, checkpoint_everybatch, Deadtime, tf_epoch)
        Eacc_list, Ecurr_loss_list, Ecurr_loss_list_v, EpredEvents_list, best_model_name = history

        # History Saving
        np.save(checkpoint_address + 'SNR{}_fs{}_T{}w{}_Eacc_list'.format(SNR, fs, T, wind_size), Eacc_list)
        np.save(checkpoint_address + 'SNR{}_fs{}_T{}w{}_Ecurr_loss_list'.format(SNR, fs, T, wind_size), Ecurr_loss_list)
        np.save(checkpoint_address + 'SNR{}_fs{}_T{}w{}_Ecurr_loss_list_v'.format(SNR, fs, T, wind_size), Ecurr_loss_list_v)
        np.save(checkpoint_address + 'SNR{}_fs{}_T{}w{}_EpredEvents_list'.format(SNR, fs, T, wind_size), EpredEvents_list)
        logger.debug('History Saving at SNR{}'.format(SNR))

        # Transfering
        net.load_parameters(checkpoint_address+best_model_name)
        logger.debug('Transfering best model from SNR{}'.format(SNR))

if __name__ == '__main__': 
    ##################################################
    ######## Global Variables Start ##################
    fs = 4096
    T = 5
    wind_size = 1
    shift_size = True
    if shift_size:
        shift_size = wind_size * 0.8 - wind_size/2
        assert T - wind_size > shift_size
        assert wind_size > shift_size
    margin = 0.5 # %
    C = 2
    ctx = [mx.gpu(0), mx.gpu(3)]

    learning_rate = 0.003
    batch_size = 16
    Epoch = 30

    calEvents = True        # During batch_training | Default False
    checkpoint_address = './data/checkpointing_MF4MXNet_randomNoise/'
    checkpoint_name = "SNR{}-fs{}-T{}w{}-{:02d}-{:.4f}.params"
    checkpoint_everybatch = False   # Default False; True only for Epoch = 1 and one SNR
    Deadtime = '07:00:00'   # Format: %H:%M:%S

    mkdir(checkpoint_address)
    SNR_list = [1, 0.1, 
                0.03, 
                0.02, 0.019]

    # checkpoint everybatch only for Epoch = 1 and one SNR
    if checkpoint_everybatch:
        assert Epoch == len(SNR_list) == 1

    # global RP, template_block, dataset, keys
    ######## Global Variables End #####################
    ###################################################

    
    # Prepare datasets
    predata = preDataset1(fs, T, C, shift_size, wind_size, margin, debug = True)
    aroundEvent_psd_block = preEventsDataset(fs, T, C)
    _, template_block, _, _,_,_,_,_,_  = predata

    # Define network
    net, bloss, trainer = preNeuralNet(fs, T, ctx, template_block, margin, learning_rate)

    ##########
    tf_SNR = 0.1
    tf_epoch = 14
    net.load_parameters(checkpoint_address+'SNR{}-fs4096-T5w1-{:02d}-0.9994.params'.format(tf_SNR, tf_epoch))
    ##########

    SNR_Training(SNR_list, predata, net, batch_size, Epoch, bloss, trainer, aroundEvent_psd_block, calEvents, checkpoint_address, checkpoint_everybatch, Deadtime, tf_epoch)

