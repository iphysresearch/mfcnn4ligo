#! usr/bin/python
# coding=utf-8

import sys,os,datetime
from tqdm import tqdm
import readligo as rl
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import data as gdata
from script_MF4MXNet import EquapEvent, preTemplateFloyd, preNeuralNet
from loguru import logger
config = {
    "handlers": [
        {"sink": "MF4MXNet_{}.log".format(datetime.date.today()), "level":"DEBUG" ,"format": '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>'},
        {"sink": sys.stdout, "format": '<green>{time:YYYY-MM-DD}</green> <cyan>{time:HH:mm:ss}</cyan> | <level>{level: <7}</level> | <level>{message}</level>',
        "level": "DEBUG"},
    ],
}
#### REF #### https://loguru.readthedocs.io/en/stable/api/logger.html
# DEBUG 10  # INFO 20  # WARNING 30  # ERROR 40  # CRITICAL 50
logger.configure(**config)


def LoadLIGO(data_address, GPS):

    H1_list = [ file for file in os.listdir(data_address) if 'H1' in file if str(GPS) in file]
    L1_list = [ file for file in os.listdir(data_address) if 'L1' in file if str(GPS) in file]
    H1_list.sort()
    L1_list.sort()
    assert len(H1_list) == len(L1_list) == 1  # Same length for H1/L1 file
    # Same GPS for sorted H1/L1 files list:
    assert len([i for i, H1 in enumerate(H1_list) if H1.split('-')[2] != L1_list[i].split('-')[2]]) == 0

    # Loading data
    strain_H1, time_H1, chan_dict_H1 = rl.loaddata(os.path.join(data_address, H1_list[0]), "H1")
    strain_L1, time_L1, chan_dict_L1 = rl.loaddata(os.path.join(data_address, L1_list[0]), "L1")

    assert np.allclose(time_L1, time_H1, atol=0, rtol=0) # Check GPS time for H1/L1
    GPStime = time_H1

    logger.debug('Loaded strain_H1/GPStime/chan_dict_H1 shape: {}/{}/{}', strain_H1.shape, GPStime.shape, len(chan_dict_H1))
    logger.debug('Loaded strain_L1/GPStime/chan_dict_L1 shape: {}/{}/{}', strain_L1.shape, GPStime.shape, len(chan_dict_L1))
    # Until here may costs 2.1 s
    return GPStime, strain_H1, chan_dict_H1, strain_L1, chan_dict_L1

def iteration_LIGOStrain(T, fs, LIGOdata, step, batch_size = 8):
    '''
    Input Strain.
    '''
    GPStime, strain_H1, chan_dict_H1, strain_L1, chan_dict_L1 = LIGOdata
    assert strain_H1.size == strain_L1.size == GPStime.size
    lentime = int(GPStime.size/fs)  # 4096 length of strain time
    bolnotnan = ~(np.isnan(strain_H1) + np.isnan(strain_L1))

    isvalid = lambda a, b: not False in bolnotnan[a:b]
    Slice = [( i*int(step*fs), (i*int(step*fs)+fs*T) ) for i in tqdm(range(lentime*fs), desc = 'Generate slicing windows ({}/{})'.format(strain_H1[bolnotnan].size/fs, lentime)) if isvalid(i*int(step*fs), (i*int(step*fs)+fs*T)) if (i*int(step*fs)+fs*T) <= lentime*fs]
    # Slice above may cost 1m11s
    # Check the validation of Slice
    assert [0 for i, j in Slice if False in bolnotnan[i:j] ] == []
    logger.debug('Num of valid slice/steps: {}', len(Slice))
    
    H1_block = nd.array(np.concatenate( [strain_H1[i:j].reshape(1,-1) for (i, j) in Slice] ))
    L1_block = nd.array(np.concatenate( [strain_L1[i:j].reshape(1,-1) for (i, j) in Slice] ))
    logger.debug('H1_block shape: {}; L1_block shape: {}', H1_block.shape, L1_block.shape) # (Steps, T*fs)   

    H1_psd_block = nd.concatenate([ EquapEvent(fs, i) for i in tqdm(H1_block.expand_dims(1), desc = 'Calc. PSD for H1_block') ])
    L1_psd_block = nd.concatenate([ EquapEvent(fs, i) for i in tqdm(L1_block.expand_dims(1), desc = 'Calc. PSD for L1_block') ])
    logger.debug('H1_psd_block shape: {}; L1_psd_block shape: {}', H1_psd_block.shape, L1_psd_block.shape) # (Steps, 2, 1, T*fs)
    # psd_block here may cost 26s

    O1_psd_block = nd.concat(H1_psd_block.expand_dims(2), 
                             L1_psd_block.expand_dims(2), dim=2)
    logger.debug('O1_psd_block shape: {}', O1_psd_block.shape) # # (Steps, 2, 2, 1, T*fs)

    slice2df = lambda strain,i,j: pd.DataFrame(strain).iloc[int(i/fs):int(j/fs)]
    df2dict = lambda df: {col:df[col].values for col in df.columns} 

    chan_dict_H1 = [ df2dict(slice2df(chan_dict_H1, i, j)) for (i,j) in tqdm(Slice, desc='Slicing chan_dict_H1')]
    chan_dict_L1 = [ df2dict(slice2df(chan_dict_L1, i, j)) for (i,j) in tqdm(Slice, desc='Slicing chan_dict_L1')]
    chan_dict_block = np.concatenate(([chan_dict_H1], [chan_dict_L1]), axis=0).T
    logger.debug('chan_dict_block shape: {}', chan_dict_block.shape)
    
    # O1_psd_block (steps, 2, 2, 1, T*fs) nd.array cpu
    # GPStime_block (steps, ) list
    # chan_list_block (steps, 2) np.array([dict], [dict])
    GPStime_block = [GPStime[i + T*fs//2] for (i,_) in Slice]
    dataset = gluon.data.ArrayDataset(O1_psd_block, GPStime_block)
    iterator = gdata.DataLoader(dataset, batch_size=batch_size, shuffle=False, last_batch = 'keep', num_workers=0)    
    
    return dataset, iterator, chan_dict_block



if __name__ == "__main__":


    df_url = pd.read_table('segsurl_O1_4KHZ_0_689.txt', names='U').drop_duplicates()
    df_url['GPS'] = df_url.U.map(lambda x: int(x.split('/')[7:][0].split('-')[2]))

    dfH1 = df_url[df_url.U.map(lambda x: x.split('/')[7:][0].split('-')[0]) == 'H']
    dfH1 = dfH1.sort_values('GPS')
    dfH1.reset_index(inplace=True)

    dfL1 = df_url[df_url.U.map(lambda x: x.split('/')[7:][0].split('-')[0]) == 'L']
    dfL1 = dfL1.sort_values('GPS')
    dfL1.reset_index(inplace=True)

    # Check the corelationship of GPS => Get the GPSlist
    assert (dfH1.GPS - dfL1.GPS).unique() == 0
    GPSlist = dfH1.GPS.tolist()
    H1list = dfH1.U.tolist()
    L1list = dfL1.U.tolist()

    logger.debug('Total blocks: {}', len(GPSlist))
    
    target = 10
    GPSlist = GPSlist[target:target+90]
    L1list = L1list[target:target+90]
    H1list = H1list[target:target+90]

    # Let loop 1297
    for index, (H1url, L1url, GPS) in enumerate(zip(H1list, L1list, GPSlist)):
        print('{{"metric": "GPSlist", "value": {}}}'.format(index))
        print('{{"metric": "GPS", "value": {}}}'.format(GPS))

        logger.debug('#'*40)
        # H1url = 'https://www.gw-openscience.org/archive/data/O1/1125122048/H-H1_LOSC_4_V1-{}-4096.hdf5'.format(GPS)
        # L1url = 'https://www.gw-openscience.org/archive/data/O1/1125122048/L-L1_LOSC_4_V1-{}-4096.hdf5'.format(GPS)
        os.system('wget {} {}'.format(H1url, L1url))

        data_address = os.path.join('./')
        LIGOdata = LoadLIGO(data_address, GPS)

        fs = 4096
        T = 5
        frac = 5
        step = T/frac  # second
        batch_size = 14
        print(step)

        wind_size = 1
        shift_size = True
        if shift_size:
            shift_size = wind_size * 0.8 - wind_size/2
            assert T - wind_size > shift_size
            assert wind_size > shift_size
        margin = 0.5 # percentage %
        C = 2
        ctx = [mx.gpu(0)]


        dataset, iterator, chan_dict_block = iteration_LIGOStrain(T, fs, LIGOdata, step, batch_size = batch_size)

        predata = preTemplateFloyd(fs, T, C, shift_size, wind_size, margin, debug = True)
        _, template_block, _, _,_,_,_,_,_  = predata
        net, _, _ = preNeuralNet(fs, T, ctx, template_block, margin)


        net.load_parameters('./checkpointing_MF4MXNet/'+'SNR0.02-fs4096-T5w1-27-0.9519.params')

        allmidGPStime_list, chan_list, allpred_list = [], [], []

        for index, (data, GPStime) in enumerate(tqdm(iterator, leave=False)):
            oo = net(data.as_in_context(ctx[-1])) # Output MF features
            pred_list = nd.softmax(oo)[:,1].asnumpy().tolist()
            midGPStime_list = GPStime.asnumpy().tolist()

            allpred_list.extend(pred_list)
            allmidGPStime_list.extend(midGPStime_list)
            chan_list.extend(chan_dict_block[index*batch_size:batch_size + index*batch_size].tolist())
        # It may cost 7min or 10min

        save_address = './GPS{}_index_SNR0.02-fs4096-T5w1-27-frac5.output'.format(GPS, index)
        np.save(save_address, [allpred_list, allmidGPStime_list, chan_list])


        os.system('rm -f H-H1_LOSC_4_V1-{}-4096.hdf5'.format(GPS))
        os.system('rm -f L-L1_LOSC_4_V1-{}-4096.hdf5'.format(GPS))
    
    
    [ os.system('rm -rf {}'.format(file)) for file in os.listdir('./') if 'GPS' not in file]