import numpy as np
import os
from configargparse import ArgParser
import torch
from SMAPdatacode import generatedataSMAP

def get_data_and_lable(SMAPdata,seq_length,lead_time):
    rawdata = SMAPdata
    data = []
    label = []
    for i in range(0, rawdata.shape[0] - seq_length-lead_time):
        tmp_data = rawdata[i:i + seq_length]
        tmp_label = rawdata[i + seq_length + lead_time,rawdata.shape[1]-1,:]
        data.append(tmp_data)
        label.append(tmp_label)
    data = np.array(data)
    label = np.array(label)
    return data, label

def NormalizationData(data):
    scalar_set=np.zeros((data.shape[1],2)).astype(np.float32)
    for i in range(0, data.shape[1]):
        re = data[:,i,:,:]
        ##NormalizationData
        re = re.reshape(data.shape[0],-1)
        scalar_set[i,0]=re.min()
        scalar_set[i, 1] = re.max()
        re = (re-re.min())/(re.max()-re.min())
        data[:, i, :, :] = re.reshape(data[:,i,:,:].shape)
    return data, scalar_set

def NormalizationTest(data,scalar_set):
    for i in range(0, data.shape[2]):
        re = data[:,:,i,:,:]
        ##NormalizationData
        re = re.reshape(data.shape[0], -1)
        scalar_min=scalar_set[i,0]
        scalar_max=scalar_set[i, 1]
        re = (re - scalar_min) / (scalar_max - scalar_min)
        data[:, :,i, :, :] = re.reshape(data[:,:,i,:,:].shape)
    return data

def main(datadir,seq_length,lead_time):


    # TODO: Flexible input data
    SMAPdata=generatedataSMAP(datadir)
    print('finished generating SMAP data')
    # TODO: split datasets into training dataset and testing dataset
    ####training dataset
    data_train = SMAPdata[:int( SMAPdata.shape[0])-365-seq_length-lead_time]
    ####testing dataset
    data_test = SMAPdata[int(SMAPdata.shape[0])-365-seq_length-lead_time:int(1.0 * SMAPdata.shape[0])]
    print('finished splitting datasets into training dataset and testing dataset')
    # TODO: Normalization
    ####training dataset
    data_train,scalar_set=NormalizationData(data_train)
    print('finished Normalization')
    # TODO: split data into features and observed soil moisture
    features_train, label_train = get_data_and_lable(data_train, seq_length, lead_time)
    print('finished splitting training  data into feature data and observed soil moisture')
    features_test, label_test = get_data_and_lable(data_test, seq_length, lead_time)
    # TODO: Normalization
    ####testing dataset
    features_test=NormalizationTest(features_test,scalar_set)
    # TODO: transform data for LSTM model
    print('features_train.shape',features_train.shape)
    print('label_train.shape', label_train.shape)
    print('features_test',features_test.shape)
    print('label_test', label_test.shape)



    dir = r"./data-ConLSTM-model"
    if not os.path.exists(dir):
        os.mkdir(dir)
    np.save('./data-ConLSTM-model/features_train.npy', features_train)
    np.save('./data-ConLSTM-model/label_train.npy', label_train)
    np.save('./data-ConLSTM-model/features_test.npy', features_test)
    np.save('./data-ConLSTM-model/label_test.npy', label_test)
    np.save('./data-ConLSTM-model/scalar_set.npy', scalar_set)
    print('finished saving  data for conv model')


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--datadir', type=str, default='../../../ERA5npydata/', help='Path to data')
    p.add_argument('--seq_length', type=int, default=3, help='input timesteps for lstm model')
    p.add_argument('--lead_time', type=int, default=3, help='Forecast lead time')
    args = p.parse_args()

    main(
        datadir=args.datadir,
        seq_length=args.seq_length,
        lead_time=args.lead_time
    )

