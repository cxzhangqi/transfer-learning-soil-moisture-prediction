import numpy as np
from configargparse import ArgParser
import torch
import torch.utils.data as Data
from torch.optim import lr_scheduler
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
from torch.autograd import Variable
from convlstm import ConvLSTM
import torch.nn as nn

import torch.nn.functional as F






class LR(torch.nn.Module):
    def __init__(self, channel_number, output_size):
        """
        :param size:一日数据
        :param day: n日
        :param hidden_dim:隐藏层神经元
        :param layer_dim: 隐藏层个数
        :param output_dim: 输出
        """
        super(LR, self).__init__()
        self.output_size=output_size
        self.channel_number=channel_number
        self.dense = torch.nn.Linear(in_features=channel_number*output_size, out_features=output_size)

    def forward(self, x):
        x = x.reshape(-1,self.channel_number*self.output_size)
        x =F.relu(x)
        output = self.dense(x)
        return output







def train_ConvLSTM(ConvLSTM_net,LR_net,lr,train_loader,total_epoch,channel_number):
    loss_func = torch.nn.MSELoss()
    threshold = torch.nn.Threshold(0., 0.0)
    params = list(ConvLSTM_net.parameters())+list(LR_net.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    s = 1
    global_step = 1
    loss_metrics = AverageValueMeter()
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(0, 500, 150)][1:], gamma=0.05)
    ########## training set##########
    loadData = np.load('../../A.npy')
    loadData = torch.tensor(loadData.reshape(loadData.shape[0] , loadData.shape[1]),
                            dtype=torch.float32).cuda()
    for epoch in range(total_epoch):
        epoch_loss = 0
        for step, (x, y) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            x = x.view(x.shape[1], x.shape[0], x.shape[2], x.shape[3], x.shape[4])
            y = y.view(y.shape[0], y.shape[1], y.shape[2])
            # Encoder
            hidden = ConvLSTM_net.get_init_states(x.shape[1])
            last_state, encoder_state = ConvLSTM_net(x.clone(), hidden)
            last_state=last_state.permute(1, 0, 2, 3, 4)

            shape=last_state.shape[0]
            last_state=last_state.reshape(shape,channel_number,52,65)

            cut = threshold(last_state)

            cut=LR_net(cut)

            cut=cut.reshape(shape, 52, 65)

            cut=cut*loadData
            y  =y*loadData
            train_loss = loss_func(cut, y)
            train_loss.backward()
            optimizer.step()
            global_step = global_step + 1
            epoch_loss += train_loss.item()
            loss_metrics.add(train_loss.item())

        print("[epcho {}]:loss {}".format(epoch, loss_metrics.value()[0]))
        loss_metrics.reset()
        scheduler.step()
    return ConvLSTM_net




def main(lr,total_epoch,ConvLSTM_kernel_size,batch_size,ConvLSTM_num_layers1,ConvLSTM_num_layers2):
    features_train=np.load('./data-ConLSTM-model/features_train.npy')
    label_train = np.load('./data-ConLSTM-model/label_train.npy')
    data_train = np.load('./data-ConLSTM-model/data_train.npy')

    print('finished loading  data for ConLSTM model')

    # TODO: transform data for ConLstm  model
    features_train = torch.tensor(features_train.reshape
                                  (-1, features_train.shape[1], features_train.shape[2], features_train.shape[3],
                                   features_train.shape[4])
                                  , dtype=torch.float32).cuda()
    label_train = torch.tensor(label_train.reshape
                               (-1, label_train.shape[1], label_train.shape[2])
                               , dtype=torch.float32).cuda()

    dataset = Data.TensorDataset(features_train, label_train)
    train_loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    print('finished transforming  data for Con-LSTM model')
    # TODO: build ConvLSTM model
    ConvLSTM_net = ConvLSTM(input_size=(data_train.shape[2], data_train.shape[3]),
                       input_dim=data_train.shape[1],
                       hidden_dim=[ConvLSTM_num_layers1, ConvLSTM_num_layers2],
                       kernel_size=(ConvLSTM_kernel_size, ConvLSTM_kernel_size),
                       num_layers=2,
                       )


    ConvLSTM_net.cuda()

    LR_net = LR(channel_number=ConvLSTM_num_layers2, output_size=features_train.shape[3]*features_train.shape[4])
    LR_net.cuda()
    # TODO: train ConvLSTM model
    train_ConvLSTM(ConvLSTM_net,LR_net,lr,train_loader,total_epoch,channel_number=ConvLSTM_num_layers2)
    print('finished training Con-LSTM model')

    torch.save(ConvLSTM_net.state_dict(), './data-ConLSTM-model/ConvLSTM_net_params.pth')
    torch.save(LR_net.state_dict(), './data-ConLSTM-model/LR_net_params.pth')


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--total_epoch', type=int, default=30, help='total epochs for training the model')
    p.add_argument('--ConvLSTM_kernel_size', type=int, default=3, help='kernel size for ConvLSTM model')
    p.add_argument('--ConvLSTM_num_layers1', type=int, default=8, help='numbers of layer for ConvLSTM model')
    p.add_argument('--ConvLSTM_num_layers2', type=int, default=4, help='numbers of layer for ConvLSTM model')
    p.add_argument('--batch_size', type=int, default=64, help='batch_size')
    args = p.parse_args()

    main(
        lr=args.lr,
        total_epoch=args.total_epoch,
        ConvLSTM_kernel_size=args.ConvLSTM_kernel_size,
        ConvLSTM_num_layers1=args.ConvLSTM_num_layers1,
        ConvLSTM_num_layers2=args.ConvLSTM_num_layers2,
        batch_size=args.batch_size
    )

