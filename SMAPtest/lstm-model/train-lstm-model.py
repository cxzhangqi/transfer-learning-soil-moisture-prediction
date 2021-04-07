import numpy as np
from configargparse import ArgParser
import torch
import torch.utils.data as Data
from torch.optim import lr_scheduler
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

class myLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, layer_dim=1):
        """
        :param size:一日数据
        :param day: n日
        :param hidden_dim:隐藏层神经元
        :param layer_dim: 隐藏层个数
        :param output_dim: 输出
        """
        super(myLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.dense = torch.nn.Linear(in_features=hidden_dim,out_features=hidden_dim)
    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(1, 0, 2)
        print(1,x.shape)
        output, (w, y) = self.lstm(x)
        output = output.permute(1, 0, 2)
        output = output[:, -1, :]
        output = self.dense(output)
        return output

def train_lstm(net,lr,train_loader,total_epoch):
    global_step = 1
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(0, 500, 150)][1:], gamma=0.05)
    loss_func = torch.nn.MSELoss()
    loss_metrics = AverageValueMeter()
    # with torch.no_grad():
    ########## training set##########
    for epoch in range(total_epoch):
        epoch_loss = 0
        for step, (x, y) in tqdm(enumerate(train_loader)):
            print(x.shape)
            output = net(x)
            print(output.shape)
            ##########加mask训练
            loadData = np.load('../../A.npy')
            loadData = torch.tensor(loadData.reshape(loadData.shape[0] * loadData.shape[1]),
                                    dtype=torch.float32).cuda()
            output = output * loadData
            y = y * loadData
            ##########
            train_loss = loss_func(output, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            global_step = global_step + 1
            epoch_loss += train_loss.item()
            loss_metrics.add(train_loss.item())
        print("[epcho {}]:loss {}".format(epoch, loss_metrics.value()[0]))
        loss_metrics.reset()
        scheduler.step()
    return net




def main(lr,total_epoch,batch_size):
    features_train = np.load('./data-lstm-model/features_train.npy')
    label_train = np.load('./data-lstm-model/label_train.npy')
    print('finished loading  data for LSTM model')


    # TODO: transform data for LSTM model
    IN_FEATURE=features_train.shape[2]*features_train.shape[3]*features_train.shape[4]
    features_train = torch.tensor(features_train.reshape(-1, features_train.shape[1],IN_FEATURE) , dtype=torch.float32).cuda()

    OUT_FEATURE=label_train.shape[1]*label_train.shape[2]
    label_train = torch.tensor(label_train.reshape(-1, OUT_FEATURE), dtype=torch.float32).cuda()


    dataset = Data.TensorDataset(features_train, label_train)
    train_loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    print('finished transforming  data for LSTM model')
    # TODO: build LSTM model
    net = myLSTM(input_size=IN_FEATURE, hidden_dim=OUT_FEATURE, layer_dim=1)
    print('torch.cuda.device_count()',torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net, device_ids=[0, 1])  # 模型分解
    net.cuda()
    # TODO: train LSTM model
    model=train_lstm(net,lr,train_loader,total_epoch)
    # model = model(IN_FEATURE, OUT_FEATURE)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(net, device_ids=[1, 2])  # 模型分解
    device=torch.device('cuda')
    model.to(device)
    print('finished training LSTM model')

    torch.save(model.state_dict(), './data-lstm-model/lstm_params.pth')


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--total_epoch', type=int, default=45, help='total epochs for training the model')
    p.add_argument('--batch_size', type=int, default=128, help='batch_size')
    args = p.parse_args()

    main(
        lr=args.lr,
        total_epoch=args.total_epoch,
        batch_size=args.batch_size
    )

