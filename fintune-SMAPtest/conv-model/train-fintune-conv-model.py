import numpy as np
from configargparse import ArgParser
import torch
import torch.utils.data as Data
from torch.optim import lr_scheduler
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
import os
import torch.nn.functional as F

class myCNN(torch.nn.Module):
    def __init__(self, input_channel, output_feature_seq_length,Height,Width,out_channel=48, kernel_size=3, stride=3):
        """
        :param size:一日数据
        :param day: n日
        :param hidden_dim:隐藏层神经元
        :param layer_dim: 隐藏层个数
        :param output_dim: 输出
        """
        super(myCNN, self).__init__()
        self.HeightNew = (Height - kernel_size) // stride + 1
        self.WidthNew = (Width - kernel_size)//stride + 1
        self.cnn = torch.nn.Conv2d(in_channels=input_channel, kernel_size=kernel_size, stride=stride, out_channels=out_channel)
        self.dense = torch.nn.Linear(in_features=out_channel * self.HeightNew * self.WidthNew, out_features=output_feature_seq_length)

    def forward(self, x):
        x = self.cnn(x)

        x = F.relu(x, inplace=True)

        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        return x

def train_conv(net,lr,train_loader,total_epoch):
    global_step = 1
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(0, 500, 150)][1:], gamma=0.05)
    loss_func = torch.nn.MSELoss()
    loss_metrics = AverageValueMeter()
    ########## training set##########
    for epoch in range(total_epoch):
        epoch_loss = 0
        for step, (x, y) in tqdm(enumerate(train_loader)):
            output = net(x)
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



def main(lr,total_epoch,model_name,batch_size):
    dir = r"./data-con-model"
    if not os.path.exists(dir):
        os.mkdir(dir)
    features_train=np.load('../../SMAPtest/conv-model/data-con-model/features_train.npy')
    label_train = np.load('../../SMAPtest/conv-model/data-con-model/label_train.npy')

    print('finished loading  data for finetune-conv model')

    features_train = torch.tensor(features_train.reshape(-1, features_train.shape[1]*features_train.shape[2],
                                  features_train.shape[3],features_train.shape[4]),
                                  dtype=torch.float32).cuda()

    label_train = torch.tensor(label_train.reshape(-1, label_train.shape[1]*label_train.shape[2]), dtype=torch.float32).cuda()

    dataset = Data.TensorDataset(features_train, label_train)
    train_loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    # TODO: build CNN model
    net = torch.load('../../ERA5test/conv-model/data-con-model/con_params.pkl')
    net.cuda()
    # TODO: train CNN model
    model=train_conv(net,lr,train_loader,total_epoch)
    print('finished training conv model')
    torch.save(model, './data-con-model/con_params.pkl')


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--total_epoch', type=int, default=30, help='total epochs for training the model')
    p.add_argument('--model_name', type=str, default='Conv-model', help='name for prediction model')
    p.add_argument('--batch_size', type=int, default=128, help='batch_size')
    args = p.parse_args()

    main(
        lr=args.lr,
        total_epoch=args.total_epoch,
        model_name=args.model_name,
        batch_size=args.batch_size
    )

