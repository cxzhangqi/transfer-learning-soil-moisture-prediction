import numpy as np
from configargparse import ArgParser
import torch
import torch.utils.data as Data
from torch.optim import lr_scheduler
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

class myLR(torch.nn.Module):
    def __init__(self, input_size, output_size):
        """
        :param size:一日数据
        :param day: n日
        :param hidden_dim:隐藏层神经元
        :param layer_dim: 隐藏层个数
        :param output_dim: 输出
        """
        super(myLR, self).__init__()
        self.input_size = input_size
        self.dense = torch.nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        x = x.reshape(-1,self.input_size)
        # print("xshape",x.shape)
        output = self.dense(x)
        return output

def train_LR(net, lr, train_loader, total_epoch):
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
            y = y.reshape(y.shape[0],-1)
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
    features_train = np.load('./data-lr-model/features_train.npy')
    label_train = np.load('./data-lr-model/label_train.npy')
    print('finished loading  data for lr model')


    # TODO: transform data for LR model
    IN_FEATURE = features_train.shape[2] * features_train.shape[3] * features_train.shape[4]
    features_train = torch.tensor(features_train.reshape(-1, features_train.shape[1], IN_FEATURE),
                                  dtype=torch.float32).cuda()
    OUT_FEATURE = label_train.shape[1] * label_train.shape[2]
    label_train = torch.tensor(label_train.reshape(-1, OUT_FEATURE), dtype=torch.float32).cuda()
    dataset = Data.TensorDataset(features_train, label_train)
    train_loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    print('finished transforming data for lr model')
    # TODO: build LSTM model
    net = myLR(input_size=IN_FEATURE*features_train.shape[1], output_size=OUT_FEATURE)
    net.cuda()
    # TODO: train LSTM model
    model = train_LR(net, lr, train_loader, total_epoch)
    print('finished training lr model')

    torch.save(model.state_dict(), './data-lr-model/lr_params.pth')


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--total_epoch', type=int, default=200, help='total epochs for training the model')
    p.add_argument('--batch_size', type=int, default=64, help='batch_size')
    args = p.parse_args()

    main(
        lr=args.lr,
        total_epoch=args.total_epoch,
        batch_size=args.batch_size
    )

