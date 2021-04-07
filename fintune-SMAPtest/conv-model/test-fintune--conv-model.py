import numpy as np
import os
from configargparse import ArgParser
from Drawmap import Draw_map
from Evaluation import compute_rmse_r2
import torch.nn.functional as F
import torch


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

def create_predictions(net, data_test_x,scalar_set,a,b):
    """Create  predictions"""
    predict = net(data_test_x)
    # Unnormalize
    predict=predict*(scalar_set[-1,1]-scalar_set[-1,0])+scalar_set[-1,0]
    predict = predict.cpu().detach().numpy()
    predict=predict.reshape(-1,a,b)

    return predict

def main(model_name):
    label_test=np.load('../../SMAPtest/conv-model/data-con-model/label_test.npy')
    features_test=np.load('../../SMAPtest/conv-model/data-con-model/features_test.npy')
    scalar_set=np.load('../../SMAPtest/conv-model/data-con-model/scalar_set.npy')


    Height = features_test.shape[3]
    Width = features_test.shape[4]

    features_test = torch.tensor(features_test.reshape(-1, features_test.shape[1]*features_test.shape[2],
                                  features_test.shape[3],features_test.shape[4]),
                                  dtype=torch.float32).cuda()
    print('finished loading  data for conv model')

    # TODO: Create predictions based on the test sets
    print('label_test.shapel',label_test.shape)

    model = torch.load('./data-con-model/con_params.pkl')

    model = model.cuda()


    pred = create_predictions(model, features_test, scalar_set,Height,Width)
    print('finished testing conv model')
    # TODO: Computer score of R2 and RMSE
    dir = r"./results"
    if not os.path.exists(dir):
        os.mkdir(dir)
    result_name = model_name  + '.csv'
    compute_rmse_r2(label_test, pred, result_name)
    # TODO: Draw the map of predicted soil moisture

    print('label_test ',label_test.shape)
    label_test=label_test.reshape(-1,Height,Width)
    pred[pred[:] < 1e-5] = 0
    Draw_map(pred,label_test,model_name)
    np.save(dir+'/observed soil moisture.npy', label_test)
    np.save(dir+'/predicted soil moisture by Conv model.npy', pred)

if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--model_name', type=str, default='Conv-model', help='name for prediction model')
    args = p.parse_args()

    main(
        model_name=args.model_name
    )

