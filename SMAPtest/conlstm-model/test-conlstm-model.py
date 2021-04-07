import numpy as np
import os
from configargparse import ArgParser
from Drawmap import Draw_map
from Evaluation import compute_rmse_r2
import torch

class myConvLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_dim,height,width, kernel_size=3, stride=3, layer_dim=1):
        """
        :param size:一日数据
        :param day: n日
        :param hidden_dim:隐藏层神经元
        :param layer_dim: 隐藏层个数
        :param output_dim: 输出
        """
        super(myConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.HeightNew = (height - kernel_size) // stride + 1
        self.WidthNew = (width - kernel_size) // stride + 1
        self.conv3d = torch.nn.Conv3d(3, 10, kernel_size=(1, kernel_size, kernel_size), stride=(1, kernel_size, kernel_size))
        self.lstm = torch.nn.LSTM(input_size=10 * self.HeightNew * self.WidthNew, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.dense = torch.nn.Linear(in_features=hidden_dim,out_features=hidden_dim)
        # self.relu1 = torch.nn.ReLU()
        # self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = x.reshape(-1, x.shape[1],x.shape[2], x.shape[3], x.shape[4])
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        x = x.permute(0, 2, 1, 3, 4)
        print('x.shape',x.shape)
        x = x.reshape(-1, x.shape[1],10 * self.HeightNew * self.WidthNew)
        x = x.permute(1,0,2)
        output, (w, y) = self.lstm(x)
        output = output.permute(1, 0, 2)
        output = output[:, -1, :]
        output = self.dense(output)
        return output



def create_predictions(net, data_test_x,scalar_set,a,b):
    """Create  predictions"""
    predict = net(data_test_x)
    # Unnormalize
    predict=predict*(scalar_set[-1,1]-scalar_set[-1,0])+scalar_set[-1,0]
    predict = predict.cpu().detach().numpy()
    predict=predict.reshape(-1,a,b)
    return predict

def main(model_name,ConvLSTM_kernel_size,ConvLSTM_num_layers):
    label_test=np.load('./data-ConLSTM-model/label_test.npy')
    print('label_test',label_test.shape)
    features_test=np.load('./data-ConLSTM-model/features_test.npy')
    scalar_set=np.load('./data-ConLSTM-model/scalar_set.npy')
    data_train = np.load('./data-ConLSTM-model/data_train.npy')


    print('features_test.shapel', features_test.shape)
    print('label_test.shapel', label_test.shape)
    features_test=torch.tensor(features_test.reshape
    (-1, features_test.shape[1],features_test.shape[2],features_test.shape[3],features_test.shape[4])
    , dtype=torch.float32).cuda()

    Height = features_test.shape[3]
    Width = features_test.shape[4]
    IN_FEATURE = features_test.shape[2] * features_test.shape[3] * features_test.shape[4]
    OUT_FEATURE =  features_test.shape[3] * features_test.shape[4]
    model = myConvLSTM(input_size=IN_FEATURE, hidden_dim=OUT_FEATURE, height=Height, width=Width, kernel_size=3, stride=3,
                     layer_dim=1)

    model.cuda()
    model.load_state_dict(torch.load('./data-ConLSTM-model/convlstm_params.pth'))
    print('finished loading  data for ConvLSTM model')
    # TODO: Create predictions based on the test sets
    pred = create_predictions(model, features_test, scalar_set, label_test.shape[1], label_test.shape[2])

    print('finished testing ConvLSTM model')
    # TODO: Computer score of R2 and RMSE
    dir = r"./results"
    if not os.path.exists(dir):
        os.mkdir(dir)
    result_name = model_name  + '.csv'
    print('label_test',label_test.shape)
    print('pred', pred.shape)
    compute_rmse_r2(label_test, pred, result_name)
    # TODO: Draw the map of predicted soil moisture

    print('label_test ',label_test.shape)
    Draw_map(pred,label_test,model_name)
    np.save(dir+'/observed soil moisture.npy', label_test)
    np.save(dir+'/predicted soil moisture by ConLSTM model.npy', pred)

if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--ConvLSTM_kernel_size', type=int, default=3, help='kernel size for ConvLSTM model')
    p.add_argument('--ConvLSTM_num_layers', type=int, default=2, help='numbers of layer for ConvLSTM model')
    p.add_argument('--model_name', type=str, default='ConvLSTM-model', help='name for prediction model')
    args = p.parse_args()

    main(
        ConvLSTM_kernel_size=args.ConvLSTM_kernel_size,
        ConvLSTM_num_layers=args.ConvLSTM_num_layers,
        model_name=args.model_name
    )

