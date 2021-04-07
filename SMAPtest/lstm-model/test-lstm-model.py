import numpy as np
import os
from configargparse import ArgParser
from Drawmap import Draw_map
from Evaluation import compute_rmse_r2
import torch


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
        x = x.permute(1, 0, 2)
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

def main(model_name):
    label_test=np.load('./data-lstm-model/label_test.npy')
    features_test=np.load('./data-lstm-model/features_test.npy')
    scalar_set=np.load('./data-lstm-model/scalar_set.npy')
    IN_FEATURE = features_test.shape[2] * features_test.shape[3] * features_test.shape[4]
    OUT_FEATURE = label_test.shape[1] * label_test.shape[2]
    features_test = torch.tensor(features_test.reshape(-1, features_test.shape[1], IN_FEATURE),
                                 dtype=torch.float32).cuda()

    print('finished loading  data for LSTM model')

    model = myLSTM(input_size=IN_FEATURE, hidden_dim=OUT_FEATURE, layer_dim=1)
    model.cuda()

    model.load_state_dict(torch.load('./data-lstm-model/lstm_params.pth'))

    # TODO: Create predictions based on the test sets
    pred = create_predictions(model, features_test, scalar_set,label_test.shape[1],label_test.shape[2])
    print('finished testing LSTM model')
    # TODO: Computer score of R2 and RMSE
    dir = r"./results"
    if not os.path.exists(dir):
        os.mkdir(dir)
    result_name = model_name  + '.csv'
    compute_rmse_r2(label_test, pred, result_name)
    # TODO: Draw the map of predicted soil moisture

    print('label_test ',label_test.shape)
    Draw_map(pred, label_test, model_name)
    np.save(dir+'/observed soil moisture.npy', label_test)
    np.save(dir+'/predicted soil moisture by LSTM model.npy', pred)

if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--model_name', type=str, default='LSTM-model', help='name for prediction model')
    args = p.parse_args()

    main(
        model_name=args.model_name
    )

