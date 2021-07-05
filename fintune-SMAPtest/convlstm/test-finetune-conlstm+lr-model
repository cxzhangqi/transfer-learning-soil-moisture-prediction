import numpy as np
import os
from configargparse import ArgParser
from Drawmap import Draw_map
from Evaluation import compute_rmse_r2
import torch
from torch.autograd import Variable
from convlstm import ConvLSTM




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


def create_predictions(encoder,LR_net,x,scalar_set,a,b,channel_number):
    """Create  predictions"""
    x = x.view(x.shape[1], x.shape[0], x.shape[2], x.shape[3], x.shape[4])
    hidden = encoder.get_init_states(x.shape[1])
    predict, encoder_state = encoder(x.clone(), hidden)

    predict = predict.permute(1, 0, 2, 3, 4)
    shape = predict.shape[0]
    predict = predict.view(shape,channel_number, a, b)
    # predict=predict.cpu().detach().numpy()
    predict = LR_net(predict)
    predict = predict.view(shape,1,1,a,b)
    predict = predict[-1,:,:,:,:] * (scalar_set[-1, 1] - scalar_set[-1, 0]) + scalar_set[-1, 0]
    predict = predict.cpu().detach().numpy()
    predict=predict.reshape(-1,a,b)

    return predict

def main(model_name,ConvLSTM_kernel_size,ConvLSTM_num_layers1,ConvLSTM_num_layers2):
    # label_test=np.load('./data-ConLSTM-model/label_test.npy')
    # features_test=np.load('./data-ConLSTM-model/features_test.npy')
    # scalar_set=np.load('./data-ConLSTM-model/scalar_set.npy')

    label_test = np.load('../../SMAPtest/convlstm-model/data-ConLSTM-model/label_test.npy')
    features_test = np.load('../../SMAPtest/convlstm-model/data-ConLSTM-model/features_test.npy')
    scalar_set = np.load('../../SMAPtest/convlstm-model/data-ConLSTM-model/scalar_set.npy')



    features_test=torch.tensor(features_test.reshape
    (-1, features_test.shape[1],features_test.shape[2],features_test.shape[3],features_test.shape[4])
    , dtype=torch.float32).cuda()


    ConvLSTM_net = ConvLSTM(input_size=(features_test.shape[3], features_test.shape[4]),
                       input_dim=features_test.shape[1],
                       hidden_dim=[ConvLSTM_num_layers1, ConvLSTM_num_layers2],
                       kernel_size=(ConvLSTM_kernel_size, ConvLSTM_kernel_size),
                       num_layers=2,
                       ).cuda()

  
    LR_net = LR(channel_number=ConvLSTM_num_layers2, output_size=features_test.shape[3] * features_test.shape[4]).cuda()

    ConvLSTM_net.load_state_dict(torch.load('./data-ConLSTM-model/ConvLSTM_net_params.pth'))
    LR_net.load_state_dict(torch.load('./data-ConLSTM-model/LR_net_params.pth'))
    
    print('finished loading  data for ConvLSTM model')
    # TODO: Create predictions based on the test sets
    print('label_test.shapel',label_test.shape)
    pred = create_predictions(ConvLSTM_net,LR_net, features_test, scalar_set, label_test.shape[1], label_test.shape[2],channel_number=ConvLSTM_num_layers2)
    print('pred',pred.shape)
    print('finished testing ConvLSTM model')
    # TODO: Computer score of R2 and RMSE
    dir = r"./results"
    if not os.path.exists(dir):
        os.mkdir(dir)
    result_name = model_name  + '.csv'
    compute_rmse_r2(label_test, pred, result_name)
    # TODO: Draw the map of predicted soil moisture

    print('label_test ',label_test.shape)
    Draw_map(pred,label_test,model_name)
    np.save(dir+'/observed soil moisture.npy', label_test)
    np.save(dir+'/predicted soil moisture by ConLSTM model.npy', pred)

if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--ConvLSTM_kernel_size', type=int, default=3, help='kernel size for ConvLSTM model')
    p.add_argument('--ConvLSTM_num_layers1', type=int, default=8, help='numbers of layer for ConvLSTM model')
    p.add_argument('--ConvLSTM_num_layers2', type=int, default=4, help='numbers of layer for ConvLSTM model')
    p.add_argument('--model_name', type=str, default='ConvLSTM-model', help='name for prediction model')
    args = p.parse_args()

    main(
        ConvLSTM_kernel_size=args.ConvLSTM_kernel_size,
        ConvLSTM_num_layers1=args.ConvLSTM_num_layers1,
        ConvLSTM_num_layers2=args.ConvLSTM_num_layers2,
        model_name=args.model_name
    )

