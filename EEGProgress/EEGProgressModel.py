import torch
import torch.nn as nn
import torch.utils.data
import torch.fft

class EEGProgress(nn.Module):  # New architecture
    def __init__(self, act_func):
        super(EEGProgress, self).__init__()
        self.F1 = 8

        self.TEMP_Sequence1 = nn.Sequential()
        self.TEMP_Sequence1.add_module('T1-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.TEMP_Sequence1.add_module('T1-Norm1', nn.BatchNorm2d(self.F1, False))

        self.ELE_Sequence1 = nn.Sequential()
        self.ELE_Sequence1.add_module('E1-Conv1', nn.Conv2d(self.F1, self.F1 * 2, (15, 1), stride=(10, 1)))
        self.ELE_Sequence1.add_module('E1-Norm1', nn.BatchNorm2d(self.F1 * 2, False))

        self.ELE_Sequence2 = nn.Sequential()
        self.ELE_Sequence2.add_module('E2-Conv1', nn.Conv2d(self.F1 * 2, self.F1 * 2, (5, 1), stride=(5, 1)))
        self.ELE_Sequence2.add_module('E2-Norm1', nn.BatchNorm2d(self.F1 * 2, False))
        self.ELE_Sequence2.add_module('E2-ELU1', nn.ELU())
        self.ELE_Sequence2.add_module('E2-AVGPool1', nn.AvgPool2d((1, 5)))
        self.ELE_Sequence2.add_module('E2-Drop1', nn.Dropout(p=0.25))

        self.TEMP_Sequence2 = nn.Sequential()
        self.TEMP_Sequence2.add_module('B-Conv1', nn.Conv2d(self.F1 * 2, self.F1 * 2, (1, 10), stride=(1, 1)))
        self.TEMP_Sequence2.add_module('B-Norm1', nn.BatchNorm2d(self.F1 * 2, False))
        self.TEMP_Sequence2.add_module('B-ELU1', nn.ELU())
        self.TEMP_Sequence2.add_module('B-AVGPool1', nn.AvgPool2d((1, 5)))
        self.TEMP_Sequence2.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.ELE_Sequence3 = nn.Sequential()
        self.ELE_Sequence3.add_module('E2-Conv1', nn.Conv2d(self.F1 * 2, self.F1 * 2, (2, 1), stride=(2, 1)))
        self.ELE_Sequence3.add_module('E2-Norm1', nn.BatchNorm2d(self.F1 * 2, False))
        self.ELE_Sequence3.add_module('E2-ELU1', nn.ELU())
        self.ELE_Sequence3.add_module('E2-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 18))
        self.FC1.add_module('E_FC2', nn.Linear(18, 2))

    def forward(self, source_L, source_R):

        source_L = self.TEMP_Sequence1(source_L)
        source_R = self.TEMP_Sequence1(source_R)

        source_ele_L = self.ELE_Sequence1(source_L)
        source_ele_R = self.ELE_Sequence1(source_R)

        source_ele_L = self.ELE_Sequence2(source_ele_L)
        source_ele_R = self.ELE_Sequence2(source_ele_R)

        source_L = self.TEMP_Sequence2(source_ele_L)
        source_R = self.TEMP_Sequence2(source_ele_R)

        source_LR = torch.cat((source_L, source_R), -2)  # 拼接左右脑数据
        source_LR = self.ELE_Sequence3(source_LR)

        # 拉伸成条形处理
        s0, s1, _, s2 = source_LR.shape[:4]  # 读取张量大小
        source_LR = source_LR.reshape(s0, s1 * s2)  # [28,118*2]
        output = self.FC1(source_LR)
        return output

'''选取激活函数类型'''
def choose_act_func(act_name):
    if act_name == 'elu':
        return nn.ELU()
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'lrelu':
        return nn.LeakyReLU()
    else:
        raise TypeError('activation_function type not defined.')

'''处理预定义网络和训练参数'''
def handle_param(args, net):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'rmsp':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    else:
        raise TypeError('optimizer type not defined.')
    if args.loss_function == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    else:
        raise TypeError('loss_function type not defined.')
    return optimizer, loss_function

'''选取网络和激活函数'''
def choose_net(args):
    if args.model == 'EEGProgress':
        return {
        'elu': [EEGProgress('relu')]
        }
    else:
        raise TypeError('model type not defined.')