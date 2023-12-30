import numpy as np
import torch
import torch.nn as nn
from MMD import mmd_rbf
from MatrixDistance import euclidean_dist
import torch.utils.data
from torch.nn import functional as F
import torch.fft

class EEGProgress(nn.Module):
    def __init__(self, act_func):
        super(EEGProgress, self).__init__()
        self.F1 = 8
        self.fe_1 = torch.from_numpy(np.eye(5)).cuda()  # 特征提取矩阵-1
        self.fe_2 = torch.from_numpy(np.ones((5, 1))).cuda()  # 特征提取矩阵-2
        self.ET_L = torch.from_numpy(np.load('ET_L.npy', allow_pickle=True)).cuda()  # 读取权重
        self.ET_R = torch.from_numpy(np.load('ET_R.npy', allow_pickle=True)).cuda()  # 读取权重

        '''增加通道滤波器、数据预卷积'''
        self.TEMP_Sequence1 = nn.Sequential()
        self.TEMP_Sequence1.add_module('T1-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.TEMP_Sequence1.add_module('T1-Norm1', nn.BatchNorm2d(self.F1, False))

        '''电极融合层-1'''
        self.ELE_Sequence1 = nn.Sequential()
        # self.ELE_Sequence1.add_module('E1-ZeroPad1', nn.ZeroPad2d(padding=(0, 0, 1, 1)))  #
        self.ELE_Sequence1.add_module('E1-Conv1', nn.Conv2d(self.F1, self.F1 * 2, (15, 1), stride=(10, 1)))
        self.ELE_Sequence1.add_module('E1-Norm1', nn.BatchNorm2d(self.F1 * 2, False))

        '''电极融合层-2'''
        self.ELE_Sequence2 = nn.Sequential()
        self.ELE_Sequence2.add_module('E2-Conv1', nn.Conv2d(self.F1 * 2, self.F1 * 2, (5, 1), stride=(5, 1)))
        self.ELE_Sequence2.add_module('E2-Norm1', nn.BatchNorm2d(self.F1 * 2, False))
        self.ELE_Sequence2.add_module('E2-ELU1', nn.ELU())
        self.ELE_Sequence2.add_module('E2-AVGPool1', nn.AvgPool2d((1, 5)))
        self.ELE_Sequence2.add_module('E2-Drop1', nn.Dropout(p=0.25))

        '''进一步进行时间卷积和降采样'''
        self.TEMP_Sequence2 = nn.Sequential()
        self.TEMP_Sequence2.add_module('B-Conv1', nn.Conv2d(self.F1 * 2, self.F1 * 2, (1, 10), stride=(1, 1)))
        self.TEMP_Sequence2.add_module('B-Norm1', nn.BatchNorm2d(self.F1 * 2, False))
        self.TEMP_Sequence2.add_module('B-ELU1', nn.ELU())
        self.TEMP_Sequence2.add_module('B-AVGPool1', nn.AvgPool2d((1, 5)))
        self.TEMP_Sequence2.add_module('B-Drop1', nn.Dropout(p=0.25))

        '''电极融合层-3'''
        self.ELE_Sequence3 = nn.Sequential()
        self.ELE_Sequence3.add_module('E2-Conv1', nn.Conv2d(self.F1 * 2, self.F1 * 2, (2, 1), stride=(2, 1)))
        self.ELE_Sequence3.add_module('E2-Norm1', nn.BatchNorm2d(self.F1 * 2, False))
        self.ELE_Sequence3.add_module('E2-ELU1', nn.ELU())
        self.ELE_Sequence3.add_module('E2-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 2))

    def forward(self, source_L, target_L, source_R, target_R):
        inter_glob_loss = torch.from_numpy(np.array(0)).cuda()  # 数据整体分布距离
        inter_location_loss_ST = torch.from_numpy(np.array(0)).cuda()
        inter_location_loss_LR = torch.from_numpy(np.array(0)).cuda()
        inter_hemisphere_loss = torch.from_numpy(np.array(0)).cuda()

        source_L = self.TEMP_Sequence1(source_L)
        source_R = self.TEMP_Sequence1(source_R)
        target_L = self.TEMP_Sequence1(target_L)
        target_R = self.TEMP_Sequence1(target_R)

        source_ele_L = self.ELE_Sequence1(source_L)
        source_ele_R = self.ELE_Sequence1(source_R)
        target_ele_L = self.ELE_Sequence1(target_L)
        target_ele_R = self.ELE_Sequence1(target_R)

        source_ele_L = self.ELE_Sequence2(source_ele_L)
        source_ele_R = self.ELE_Sequence2(source_ele_R)
        target_ele_L = self.ELE_Sequence2(target_ele_L)
        target_ele_R = self.ELE_Sequence2(target_ele_R)

        source_L = self.TEMP_Sequence2(source_ele_L)
        source_R = self.TEMP_Sequence2(source_ele_R)
        target_L = self.TEMP_Sequence2(target_ele_L)
        target_R = self.TEMP_Sequence2(target_ele_R)

        # For EEGProgress 2.0
        psd_hemi_S_L = compute_region_psd(source_L)  # 形状为 (M, N, 5, K)
        psd_hemi_S_R = compute_region_psd(source_R)
        psd_hemi_T_L = compute_region_psd(target_L)
        psd_hemi_T_R = compute_region_psd(target_R)
        source_L = psd_hemi_T_L / psd_hemi_S_L * source_L
        source_R = psd_hemi_T_R / psd_hemi_S_R * source_R

        source_LR = torch.cat((source_L, source_R), -2)  # 拼接左右脑数据
        source_LR = self.ELE_Sequence3(source_LR)

        # 拉伸成条形处理
        s0, s1, _, s2 = source_LR.shape[:4]  # 读取张量大小
        source_LR = source_LR.reshape(s0, s1 * s2)  # [28,118*2]

        source_LR = self.FC1(source_LR)
        output = self.F_FC1(source_LR)
        return output, inter_hemisphere_loss, inter_location_loss_LR, inter_location_loss_ST, inter_glob_loss

def compute_region_psd(region_data):
    brain_region_fft = torch.fft.fft(region_data, dim=-1)
    brain_region_psd = torch.sum(torch.abs(brain_region_fft) ** 2)
    return brain_region_psd.item()

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
        # 'relu': [EEGNet('relu')],
        # 'lrelu': [EEGNet('lrelu')],
        }

    elif args.model == 'HDAN_1':
        return {
        'elu': [HDAN_1('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'HDAN_2':
        return {
        'elu': [HDAN_2('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'HDAN_3':
        return {
        'elu': [HDAN_3('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'HDAN_4':
        return {
        'elu': [HDAN_4('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'HDAN_5':
        return {
        'elu': [HDAN_5('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'HDAN_6':
        return {
        'elu': [HDAN_6('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'HDAN_7':
        return {
        'elu': [HDAN_7('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'HDAN_8':
        return {
        'elu': [HDAN_8('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'HDAN_9':
        return {
        'elu': [HDAN_9('relu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net3F':
        return {
        'elu': [DDC_FLOPs()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net4F':
        return {
        'elu': [DeepCoral_FLOPs()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net7F':
        return {
        'elu': [IA_EDAN_FLOPs('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net8F':
        return {
        'elu': [IE_EDAN_FLOPs('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net9F':
        return {
        'elu': [EDAN_Flops('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    else:
        raise TypeError('model type not defined.')