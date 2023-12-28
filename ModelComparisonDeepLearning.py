import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from base.layers import Conv2dWithConstraint, LinearWithConstraint, Swish, LogVarLayer
import sys

current_module = sys.modules[__name__]

class depthwise_separable_conv(nn.Module):  # 深度可分离卷积
    def __init__(self, nin, nout, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=(1, kernel_size), padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nout, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class EEGNet(nn.Module):  # Net1: EEGNet
    def __init__(self):
        super(EEGNet, self).__init__()
        self.Kernel = 80
        self.F1 = 8
        self.DF = 2
        self.Channel = 118
        self.Class = 2
        self.mapsize = 160

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.Kernel), padding=0))
        self.Extractor.add_module('p-1', nn.ZeroPad2d((int(self.Kernel / 2) - 1, int(self.Kernel / 2), 0, 0)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.Extractor.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.DF, (self.Channel, 1), groups=8))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-1', nn.ELU())

        self.Extractor.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.25))

        self.Extractor.add_module('c-3', depthwise_separable_conv(self.F1 * self.DF, self.F1 * self.DF, int(self.Kernel / 4)))
        self.Extractor.add_module('p-2', nn.ZeroPad2d((int(self.Kernel / 8) - 1, int(self.Kernel / 8), 0, 0)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 128))
        self.class_classifier.add_module('fb-1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('fc-2', nn.Linear(128, 64))
        self.class_classifier.add_module('fb-2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('fc-3', nn.Linear(64, self.Class))

    def forward(self, source_data):
        loss_1 = torch.from_numpy(np.array(0)).cuda()
        loss_2 = torch.from_numpy(np.array(0)).cuda()
        loss_3 = torch.from_numpy(np.array(0)).cuda()
        feature = self.Extractor(source_data)
        feature = feature.view(-1, self.mapsize)
        class_output = self.class_classifier(feature)

        return class_output, loss_1, loss_2, loss_3

class EEGNet_FLOPS(nn.Module):  # Net1: EEGNet
    def __init__(self):
        super(EEGNet_FLOPS, self).__init__()
        self.kernel_size = 80
        self.Filter_1 = 8
        self.DFilter = 2
        self.Filter_2 = 16
        self.ch_num = 118
        self.class_num = 2
        self.mapsize = 160

        self.feature = nn.Sequential()
        self.feature.add_module('c-1', nn.Conv2d(1, self.Filter_1, (1, self.kernel_size), padding=0))
        self.feature.add_module('p-1', nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('b-1', nn.BatchNorm2d(self.Filter_1, False))

        self.feature.add_module('c-2', nn.Conv2d(self.Filter_1, self.Filter_1 * self.DFilter, (self.ch_num, 1), groups=8))
        self.feature.add_module('b-2', nn.BatchNorm2d(self.Filter_1 * self.DFilter, False))
        self.feature.add_module('e-1', nn.ELU())

        self.feature.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('d-1', nn.Dropout(p=0.25))

        self.feature.add_module('c-3', depthwise_separable_conv(self.Filter_1 * self.DFilter, self.Filter_2, int(self.kernel_size / 4)))
        self.feature.add_module('p-2', nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('b-3', nn.BatchNorm2d(self.Filter_2, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 256))
        self.class_classifier.add_module('fb-1', nn.BatchNorm1d(256))
        self.class_classifier.add_module('fc-2', nn.Linear(256, 64))
        self.class_classifier.add_module('fb-2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('fc-3', nn.Linear(64, self.class_num))

    def forward(self, source_data):
        loss_1 = torch.from_numpy(np.array(0)).cuda()
        loss_2 = torch.from_numpy(np.array(0)).cuda()
        loss_3 = torch.from_numpy(np.array(0)).cuda()
        feature = self.feature(source_data)
        feature = feature.view(-1, self.mapsize)
        class_output = self.class_classifier(feature)

class DeepConvNet(nn.Module):  # Net2: DeepConvNet
    def __init__(self):
        super(DeepConvNet, self).__init__()
        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, 25, (1, 5), (1, 1)))
        self.Extractor.add_module('c-2', nn.Conv2d(25, 25, (118, 1), (1, 1)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(25, False))
        self.Extractor.add_module('e-1', nn.ELU())
        self.Extractor.add_module('p-1', nn.MaxPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.5))
        self.Extractor.add_module('c-3', nn.Conv2d(25, 50, (1, 5), (1, 1)))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(50, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('p-2', nn.MaxPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.5))
        self.Extractor.add_module('c-4', nn.Conv2d(50, 100, (1, 5), (1, 1)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(100, False))
        self.Extractor.add_module('e-3', nn.ELU())
        self.Extractor.add_module('p-3', nn.MaxPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-3', nn.Dropout(p=0.5))
        self.Extractor.add_module('c-5', nn.Conv2d(100, 200, (1, 5), (1, 1)))
        self.Extractor.add_module('b-4', nn.BatchNorm2d(200, False))
        self.Extractor.add_module('e-4', nn.ELU())
        self.Extractor.add_module('p-4', nn.MaxPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-4', nn.Dropout(p=0.5))

        self.Classifier = nn.Sequential()
        self.Classifier.add_module('fc-0', nn.Linear(3600, 256))
        self.Classifier.add_module('fc-b0', nn.BatchNorm1d(256))
        self.Classifier.add_module('fc-1', nn.Linear(256, 128))
        self.Classifier.add_module('fc-b1', nn.BatchNorm1d(128))
        self.Classifier.add_module('fc-2', nn.Linear(128, 64))
        self.Classifier.add_module('fc-b2', nn.BatchNorm1d(64))
        self.Classifier.add_module('fc3', nn.Linear(64, 2))

    def forward(self, source_data):
        loss_1 = torch.from_numpy(np.array(0)).cuda()
        loss_2 = torch.from_numpy(np.array(0)).cuda()
        loss_3 = torch.from_numpy(np.array(0)).cuda()

        source_all = self.Extractor(source_data)
        source_all = source_all.view(-1, 3600)
        output = self.Classifier(source_all)
        return output, loss_1, loss_2, loss_3

class DeepConvNet_FLOPS(nn.Module):  # Net2: DeepConvNet
    def __init__(self):
        super(DeepConvNet_FLOPS, self).__init__()
        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, 25, (1, 5), (1, 1)))
        self.Extractor.add_module('c-2', nn.Conv2d(25, 25, (118, 1), (1, 1)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(25, False))
        self.Extractor.add_module('e-1', nn.ELU())
        self.Extractor.add_module('p-1', nn.MaxPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.5))
        self.Extractor.add_module('c-3', nn.Conv2d(25, 50, (1, 5), (1, 1)))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(50, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('p-2', nn.MaxPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.5))
        self.Extractor.add_module('c-4', nn.Conv2d(50, 100, (1, 5), (1, 1)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(100, False))
        self.Extractor.add_module('e-3', nn.ELU())
        self.Extractor.add_module('p-3', nn.MaxPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-3', nn.Dropout(p=0.5))
        self.Extractor.add_module('c-5', nn.Conv2d(100, 200, (1, 5), (1, 1)))
        self.Extractor.add_module('b-4', nn.BatchNorm2d(200, False))
        self.Extractor.add_module('e-4', nn.ELU())
        self.Extractor.add_module('p-4', nn.MaxPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-4', nn.Dropout(p=0.5))

        self.Classifier = nn.Sequential()
        self.Classifier.add_module('fc-0', nn.Linear(3600, 256))
        self.Classifier.add_module('fc-b0', nn.BatchNorm1d(256))
        self.Classifier.add_module('fc-1', nn.Linear(256, 128))
        self.Classifier.add_module('fc-b1', nn.BatchNorm1d(128))
        self.Classifier.add_module('fc-2', nn.Linear(128, 64))
        self.Classifier.add_module('fc-b2', nn.BatchNorm1d(64))
        self.Classifier.add_module('fc3', nn.Linear(64, 2))

    def forward(self, source_data):
        loss_1 = torch.from_numpy(np.array(0)).cuda()
        loss_2 = torch.from_numpy(np.array(0)).cuda()
        loss_3 = torch.from_numpy(np.array(0)).cuda()

        source_all = self.Extractor(source_data)
        source_all = source_all.view(-1, 3600)
        output = self.Classifier(source_all)

class InceptionEEGNet_Block1(nn.Module):
    def __init__(self, kernel_size, num_channel=36):
        super(InceptionEEGNet_Block1, self).__init__()
        self.F=8
        self.D=2
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, self.F, kernel_size=(1, kernel_size)),
            nn.ZeroPad2d((int(kernel_size / 2) - 1, int(kernel_size / 2), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(self.F, self.F*self.D, kernel_size=(num_channel, 1),groups=self.F),
            nn.BatchNorm2d(self.F*self.D, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, self.F, kernel_size=(1, int(kernel_size/2))),
            nn.ZeroPad2d((int(kernel_size / 4) - 1, int(kernel_size / 4), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(self.F, self.F*self.D, kernel_size=(num_channel, 1),groups=self.F),
            nn.BatchNorm2d(self.F*self.D ,False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, self.F, kernel_size=(1, int(kernel_size / 4))),
            nn.ZeroPad2d((int(kernel_size / 8) - 1, int(kernel_size / 8), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(self.F, self.F * self.D, kernel_size=(num_channel, 1), groups=self.F),
            nn.BatchNorm2d(self.F*self.D, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch_pool = nn.AvgPool2d(kernel_size=(1, 4))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        N1 = torch.cat((branch1, branch2, branch3), dim=1)
        A1 = self.branch_pool(N1)
        return A1


class InceptionEEGNet_Block2(nn.Module):
    def __init__(self, kernel_size, num_channel=59):
        super(InceptionEEGNet_Block2, self).__init__()
        self.F=8
        self.D=2
        self.branch1 = nn.Sequential(
            nn.Conv2d(48, self.F, kernel_size=(1, int(kernel_size/4))),
            nn.ZeroPad2d((int(kernel_size / 8) - 1, int(kernel_size / 8), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(48, self.F, kernel_size=(1, int(kernel_size/8))),
            nn.ZeroPad2d((int(int(kernel_size/8)/2) - 1, int(int(kernel_size/8)/2), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(48, self.F, kernel_size=(1, int(kernel_size / 16))),
            nn.ZeroPad2d((int(int(kernel_size/16)/2) , int(int(kernel_size/16)/2), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch_pool = nn.AvgPool2d(kernel_size=(1, 2))

    def forward(self, x):
        branch1 = self.branch1(x)
        #print(branch1.size())
        branch2 = self.branch2(x)
        #print(branch2.size())
        branch3 = self.branch3(x)
        #print(branch3.size())
        N2 = torch.cat((branch1, branch2, branch3), dim=1)
        A2 = self.branch_pool(N2)
        return A2

class EEGInception(nn.Module):
    def __init__(self):
        super(EEGInception, self).__init__()
        # 定义了特征提取器，两个卷积层
        self.kernel_size = 40
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.Channel = 118

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('f_block1', InceptionEEGNet_Block1(kernel_size=80, num_channel=self.Channel))
        self.Extractor.add_module('f_block2', InceptionEEGNet_Block2(kernel_size=80, num_channel=self.Channel))
        self.Extractor.add_module('f_conv3', nn.Conv2d(24, 12, kernel_size=(1, int(self.kernel_size / 8))))
        self.Extractor.add_module('f_padding3', nn.ZeroPad2d((int(self.kernel_size / 16) - 1, int(self.kernel_size / 16), 0, 0)))
        self.Extractor.add_module('f_batchnorm3', nn.BatchNorm2d(12, False))
        self.Extractor.add_module('f_ELU3', nn.ELU())

        self.Extractor.add_module('f_dropout3', nn.Dropout(p=0.25))
        self.Extractor.add_module('f_pooling3', nn.AvgPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('f_conv4', nn.Conv2d(12, 6, kernel_size=(1, int(self.kernel_size / 16))))
        self.Extractor.add_module('f_padding4', nn.ZeroPad2d((int(self.kernel_size / 32), int(self.kernel_size / 32), 0, 0)))
        self.Extractor.add_module('f_batchnorm4', nn.BatchNorm2d(6, False))
        self.Extractor.add_module('f_ELU4', nn.ELU())
        self.Extractor.add_module('f_dropout4', nn.Dropout(p=0.25))
        self.Extractor.add_module('f_pooling4', nn.AvgPool2d(kernel_size=(1, 2)))
        self.Linear = nn.Sequential()
        self.Linear.add_module('c_fc1', nn.Linear(66, 64))
        self.Linear.add_module('f_dropout1', nn.Dropout(p=0.25))
        self.Linear.add_module('c_bn1', nn.BatchNorm1d(64))
        self.Linear.add_module('c_fc2', nn.Linear(64, 16))
        self.Linear.add_module('f_dropout2', nn.Dropout(p=0.25))
        self.Linear.add_module('c_bn2', nn.BatchNorm1d(16))
        self.Linear.add_module('c_fc3', nn.Linear(16, 2))

    def forward(self, source_data):
        loss_1 = torch.from_numpy(np.array(0)).cuda()
        loss_2 = torch.from_numpy(np.array(0)).cuda()
        loss_3 = torch.from_numpy(np.array(0)).cuda()
        source_global = self.Extractor(source_data)
        source_global = source_global.view(source_global.size()[0], -1)
        output = self.Linear(source_global)
        return output, loss_1, loss_2, loss_3

class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)

class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps
    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))

class EEGShallowConvNet(nn.Module):  # Net1: EEGNet
    def __init__(self):
        super(EEGShallowConvNet, self).__init__()
        self.K1 = 13
        self.K2 = 35
        self.S1 = 7
        self.F1 = 40
        self.DF = 2
        self.Channel = 118
        self.Class = 2
        self.Map = 1760

        self.Block1 = nn.Sequential()
        self.Block1.add_module('C-1', nn.Conv2d(1, self.F1, (1, self.K1), padding=0))
        self.Block1.add_module('C-2', nn.Conv2d(self.F1, self.F1, (self.Channel, 1), padding=0))
        self.Block1.add_module('B-1', nn.BatchNorm2d(self.F1, False))
        self.Block1.add_module('AC-1', ActSquare())
        self.Block1.add_module('A-1', nn.AvgPool2d(kernel_size=(1, self.K2), stride= (1, self.S1)))
        self.Block1.add_module('AC-2', ActLog())
        self.Block1.add_module('D-1', nn.Dropout(p=0.25))

        self.Linear = nn.Sequential()
        self.Linear.add_module('fc-1', nn.Linear(self.Map, 128))
        self.Linear.add_module('fb-1', nn.BatchNorm1d(128))
        self.Linear.add_module('fc-2', nn.Linear(128, 64))
        self.Linear.add_module('fb-2', nn.BatchNorm1d(64))
        self.Linear.add_module('fc-3', nn.Linear(64, self.Class))

    def forward(self, source_data):

        loss_1 = torch.from_numpy(np.array(0)).cuda()
        loss_2 = torch.from_numpy(np.array(0)).cuda()
        loss_3 = torch.from_numpy(np.array(0)).cuda()

        source_global = self.Block1(source_data)
        source_global = source_global.view(source_global.size()[0], -1)
        class_output = self.Linear(source_global)

        return class_output, loss_1, loss_2, loss_3

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
    if args.model == 'EEGNet':
        return {
        'elu': [EEGNet()]
        }
    elif args.model == 'DeepConvNet':
        return {
        'elu': [DeepConvNet()]
        }
    elif args.model == 'DeepConvNet_FLOPS':
        return {
        'elu': [DeepConvNet_FLOPS()]
        }
    elif args.model == 'EEGNet_FLOPS':
        return {
        'elu': [EEGNet_FLOPS()]
        }
    elif args.model == 'EEGInception':
        return {
        'elu': [EEGInception()]
        }
    elif args.model == 'EEGShallowConvNet':
        return {
        'elu': [EEGShallowConvNet()]
        }
    elif args.model == 'FBCNet':
        return {
        'elu': [FBCNet(nChan=118, nTime=350)]
        }
    else:
        raise TypeError('model type not defined.')