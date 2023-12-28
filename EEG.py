import numpy as np
import torch
import torch.nn as nn
from MMD import mmd_rbf
from MatrixDistance import euclidean_dist
import torch.utils.data
from torch.nn import functional as F


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
        self.kernel_size = 80
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = 118
        self.num_classes = 2
        self.feature_map_size = 160
        self.feature = nn.Sequential()

        self.feature.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.kernel_size), padding=0))
        self.feature.add_module('p-1', nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.feature.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.D, (self.num_channel, 1), groups=8))
        self.feature.add_module('b-2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.feature.add_module('e-1', nn.ELU())

        self.feature.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('d-1', nn.Dropout(p=0.25))

        self.feature.add_module('c-3', depthwise_separable_conv(self.F1 * self.D, self.F2, int(self.kernel_size / 4)))
        self.feature.add_module('p-2', nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('b-3', nn.BatchNorm2d(self.F2, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.feature_map_size, 128))
        self.class_classifier.add_module('fb-1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('fc-2', nn.Linear(128, 64))
        self.class_classifier.add_module('fb-2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('fc-3', nn.Linear(64, self.num_classes))

    def forward(self, source_data, target_data):
        inter_global_loss = torch.from_numpy(np.array(0)).cuda()
        inter_ele_loss = torch.from_numpy(np.array(0)).cuda()
        intra_ele_loss = torch.from_numpy(np.array(0)).cuda()
        feature = self.feature(source_data)
        feature = feature.view(-1, self.feature_map_size)
        class_output = self.class_classifier(feature)
        return class_output, inter_global_loss, intra_ele_loss, inter_ele_loss


class EEGNet_FLOPs(nn.Module):  # Net1: EEGNet
    def __init__(self):
        super(EEGNet_FLOPs, self).__init__()
        self.kernel_size = 80
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = 118
        self.num_classes = 2
        self.feature_map_size = 160
        self.feature = nn.Sequential()

        self.feature.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.kernel_size), padding=0))
        self.feature.add_module('p-1', nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.feature.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.D, (self.num_channel, 1), groups=8))
        self.feature.add_module('b-2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.feature.add_module('e-1', nn.ELU())

        self.feature.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('d-1', nn.Dropout(p=0.25))

        self.feature.add_module('c-3', depthwise_separable_conv(self.F1 * self.D, self.F2, int(self.kernel_size / 4)))
        self.feature.add_module('p-2', nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('b-3', nn.BatchNorm2d(self.F2, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.feature_map_size, 128))
        self.class_classifier.add_module('fb-1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('fc-2', nn.Linear(128, 64))
        self.class_classifier.add_module('fb-2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('fc-3', nn.Linear(64, self.num_classes))

    def forward(self, source_data, target_data):
        inter_global_loss = torch.from_numpy(np.array(0)).cuda()
        inter_ele_loss = torch.from_numpy(np.array(0)).cuda()
        intra_ele_loss = torch.from_numpy(np.array(0)).cuda()
        feature = self.feature(source_data)
        feature = feature.view(-1, self.feature_map_size)
        class_output = self.class_classifier(feature)


class DeepConvNet(nn.Module):  # Net2: DeepConvNet
    def __init__(self):
        super(DeepConvNet, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('c-1', nn.Conv2d(1, 25, (1, 5), (1, 1)))
        self.feature.add_module('c-2', nn.Conv2d(25, 25, (118, 1), (1, 1)))
        self.feature.add_module('b-1', nn.BatchNorm2d(25, False))
        self.feature.add_module('e-1', nn.ELU())
        self.feature.add_module('p-1', nn.MaxPool2d(kernel_size=(1, 2)))
        self.feature.add_module('d-1', nn.Dropout(p=0.5))
        self.feature.add_module('c-3', nn.Conv2d(25, 50, (1, 5), (1, 1)))
        self.feature.add_module('b-2', nn.BatchNorm2d(50, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('p-2', nn.MaxPool2d(kernel_size=(1, 2)))
        self.feature.add_module('d-2', nn.Dropout(p=0.5))
        self.feature.add_module('c-4', nn.Conv2d(50, 100, (1, 5), (1, 1)))
        self.feature.add_module('b-3', nn.BatchNorm2d(100, False))
        self.feature.add_module('e-3', nn.ELU())
        self.feature.add_module('p-3', nn.MaxPool2d(kernel_size=(1, 2)))
        self.feature.add_module('d-3', nn.Dropout(p=0.5))
        self.feature.add_module('c-5', nn.Conv2d(100, 200, (1, 5), (1, 1)))
        self.feature.add_module('b-4', nn.BatchNorm2d(200, False))
        self.feature.add_module('e-4', nn.ELU())
        self.feature.add_module('p-4', nn.MaxPool2d(kernel_size=(1, 2)))
        self.feature.add_module('d-4', nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc-0', nn.Linear(3600, 256))
        self.classifier.add_module('fc-b0', nn.BatchNorm1d(256))
        self.classifier.add_module('fc-1', nn.Linear(256, 128))
        self.classifier.add_module('fc-b1', nn.BatchNorm1d(128))
        self.classifier.add_module('fc-2', nn.Linear(128, 64))
        self.classifier.add_module('fc-b2', nn.BatchNorm1d(64))
        self.classifier.add_module('fc3', nn.Linear(64, 2))

    def forward(self, source_data, target_data):
        inter_global_loss = torch.from_numpy(np.array(0)).cuda()
        inter_ele_loss = torch.from_numpy(np.array(0)).cuda()
        intra_ele_loss = torch.from_numpy(np.array(0)).cuda()

        source_data = source_data.type(torch.cuda.FloatTensor)
        source_all = self.feature(source_data)
        source_all = source_all.view(-1, 3600)
        output = self.classifier(source_all)
        return output, inter_global_loss, intra_ele_loss, inter_ele_loss


class DeepConvNet_FLOPs(nn.Module):  # Net2: DeepConvNet
    def __init__(self):
        super(DeepConvNet_FLOPs, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('c-1', nn.Conv2d(1, 25, (1, 5), (1, 1)))
        self.feature.add_module('c-2', nn.Conv2d(25, 25, (118, 1), (1, 1)))
        self.feature.add_module('b-1', nn.BatchNorm2d(25, False))
        self.feature.add_module('e-1', nn.ELU())
        self.feature.add_module('p-1', nn.MaxPool2d(kernel_size=(1, 2)))
        self.feature.add_module('d-1', nn.Dropout(p=0.5))
        self.feature.add_module('c-3', nn.Conv2d(25, 50, (1, 5), (1, 1)))
        self.feature.add_module('b-2', nn.BatchNorm2d(50, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('p-2', nn.MaxPool2d(kernel_size=(1, 2)))
        self.feature.add_module('d-2', nn.Dropout(p=0.5))
        self.feature.add_module('c-4', nn.Conv2d(50, 100, (1, 5), (1, 1)))
        self.feature.add_module('b-3', nn.BatchNorm2d(100, False))
        self.feature.add_module('e-3', nn.ELU())
        self.feature.add_module('p-3', nn.MaxPool2d(kernel_size=(1, 2)))
        self.feature.add_module('d-3', nn.Dropout(p=0.5))
        self.feature.add_module('c-5', nn.Conv2d(100, 200, (1, 5), (1, 1)))
        self.feature.add_module('b-4', nn.BatchNorm2d(200, False))
        self.feature.add_module('e-4', nn.ELU())
        self.feature.add_module('p-4', nn.MaxPool2d(kernel_size=(1, 2)))
        self.feature.add_module('d-4', nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc-0', nn.Linear(3600, 256))
        self.classifier.add_module('fc-b0', nn.BatchNorm1d(256))
        self.classifier.add_module('fc-1', nn.Linear(256, 128))
        self.classifier.add_module('fc-b1', nn.BatchNorm1d(128))
        self.classifier.add_module('fc-2', nn.Linear(128, 64))
        self.classifier.add_module('fc-b2', nn.BatchNorm1d(64))
        self.classifier.add_module('fc3', nn.Linear(64, 2))

    def forward(self, source_data, target_data):
        inter_global_loss = torch.from_numpy(np.array(0)).cuda()
        inter_ele_loss = torch.from_numpy(np.array(0)).cuda()
        intra_ele_loss = torch.from_numpy(np.array(0)).cuda()

        source_data = source_data
        source_all = self.feature(source_data)
        source_all = source_all.view(-1, 3600)


class DDC(nn.Module):  # Net3: DDC using EEGNet
    def __init__(self):
        super(DDC, self).__init__()
        self.kernel_size = 80
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = 118
        self.num_classes = 2
        self.feature_map_size = 160
        self.feature = nn.Sequential()

        self.feature.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.kernel_size), padding=0))
        self.feature.add_module('p-1', nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.feature.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.D, (self.num_channel, 1), groups=8))
        self.feature.add_module('b-2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.feature.add_module('e-1', nn.ELU())

        self.feature.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('d-1', nn.Dropout(p=0.25))

        self.feature.add_module('c-3', depthwise_separable_conv(self.F1 * self.D, self.F2, int(self.kernel_size / 4)))
        self.feature.add_module('p-2', nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('b-3', nn.BatchNorm2d(self.F2, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.feature_map_size, 128))
        self.class_classifier.add_module('fb-1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('fc-2', nn.Linear(128, 64))
        self.class_classifier.add_module('fb-2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('fc-3', nn.Linear(64, self.num_classes))

    def forward(self, source_data, target_data):
        inter_global_loss = 0
        inter_ele_loss = torch.from_numpy(np.array(0)).cuda()
        intra_ele_loss = torch.from_numpy(np.array(0)).cuda()
        source_all = self.feature(source_data)
        source_all = source_all.view(-1, self.feature_map_size)

        if self.training:
            target_all = self.feature(target_data)
            target_all = target_all.view(-1, self.feature_map_size)
            inter_global_loss += mmd_rbf(source_all, target_all, kernel_mul=5.0, kernel_num=10, fix_sigma=None)
            output = self.class_classifier(source_all)

        return output, inter_global_loss, intra_ele_loss, inter_ele_loss


class DDC_FLOPs(nn.Module):  # Net3: DDC using EEGNet
    def __init__(self):
        super(DDC_FLOPs, self).__init__()
        self.kernel_size = 80
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = 118
        self.num_classes = 2
        self.feature_map_size = 160
        self.feature = nn.Sequential()

        self.feature.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.kernel_size), padding=0))
        self.feature.add_module('p-1', nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.feature.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.D, (self.num_channel, 1), groups=8))
        self.feature.add_module('b-2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.feature.add_module('e-1', nn.ELU())

        self.feature.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('d-1', nn.Dropout(p=0.25))

        self.feature.add_module('c-3', depthwise_separable_conv(self.F1 * self.D, self.F2, int(self.kernel_size / 4)))
        self.feature.add_module('p-2', nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('b-3', nn.BatchNorm2d(self.F2, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.feature_map_size, 128))
        self.class_classifier.add_module('fb-1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('fc-2', nn.Linear(128, 64))
        self.class_classifier.add_module('fb-2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('fc-3', nn.Linear(64, self.num_classes))

    def forward(self, source_data, target_data):
        inter_global_loss = 0
        inter_ele_loss = torch.from_numpy(np.array(0)).cuda()
        intra_ele_loss = torch.from_numpy(np.array(0)).cuda()
        source_all = self.feature(source_data)
        source_all = source_all.view(-1, self.feature_map_size)

        target_all = self.feature(target_data)
        target_all = target_all.view(-1, self.feature_map_size)
        inter_global_loss += mmd_rbf(source_all, target_all, kernel_mul=5.0, kernel_num=10, fix_sigma=None)
        output = self.class_classifier(source_all)


class DeepCoral(nn.Module):  # Net4: DeepCoral
    def __init__(self):
        super(DeepCoral, self).__init__()
        self.kernel_size = 80
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = 118
        self.num_classes = 2
        self.feature_map_size = 160
        self.feature = nn.Sequential()

        self.feature.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.kernel_size), padding=0))
        self.feature.add_module('p-1', nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.feature.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.D, (self.num_channel, 1), groups=8))
        self.feature.add_module('b-2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.feature.add_module('e-1', nn.ELU())

        self.feature.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('d-1', nn.Dropout(p=0.25))

        self.feature.add_module('c-3', depthwise_separable_conv(self.F1 * self.D, self.F2, int(self.kernel_size / 4)))
        self.feature.add_module('p-2', nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('b-3', nn.BatchNorm2d(self.F2, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('d-2', nn.Dropout(p=0.25))

        self.classifier_1 = nn.Sequential()
        self.classifier_1.add_module('fc-1', nn.Linear(self.feature_map_size, 128))
        self.classifier_1.add_module('fb-1', nn.BatchNorm1d(128))

        self.classifier_2 = nn.Sequential()
        self.classifier_2.add_module('fc-2', nn.Linear(128, 64))
        self.classifier_2.add_module('fb-2', nn.BatchNorm1d(64))

        self.classifier_3 = nn.Sequential()
        self.classifier_3.add_module('fc-3', nn.Linear(64, self.num_classes))

    def forward(self, source_data, target_data):

        loss_1 = 0
        loss_2 = 0
        loss_3 = 0

        source_all = self.feature(source_data)
        source_all_1 = source_all.view(-1, self.feature_map_size)
        source_all_2 = self.classifier_1(source_all_1)
        source_all_3 = self.classifier_2(source_all_2)

        if self.training:
            target_all = self.feature(target_data)

            target_all_1 = target_all.view(-1, self.feature_map_size)
            s1 = torch.matmul(source_all_1.T, source_all_1)
            t1 = torch.matmul(target_all_1.T, target_all_1)
            loss_1 += euclidean_dist(s1, t1)

            target_all_2 = self.classifier_1(target_all_1)
            s2 = torch.matmul(source_all_2.T, source_all_2)
            t2 = torch.matmul(target_all_2.T, target_all_2)
            loss_2 += euclidean_dist(s2, t2)

            target_all_3 = self.classifier_2(target_all_2)
            s3 = torch.matmul(source_all_3.T, source_all_3)
            t3 = torch.matmul(target_all_3.T, target_all_3)
            loss_3 += euclidean_dist(s3, t3)

        output = self.classifier_3(source_all_3)
        return output, loss_1, loss_2, loss_3


class DeepCoral_FLOPs(nn.Module):  # Net4: DeepCoral
    def __init__(self):
        super(DeepCoral_FLOPs, self).__init__()
        self.kernel_size = 80
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = 118
        self.num_classes = 2
        self.feature_map_size = 160
        self.feature = nn.Sequential()

        self.feature.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.kernel_size), padding=0))
        self.feature.add_module('p-1', nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.feature.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.D, (self.num_channel, 1), groups=8))
        self.feature.add_module('b-2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.feature.add_module('e-1', nn.ELU())

        self.feature.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('d-1', nn.Dropout(p=0.25))

        self.feature.add_module('c-3', depthwise_separable_conv(self.F1 * self.D, self.F2, int(self.kernel_size / 4)))
        self.feature.add_module('p-2', nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('b-3', nn.BatchNorm2d(self.F2, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('d-2', nn.Dropout(p=0.25))

        self.classifier_1 = nn.Sequential()
        self.classifier_1.add_module('fc-1', nn.Linear(self.feature_map_size, 128))
        self.classifier_1.add_module('fb-1', nn.BatchNorm1d(128))

        self.classifier_2 = nn.Sequential()
        self.classifier_2.add_module('fc-2', nn.Linear(128, 64))
        self.classifier_2.add_module('fb-2', nn.BatchNorm1d(64))

        self.classifier_3 = nn.Sequential()
        self.classifier_3.add_module('fc-3', nn.Linear(64, self.num_classes))

    def forward(self, source_data, target_data):

        loss_1 = 0
        loss_2 = 0
        loss_3 = 0

        source_all = self.feature(source_data)
        source_all_1 = source_all.view(-1, self.feature_map_size)
        source_all_2 = self.classifier_1(source_all_1)
        source_all_3 = self.classifier_2(source_all_2)

        target_all = self.feature(target_data)

        target_all_1 = target_all.view(-1, self.feature_map_size)
        s1 = torch.matmul(source_all_1.T, source_all_1)
        t1 = torch.matmul(target_all_1.T, target_all_1)
        loss_1 += euclidean_dist(s1, t1)

        target_all_2 = self.classifier_1(target_all_1)
        s2 = torch.matmul(source_all_2.T, source_all_2)
        t2 = torch.matmul(target_all_2.T, target_all_2)
        loss_2 += euclidean_dist(s2, t2)

        target_all_3 = self.classifier_2(target_all_2)
        s3 = torch.matmul(source_all_3.T, source_all_3)
        t3 = torch.matmul(target_all_3.T, target_all_3)
        loss_3 += euclidean_dist(s3, t3)


class IA_EDAN(nn.Module):
    def __init__(self, act_func):
        super(IA_EDAN, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112

        self.SUB_Sequence1 = nn.Sequential()
        self.SUB_Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.SUB_Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, False))
        self.SUB_Sequence1.add_module('A-ELU1', nn.ELU())
        self.SUB_Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.SUB_Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.BOTTEN_Sequence3 = nn.Sequential()
        self.BOTTEN_Sequence3.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1*2, (1, 10), stride=(1, 1)))
        self.BOTTEN_Sequence3.add_module('B-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.BOTTEN_Sequence3.add_module('B-ELU1', nn.ELU())
        self.BOTTEN_Sequence3.add_module('B-AVGPool1', nn.AvgPool2d((1, 5)))
        self.BOTTEN_Sequence3.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.SUB_Sequence2 = nn.Sequential()
        self.SUB_Sequence2.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (118, 1), stride=(1, 1)))
        self.SUB_Sequence2.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, False))
        self.SUB_Sequence2.add_module('S-ELU1', nn.ELU())
        self.SUB_Sequence2.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 2))

    def forward(self, source, target):
        glob_loss = 0
        inter_ele_loss = torch.from_numpy(np.array(0)).cuda()
        intra_ele_loss = 0
        intra_target_ele_loss = 0
        intra_source_ele_loss = 0

        source_all = self.SUB_Sequence1(source)
        source_all = self.BOTTEN_Sequence3(source_all)
        source_ele_1 = self.SFC1(source_all)  # 3D全连接层1

        # 源域用户内电极损失
        _s0, _s1, _s2 = source_ele_1.shape[:3]
        SE1 = source_ele_1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
        _s = SE1 - SE1.unsqueeze(1)  # [s2, s2, s0, s1*s3]
        _s = _s @ _s.transpose(-1, -2)  # [s2, s2, s0, s0]
        _ms = _s.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
        intra_source_ele_loss += _ms / (_s2 * _s2) / (_s1 * _s1)

        source_ele_2 = self.SFC2(source_ele_1)  # 3D全连接层2

        _s0, _, _s2 = source_ele_2.shape[:3]
        SE2 = source_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]

        source_all = self.SUB_Sequence2(source_ele_2)

        if self.training:
            target_all = self.SUB_Sequence1(target)
            target_all = self.BOTTEN_Sequence3(target_all)
            target_ele_1 = self.SFC1(target_all)  # 3D全连接层1

            # 目标域用户内电极损失
            _t0, _t1, _t2 = target_ele_1.shape[:3]
            TE1 = target_ele_1.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3]
            _t = TE1 - TE1.unsqueeze(1)  # [s2, s2, s0, s1*s3]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0]
            _mt = _t.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
            intra_target_ele_loss += _mt / (_t2 * _t2) / (_t1 * _t1)
            intra_ele_loss += intra_source_ele_loss + intra_target_ele_loss

            target_ele_2 = self.SFC2(target_ele_1)  # 3D全连接层2
            target_all = self.SUB_Sequence2(target_ele_2)

            # 拉伸成条形处理
            s0, s1, s2, s3 = target_all.shape[:4]  # 读取张量大小
            target_all = target_all.reshape(s0, s1 * s3)  # [28,118*2]
            source_all = source_all.reshape(s0, s1 * s3)  # [28,118*2]

            target_all = self.FC1(target_all)
            source_all = self.FC1(source_all)

            glob_loss += mmd_rbf(source_all, target_all, kernel_mul=5.0, kernel_num=10, fix_sigma=None)  # 整体损失

            output = self.F_FC1(source_all)
        return output, glob_loss, intra_ele_loss, inter_ele_loss


class IA_EDAN_FLOPs(nn.Module):
    def __init__(self, act_func):
        super(IA_EDAN_FLOPs, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112

        self.SUB_Sequence1 = nn.Sequential()
        self.SUB_Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.SUB_Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, False))
        self.SUB_Sequence1.add_module('A-ELU1', nn.ELU())
        self.SUB_Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.SUB_Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.BOTTEN_Sequence3 = nn.Sequential()
        self.BOTTEN_Sequence3.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1*2, (1, 10), stride=(1, 1)))
        self.BOTTEN_Sequence3.add_module('B-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.BOTTEN_Sequence3.add_module('B-ELU1', nn.ELU())
        self.BOTTEN_Sequence3.add_module('B-AVGPool1', nn.AvgPool2d((1, 5)))
        self.BOTTEN_Sequence3.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.SUB_Sequence2 = nn.Sequential()
        self.SUB_Sequence2.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (118, 1), stride=(1, 1)))
        self.SUB_Sequence2.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, False))
        self.SUB_Sequence2.add_module('S-ELU1', nn.ELU())
        self.SUB_Sequence2.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 2))

    def forward(self, source, target):
        glob_loss = 0
        intra_source_ele_loss = 0
        intra_target_ele_loss = 0
        intra_ele_loss = 0
        inter_ele_loss = 0

        source_all = self.SUB_Sequence1(source)
        source_all = self.BOTTEN_Sequence3(source_all)
        source_ele_1 = self.SFC1(source_all)  # 3D全连接层1

        # 源域用户内电极损失
        _s0, _s1, _s2 = source_ele_1.shape[:3]
        SE1 = source_ele_1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
        _s = SE1 - SE1.unsqueeze(1)  # [s2, s2, s0, s1*s3]
        _s = _s @ _s.transpose(-1, -2)  # [s2, s2, s0, s0]
        _ms = _s.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
        intra_source_ele_loss += _ms / (_s2 * _s2) / (_s1 * _s1)

        source_ele_2 = self.SFC2(source_ele_1)  # 3D全连接层2

        _s0, _, _s2 = source_ele_2.shape[:3]
        SE2 = source_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]

        source_all = self.SUB_Sequence2(source_ele_2)

        target_all = self.SUB_Sequence1(target)
        target_all = self.BOTTEN_Sequence3(target_all)
        target_ele_1 = self.SFC1(target_all)  # 3D全连接层1
        # 目标域用户内电极损失
        _t0, _t1, _t2 = target_ele_1.shape[:3]
        TE1 = target_ele_1.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3]
        _t = TE1 - TE1.unsqueeze(1)  # [s2, s2, s0, s1*s3]
        _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0]
        _mt = _t.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
        intra_target_ele_loss += _mt / (_t2 * _t2) / (_t1 * _t1)
        intra_ele_loss += intra_source_ele_loss + intra_target_ele_loss
        target_ele_2 = self.SFC2(target_ele_1)  # 3D全连接层2
        target_all = self.SUB_Sequence2(target_ele_2)
        # 拉伸成条形处理
        s0, s1, s2, s3 = target_all.shape[:4]  # 读取张量大小
        target_all = target_all.reshape(s0, s1 * s3)  # [28,118*2]
        source_all = source_all.reshape(s0, s1 * s3)  # [28,118*2]
        target_all = self.FC1(target_all)
        source_all = self.FC1(source_all)
        glob_loss += mmd_rbf(source_all, target_all, kernel_mul=5.0, kernel_num=10, fix_sigma=None)  # 整体损失
        output = self.F_FC1(source_all)


class IE_EDAN(nn.Module):
    def __init__(self, act_func):
        super(IE_EDAN, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112

        self.SUB_Sequence1 = nn.Sequential()
        self.SUB_Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.SUB_Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, False))
        self.SUB_Sequence1.add_module('A-ELU1', nn.ELU())
        self.SUB_Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.SUB_Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.BOTTEN_Sequence3 = nn.Sequential()
        self.BOTTEN_Sequence3.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1*2, (1, 10), stride=(1, 1)))
        self.BOTTEN_Sequence3.add_module('B-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.BOTTEN_Sequence3.add_module('B-ELU1', nn.ELU())
        self.BOTTEN_Sequence3.add_module('B-AVGPool1', nn.AvgPool2d((1, 5)))
        self.BOTTEN_Sequence3.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.SUB_Sequence2 = nn.Sequential()
        self.SUB_Sequence2.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (118, 1), stride=(1, 1)))
        self.SUB_Sequence2.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, False))
        self.SUB_Sequence2.add_module('S-ELU1', nn.ELU())
        self.SUB_Sequence2.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 2))

    def forward(self, source, target):
        glob_loss = 0
        intra_ele_loss = torch.from_numpy(np.array(0)).cuda()
        inter_ele_loss = 0

        source_all = self.SUB_Sequence1(source)
        source_all = self.BOTTEN_Sequence3(source_all)
        source_ele_1 = self.SFC1(source_all)  # 3D全连接层1

        source_ele_2 = self.SFC2(source_ele_1)  # 3D全连接层2

        _s0, _, _s2 = source_ele_2.shape[:3]
        SE2 = source_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]

        source_all = self.SUB_Sequence2(source_ele_2)

        if self.training:
            target_all = self.SUB_Sequence1(target)
            target_all = self.BOTTEN_Sequence3(target_all)
            target_ele_1 = self.SFC1(target_all)  # 3D全连接层1

            target_ele_2 = self.SFC2(target_ele_1)  # 3D全连接层2

            # 源域、目标域用户间电极损失
            _t0, _t1, _t2 = target_ele_2.shape[:3]
            TE2 = target_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
            _t = TE2 - SE2.unsqueeze(1)  # [s2, s2, s0, s1*s3]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0]
            _mt = _t.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
            inter_ele_loss += _mt / (_t2 * _t2) / (_t1 * _t1)

            target_all = self.SUB_Sequence2(target_ele_2)

            # 拉伸成条形处理
            s0, s1, s2, s3 = target_all.shape[:4]  # 读取张量大小
            target_all = target_all.reshape(s0, s1 * s3)  # [28,118*2]
            source_all = source_all.reshape(s0, s1 * s3)  # [28,118*2]

            target_all = self.FC1(target_all)
            source_all = self.FC1(source_all)

            glob_loss += mmd_rbf(source_all, target_all, kernel_mul=5.0, kernel_num=10, fix_sigma=None)  # 整体损失

            output = self.F_FC1(source_all)
        return output, glob_loss, intra_ele_loss, inter_ele_loss

class IE_EDAN_FLOPs(nn.Module):
    def __init__(self, act_func):
        super(IE_EDAN_FLOPs, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112

        self.SUB_Sequence1 = nn.Sequential()
        self.SUB_Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.SUB_Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, False))
        self.SUB_Sequence1.add_module('A-ELU1', nn.ELU())
        self.SUB_Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.SUB_Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.BOTTEN_Sequence3 = nn.Sequential()
        self.BOTTEN_Sequence3.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1*2, (1, 10), stride=(1, 1)))
        self.BOTTEN_Sequence3.add_module('B-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.BOTTEN_Sequence3.add_module('B-ELU1', nn.ELU())
        self.BOTTEN_Sequence3.add_module('B-AVGPool1', nn.AvgPool2d((1, 5)))
        self.BOTTEN_Sequence3.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.SUB_Sequence2 = nn.Sequential()
        self.SUB_Sequence2.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (118, 1), stride=(1, 1)))
        self.SUB_Sequence2.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, False))
        self.SUB_Sequence2.add_module('S-ELU1', nn.ELU())
        self.SUB_Sequence2.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 2))

    def forward(self, source, target):
        glob_loss = 0
        intra_source_ele_loss = 0
        intra_target_ele_loss = 0
        intra_ele_loss = 0
        inter_ele_loss = 0

        source_all = self.SUB_Sequence1(source)
        source_all = self.BOTTEN_Sequence3(source_all)
        source_ele_1 = self.SFC1(source_all)  # 3D全连接层1

        source_ele_2 = self.SFC2(source_ele_1)  # 3D全连接层2

        _s0, _, _s2 = source_ele_2.shape[:3]
        SE2 = source_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]

        source_all = self.SUB_Sequence2(source_ele_2)

        target_all = self.SUB_Sequence1(target)
        target_all = self.BOTTEN_Sequence3(target_all)
        target_ele_1 = self.SFC1(target_all)  # 3D全连接层1

        target_ele_2 = self.SFC2(target_ele_1)  # 3D全连接层2

        # 源域、目标域用户间电极损失
        _t0, _t1, _t2 = target_ele_2.shape[:3]
        TE2 = target_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
        _t = TE2 - SE2.unsqueeze(1)  # [s2, s2, s0, s1*s3]
        _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0]
        _mt = _t.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
        inter_ele_loss += _mt / (_t2 * _t2) / (_t1 * _t1)

        target_all = self.SUB_Sequence2(target_ele_2)

        # 拉伸成条形处理
        s0, s1, s2, s3 = target_all.shape[:4]  # 读取张量大小
        target_all = target_all.reshape(s0, s1 * s3)  # [28,118*2]
        source_all = source_all.reshape(s0, s1 * s3)  # [28,118*2]

        target_all = self.FC1(target_all)
        source_all = self.FC1(source_all)

        glob_loss += mmd_rbf(source_all, target_all, kernel_mul=5.0, kernel_num=10, fix_sigma=None)  # 整体损失

        output = self.F_FC1(source_all)

class EDAN(nn.Module):
    def __init__(self, act_func):
        super(EDAN, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112
        self.ET = torch.from_numpy(np.load('ET.npy', allow_pickle=True)).cuda()  # 读取权重

        self.Sequence1 = nn.Sequential()
        self.Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, False))
        self.Sequence1.add_module('A-ELU1', nn.ReLU())
        self.Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.Sequence2 = nn.Sequential()
        self.Sequence2.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1 * 2, (3, 1), stride=(1, 1)))
        self.Sequence2.add_module('B-ELU1', nn.ReLU())
        self.Sequence2.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.Sequence3 = nn.Sequential()
        self.Sequence3.add_module('C-Conv1', nn.Conv2d(self.F1*2, self.F1*2, (1, 10), stride=(1, 1)))
        self.Sequence3.add_module('C-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.Sequence3.add_module('C-ELU1', nn.ReLU())
        self.Sequence3.add_module('C-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence3.add_module('C-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.Sequence4 = nn.Sequential()
        self.Sequence4.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (118, 1), stride=(1, 1)))
        self.Sequence4.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, False))
        self.Sequence4.add_module('S-ELU1', nn.ReLU())
        self.Sequence4.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 2))

    def forward(self, source, target):
        glob_loss = 0
        intra_source_ele_loss = 0
        intra_target_ele_loss = 0
        intra_ele_loss = 0
        inter_ele_loss = 0

        source_all = self.Sequence1(source)
        source_all = F.pad(source_all, (1, 1, 2, 0))
        source_all = self.Sequence2(source_all)
        source_all = self.Sequence3(source_all)
        source_ele_1 = self.SFC1(source_all)  # 3D全连接层1

        # 源域用户内电极损失 第一层
        _s0, _s1, _s2 = source_ele_1.shape[:3]
        SE1 = source_ele_1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3] [118, 28, 176]

        _s = SE1 - SE1.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
        _s = _s @ _s.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
        _ms1 = _s.mean(dim=-1)
        _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
        _ms3 = _ms2 * self.ET  # 赋予权重
        _ms4 = _ms3.sum()
        intra_source_ele_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1)


        source_ele_2 = self.SFC2(source_ele_1)  # 3D全连接层2


        # 源域用户内电极损失 第二层
        _s0, _, _s2 = source_ele_2.shape[:3]
        SE2 = source_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
        _s = SE2 - SE2.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
        _s = _s @ _s.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
        _ms1 = _s.mean(dim=-1)
        _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
        _ms3 = _ms2 * self.ET  # 赋予权重
        _ms4 = _ms3.sum()

        intra_source_ele_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1)

        source_all = self.Sequence4(source_ele_2)

        if self.training:
            target_all = self.Sequence1(target)
            target_all = F.pad(target_all, (1, 1, 2, 0))
            target_all = self.Sequence2(target_all)
            target_all = self.Sequence3(target_all)
            target_ele_1 = self.SFC1(target_all)  # 3D全连接层1

            # 目标域用户内电极损失 第一层
            _t0, _t1, _t2 = target_ele_1.shape[:3]
            TE1 = target_ele_1.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3] [118, 28, 176]

            _t = TE1 - TE1.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)  # [s2, s2, s0, s0]
            _mt3 = _mt2 * self.ET  # 赋予权重
            _mt4 = _mt3.sum()
            intra_target_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            target_ele_2 = self.SFC2(target_ele_1)  # 3D全连接层2

            # 目标域用户内电极损失 第二层
            _t0, _t1, _t2 = target_ele_2.shape[:3]
            TE2 = target_ele_2.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            _t = TE2 - TE2.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)  # [s2, s2, s0, s0]
            _mt3 = _mt2 * self.ET  # 赋予权重
            _mt4 = _mt3.sum()
            intra_target_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            intra_ele_loss += intra_source_ele_loss + intra_target_ele_loss

            # 源域、目标域用户间电极损失
            _t0, _t1, _t2 = target_ele_2.shape[:3]
            TE2 = target_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
            _t = TE2 - SE2.unsqueeze(1)  # [s2, s2, s0, s1*s3] [118, 118, 28, 176]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0] [118, 118, 176, 176] = [118, 118, 28, 176] * [118, 118, 176, 28]
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)  # [s2, s2, s0, s0]
            _mt3 = _mt2 * self.ET  # 赋予权重
            _mt4 = _mt3.sum()
            inter_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)


            target_all = self.Sequence4(target_ele_2)


            # 拉伸成条形处理
            s0, s1, s2, s3 = target_all.shape[:4]  # 读取张量大小
            target_all = target_all.reshape(s0, s1 * s3)  # [28,118*2]
            source_all = source_all.reshape(s0, s1 * s3)  # [28,118*2]

            target_all = self.FC1(target_all)
            source_all = self.FC1(source_all)

            glob_loss += mmd_rbf(source_all, target_all, kernel_mul=5.0, kernel_num=10, fix_sigma=None)  # 整体损失

            output = self.F_FC1(source_all)

        return output, glob_loss, intra_ele_loss, inter_ele_loss

class EDAN_Flops(nn.Module):
    def __init__(self, act_func):
        super(EDAN_Flops, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112

        self.SUB_Sequence1 = nn.Sequential()
        self.SUB_Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.SUB_Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, False))
        self.SUB_Sequence1.add_module('A-ELU1', nn.ELU())
        self.SUB_Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.SUB_Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.BOTTEN_Sequence3 = nn.Sequential()
        self.BOTTEN_Sequence3.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1*2, (1, 10), stride=(1, 1)))
        self.BOTTEN_Sequence3.add_module('B-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.BOTTEN_Sequence3.add_module('B-ELU1', nn.ELU())
        self.BOTTEN_Sequence3.add_module('B-AVGPool1', nn.AvgPool2d((1, 5)))
        self.BOTTEN_Sequence3.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.SUB_Sequence2 = nn.Sequential()
        self.SUB_Sequence2.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (118, 1), stride=(1, 1)))
        self.SUB_Sequence2.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, False))
        self.SUB_Sequence2.add_module('S-ELU1', nn.ELU())
        self.SUB_Sequence2.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 2))

    def forward(self, source, target):
        glob_loss = 0
        intra_source_ele_loss = 0
        intra_target_ele_loss = 0
        intra_ele_loss = 0
        inter_ele_loss = 0

        source_all = self.SUB_Sequence1(source)
        source_all = self.BOTTEN_Sequence3(source_all)
        source_ele_1 = self.SFC1(source_all)  # 3D全连接层1

        # 源域用户内电极损失
        _s0, _s1, _s2 = source_ele_1.shape[:3]
        SE1 = source_ele_1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
        _s = SE1 - SE1.unsqueeze(1)  # [s2, s2, s0, s1*s3]
        _s = _s @ _s.transpose(-1, -2)  # [s2, s2, s0, s0]
        _ms = _s.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
        intra_source_ele_loss += _ms / (_s2 * _s2) / (_s1 * _s1)

        source_ele_2 = self.SFC2(source_ele_1)  # 3D全连接层2

        _s0, _, _s2 = source_ele_2.shape[:3]
        SE2 = source_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]

        source_all = self.SUB_Sequence2(source_ele_2)

        target_all = self.SUB_Sequence1(target)
        target_all = self.BOTTEN_Sequence3(target_all)
        target_ele_1 = self.SFC1(target_all)  # 3D全连接层1

        # 目标域用户内电极损失
        _t0, _t1, _t2 = target_ele_1.shape[:3]
        TE1 = target_ele_1.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3]
        _t = TE1 - TE1.unsqueeze(1)  # [s2, s2, s0, s1*s3]
        _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0]
        _mt = _t.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
        intra_target_ele_loss += _mt / (_t2 * _t2) / (_t1 * _t1)
        intra_ele_loss += intra_source_ele_loss + intra_target_ele_loss

        target_ele_2 = self.SFC2(target_ele_1)  # 3D全连接层2

        # 源域、目标域用户间电极损失
        _t0, _t1, _t2 = target_ele_2.shape[:3]
        TE2 = target_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
        _t = TE2 - SE2.unsqueeze(1)  # [s2, s2, s0, s1*s3]
        _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0]
        _mt = _t.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
        inter_ele_loss += _mt / (_t2 * _t2) / (_t1 * _t1)

        target_all = self.SUB_Sequence2(target_ele_2)

        # 拉伸成条形处理
        s0, s1, s2, s3 = target_all.shape[:4]  # 读取张量大小
        target_all = target_all.reshape(s0, s1 * s3)  # [28,118*2]
        source_all = source_all.reshape(s0, s1 * s3)  # [28,118*2]

        target_all = self.FC1(target_all)
        source_all = self.FC1(source_all)

        glob_loss += mmd_rbf(source_all, target_all, kernel_mul=5.0, kernel_num=10, fix_sigma=None)  # 整体损失

        output = self.F_FC1(source_all)

class PSDAN(nn.Module):
    def __init__(self, act_func):
        super(PSDAN, self).__init__()
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
        self.ELE_Sequence1.add_module('E1-Conv1', nn.Conv2d(self.F1, self.F1*2, (15, 1), stride=(10, 1)))
        self.ELE_Sequence1.add_module('E1-Norm1', nn.BatchNorm2d(self.F1*2, False))

        '''电极融合层-2'''
        self.ELE_Sequence2 = nn.Sequential()
        self.ELE_Sequence2.add_module('E2-Conv1', nn.Conv2d(self.F1*2, self.F1*2, (5, 1), stride=(5, 1)))
        self.ELE_Sequence2.add_module('E2-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.ELE_Sequence2.add_module('E2-ELU1', nn.ELU())
        self.ELE_Sequence2.add_module('E2-AVGPool1', nn.AvgPool2d((1, 5)))
        self.ELE_Sequence2.add_module('E2-Drop1', nn.Dropout(p=0.25))

        '''进一步进行时间卷积和降采样'''
        self.TEMP_Sequence2 = nn.Sequential()
        self.TEMP_Sequence2.add_module('B-Conv1', nn.Conv2d(self.F1*2, self.F1*2, (1, 10), stride=(1, 1)))
        self.TEMP_Sequence2.add_module('B-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.TEMP_Sequence2.add_module('B-ELU1', nn.ELU())
        self.TEMP_Sequence2.add_module('B-AVGPool1', nn.AvgPool2d((1, 5)))
        self.TEMP_Sequence2.add_module('B-Drop1', nn.Dropout(p=0.25))

        '''电极融合层-3'''
        self.ELE_Sequence3 = nn.Sequential()
        self.ELE_Sequence3.add_module('E2-Conv1', nn.Conv2d(self.F1*2, self.F1*2, (2, 1), stride=(2, 1)))
        self.ELE_Sequence3.add_module('E2-Norm1', nn.BatchNorm2d(self.F1*2, False))
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
        inter_glob_loss = 0  # 数据整体分布距离
        inter_location_loss_ST = 0
        inter_location_loss_LR = 0
        intra_location_loss = 0

        source_L = self.TEMP_Sequence1(source_L)

        for location in range(0, 4):  # 依次读取区域
            location_size = 15  # 电极区域大小
            location_stride = 5  # 电极区域步进步长
            if location == 0:
                start_ele = 0  # 区域起始电极编号
            else:
                start_ele = location * location_size - location_stride
            end_ele = start_ele + location_size  # 区域终止电极编号
            # 计算电极间损失
            s1 = source_L[:, :, start_ele:end_ele, :]  # 提取相应区域的电极
            ET_L1 = self.ET_L[start_ele:end_ele, start_ele:end_ele]  # 提取相应区域的电极
            # 源域用户内各区域内电极损失
            _s0, _s1, _s2 = s1.shape[:3]
            SE1 = s1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            _s = SE1 - SE1.unsqueeze(1)
            _s = _s @ _s.transpose(-1, -2)
            _ms1 = _s.mean(dim=-1)
            _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
            _ms3 = _ms2 * ET_L1
            _ms4 = _ms3.sum()

            intra_location_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1) / 10

        source_R = self.TEMP_Sequence1(source_R)

        for location in range(0, 4):  # 依次读取区域
            location_size = 15  # 电极区域大小
            location_stride = 5  # 电极区域步进步长
            if location == 0:
                start_ele = 0  # 区域起始电极编号
            else:
                start_ele = location * location_size - location_stride
            end_ele = start_ele + location_size  # 区域终止电极编号
            # 计算电极间损失
            s1 = source_R[:, :, start_ele:end_ele, :]  # 提取相应区域的电极
            ET_R1 = self.ET_R[start_ele:end_ele, start_ele:end_ele]  # 提取相应区域的电极
            # 源域用户内各区域内电极损失
            _s0, _s1, _s2 = s1.shape[:3]
            SE1 = s1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
            _s = SE1 - SE1.unsqueeze(1)
            _s = _s @ _s.transpose(-1, -2)
            _ms1 = _s.mean(dim=-1)
            _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
            _ms3 = _ms2 * ET_R1
            _ms4 = _ms3.sum()

            intra_location_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1) / 10

        source_ele_L = self.ELE_Sequence1(source_L)
        source_ele_R = self.ELE_Sequence1(source_R)

        if self.training:
            target_L = self.TEMP_Sequence1(target_L)

            for location in range(0, 4):  # 依次读取区域
                location_size = 15  # 电极区域大小
                location_stride = 5  # 电极区域步进步长
                if location == 0:
                    start_ele = 0  # 区域起始电极编号
                else:
                    start_ele = location * location_size - location_stride
                end_ele = start_ele + location_size  # 区域终止电极编号
                # 计算电极间损失
                s1 = target_L[:, :, start_ele:end_ele, :]  # 提取相应区域的电极
                ET_L1 = self.ET_L[start_ele:end_ele, start_ele:end_ele]  # 提取相应区域的电极
                # 源域用户内各区域内电极损失
                _s0, _s1, _s2 = s1.shape[:3]
                SE1 = s1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
                _s = SE1 - SE1.unsqueeze(1)
                _s = _s @ _s.transpose(-1, -2)
                _ms1 = _s.mean(dim=-1)
                _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
                _ms3 = _ms2 * ET_L1
                _ms4 = _ms3.sum()

                intra_location_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1) / 10

            target_R = self.TEMP_Sequence1(target_R)

            for location in range(0, 4):  # 依次读取区域
                location_size = 15  # 电极区域大小
                location_stride = 5  # 电极区域步进步长
                if location == 0:
                    start_ele = 0  # 区域起始电极编号
                else:
                    start_ele = location * location_size - location_stride
                end_ele = start_ele + location_size  # 区域终止电极编号
                # 计算电极间损失
                s1 = target_R[:, :, start_ele:end_ele, :]  # 提取相应区域的电极
                ET_R1 = self.ET_R[start_ele:end_ele, start_ele:end_ele]  # 提取相应区域的电极
                # 源域用户内各区域内电极损失
                _s0, _s1, _s2 = s1.shape[:3]
                SE1 = s1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3] [118, 28, 176]
                _s = SE1 - SE1.unsqueeze(1)
                _s = _s @ _s.transpose(-1, -2)
                _ms1 = _s.mean(dim=-1)
                _ms2 = _ms1.mean(dim=-1)  # [s2, s2, s0, s0]
                _ms3 = _ms2 * ET_R1
                _ms4 = _ms3.sum()
                intra_location_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1) / 10

            target_ele_L = self.ELE_Sequence1(target_L)
            target_ele_R = self.ELE_Sequence1(target_R)

            s1, s2, s3, s4 = source_ele_L.shape[:4]  # 读取数据各维度大小
            ele_ma = source_ele_L - target_ele_L
            ele_dis_ma = torch.matmul(ele_ma, ele_ma.transpose(-1, -2))  # 电极间距离矩阵 [B, F, C, C]
            co_ele_dis_ma = torch.matmul(torch.mul(ele_dis_ma, self.fe_1), self.fe_2)  # [B, F, C, 1]
            co_ele_dis = torch.mean(co_ele_dis_ma, dim=2)  # [B, F, 1]

            inter_location_loss_ST += co_ele_dis.sum() / (s1 * s2 * s3 * 100)

            s1, s2, s3, s4 = source_ele_L.shape[:4]  # 读取数据各维度大小
            ele_ma = source_ele_R - target_ele_R
            ele_dis_ma = torch.matmul(ele_ma, ele_ma.transpose(-1, -2))  # 电极间距离矩阵 [B, F, C, C]
            co_ele_dis_ma = torch.matmul(torch.mul(ele_dis_ma, self.fe_1), self.fe_2)  # [B, F, C, 1]
            co_ele_dis = torch.mean(co_ele_dis_ma, dim=2)  # [B, F, 1]

            inter_location_loss_ST += co_ele_dis.sum() / (s1 * s2 * s3 * 100)

            target_ele_L = self.ELE_Sequence2(target_ele_L)
            source_ele_L = self.ELE_Sequence2(source_ele_L)

            target_ele_R = self.ELE_Sequence2(target_ele_R)
            source_ele_R = self.ELE_Sequence2(source_ele_R)

            s0, s1, _, s2 = target_ele_L.shape[:4]  # 读取张量大小
            h1 = target_ele_L.reshape(s0, s1 * s2)  # [28,1024]
            h2 = source_ele_L.reshape(s0, s1 * s2)
            inter_location_loss_LR += mmd_linear(h1, h2)

            h1 = target_ele_R.reshape(s0, s1 * s2)  # [28,1024]
            h2 = source_ele_R.reshape(s0, s1 * s2)
            inter_location_loss_LR += mmd_linear(h1, h2)

            source_L = self.TEMP_Sequence2(source_ele_L)
            target_L = self.TEMP_Sequence2(target_ele_L)

            source_R = self.TEMP_Sequence2(source_ele_R)
            target_R = self.TEMP_Sequence2(target_ele_R)

            source_LR = torch.cat((source_L, source_R), -2)  # 拼接左右脑数据
            source_LR = self.ELE_Sequence3(source_LR)

            target_LR = torch.cat((target_L, target_R), -2)  # 拼接左右脑数据
            target_LR = self.ELE_Sequence3(target_LR)

            # 拉伸成条形处理
            s0, s1, _, s2 = target_L.shape[:4]  # 读取张量大小
            source_LR = source_LR.reshape(s0, s1 * s2)  # [28,118*2]
            source_LR = self.FC1(source_LR)

            target_LR = target_LR.reshape(s0, s1 * s2)  # [28,118*2]
            target_LR = self.FC1(target_LR)

            inter_glob_loss += mmd_linear(source_LR, target_LR)

            output = self.F_FC1(source_LR)

        return output, intra_location_loss, inter_location_loss_LR, inter_location_loss_ST, inter_glob_loss

'''最大均值差异计算函数'''


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


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
    if args.model == 'Net1':
        return {
        'elu': [EEGNet()]
        # 'relu': [EEGNet('relu')],
        # 'lrelu': [EEGNet('lrelu')],
        }

    elif args.model == 'Net2':
        return {
        'elu': [DeepConvNet()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net3':
        return {
        'elu': [DDC()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net4':
        return {
        'elu': [DeepCoral()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net7':
        return {
        'elu': [IA_EDAN('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net8':
        return {
        'elu': [IE_EDAN('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net9':
        return {
        'elu': [EDAN('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net10':
        return {
        'elu': [PSDAN('elu')]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net1F':
        return {
        'elu': [EEGNet_FLOPs()]
        # 'relu': [DeepConvNet('relu')]
        # 'lrelu': [DeepConvNet('lrelu')],
        }
    elif args.model == 'Net2F':
        return {
        'elu': [DeepConvNet_FLOPs()]
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