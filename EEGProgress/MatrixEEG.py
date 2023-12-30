import numpy as np
import torch
import torch.nn as nn


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


class EEGNet_Modified(nn.Module):
    def __init__(self, act_func):
        super(EEGNet_Modified, self).__init__()
        # ??shape?(Batch_size, 1, C = num of channels, T = num of time points)
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 10
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112

        # Conv2D  filters = F1
        self.firstConv = nn.Sequential()
        self.firstConv.add_module('conv1', nn.Conv2d(1, self.F1, (1, self.kernel_size)))
        self.firstConv.add_module('norm1', nn.BatchNorm2d(self.F1, False))

        # Depthwise Conv2D  [filters] = F1 * D  [kernal_size] = (C = num of channels,1)
        self.ele_depthwiseConv = nn.Sequential()
        self.ele_depthwiseConv.add_module('conv2', nn.Conv2d(self.F1, self.F1 * self.D, (1, self.kernel_size)))
        self.ele_depthwiseConv.add_module('norm2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.ele_depthwiseConv.add_module('f_ELU2', nn.ELU())
        self.ele_depthwiseConv.add_module('pool1', nn.AvgPool2d((1, 4)))
        self.ele_depthwiseConv.add_module('drop1', nn.Dropout(p=0.25))

        # Depthwise Conv2D  [filters] = F1 * D  [kernal_size] = (C = num of channels,1)
        self.depthwiseConv = nn.Sequential()
        self.depthwiseConv.add_module('conv2', nn.Conv2d(self.F1, self.F1 * self.D, (self.channel, 1)))
        self.depthwiseConv.add_module('norm2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.depthwiseConv.add_module('f_ELU2', nn.ELU())
        self.depthwiseConv.add_module('pool1', nn.AvgPool2d((1, 4)))
        self.depthwiseConv.add_module('drop1', nn.Dropout(p=0.25))

        # Separable Conv2D [filters] = F2  [kenal_size] = (1,16)
        self.separableConv = nn.Sequential()
        self.separableConv.add_module('conv3', nn.Conv2d(self.F1 * self.D, self.F2, (1, int(self.kernel_size / 4))))
        self.separableConv.add_module('norm3', nn.BatchNorm2d(self.F2))
        self.separableConv.add_module('f_ELU2', nn.ELU())
        self.separableConv.add_module('pool2', nn.AvgPool2d((1, 8)))
        self.separableConv.add_module('drop2', nn.Dropout(p=0.25))

        # first ele classifier
        self.first_ele_classifier = nn.Sequential()
        self.first_ele_classifier.add_module('c_fc1', nn.Linear(self.ELE_feature, 256))
        self.first_ele_classifier.add_module('c_bn1', nn.BatchNorm1d(256))

        # ele_botten_layer
        self.ele_botten_layer = nn.Sequential()
        self.ele_botten_layer.add_module('c_fc1', nn.Linear(256, 128))
        self.ele_botten_layer.add_module('c_bn1', nn.BatchNorm1d(128))

        # first classifier
        self.first_classifier = nn.Sequential()
        self.first_classifier.add_module('c_fc1', nn.Linear(self.ALL_feature, 256))
        self.first_classifier.add_module('c_bn1', nn.BatchNorm1d(256))

        # botten layer
        self.botten_classify = nn.Sequential()
        self.botten_classify.add_module('c_fc2', nn.Linear(128, 64))
        self.botten_classify.add_module('c_bn2', nn.BatchNorm1d(64))

        # final layer
        self.final_classify = nn.Sequential(nn.Linear(64, 2))

    def forward(self, source, target):
        source_ele_loss = 0
        source = self.firstConv(source)
        source_ele = self.ele_depthwiseConv(source)
        source_ele = self.separableConv(source_ele)

        # SE1 = torch.tensor(np.zeros([source_ele.size(2),source_ele.size(0),128]))
        # SE2 = torch.tensor(np.zeros([source_ele.size(2),source_ele.size(0),128]))

        _s0, _, _s2 = source_ele.shape[:3]

        SE1 = source_ele.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)  # [s2, s0, s1*s3]
        SE1 = self.first_ele_classifier(SE1)
        SE1 = self.ele_botten_layer(SE1)
        # for i in range(0,source_ele.size(2)):
        #     se1 = source_ele[:, :, i, :]
        #     se1 = se1.contiguous().view(se1.size(0), -1)    # [s0, s1*s3]
        #     se1 = self.first_ele_classifier(se1)    # [s0, 256]
        #     se1 = self.ele_botten_layer(se1)
        #     SE1[i,:,:] = se1
        #     SE2[i,:,:] = se1

        _t = SE1 - SE1.unsqueeze(1)  # [s2, s2, s0, s1*s3]
        _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0]
        _mt = _t.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
        source_ele_loss = _mt / (_s2 * _s2)
        # for i in range(0,SE1.size(0)):
        #     for j in range(0, SE2.size(0)):
        #         if i != j:
        #             source_ele_loss += mmd_linear(SE1[i],SE2[j])/(SE1.size(0)*SE2.size(0))
        #         else:
        #             continue

        source = self.depthwiseConv(source)
        source = self.separableConv(source)
        source = source.view(source.size(0), -1)  # ????????
        source = self.first_classifier(source)
        source = self.ele_botten_layer(source)
        source = self.botten_classify(source)

        mmd_loss = 0

        if self.training:
            st_ele_loss = 0
            target_ele_loss = 0
            target = self.firstConv(target)
            target_ele = self.ele_depthwiseConv(target)
            target_ele = self.separableConv(target_ele)

            _t0, _, _t2 = target_ele.shape[:3]
            SE2 = target_ele.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)  # [s2, s0, s1*s3]
            SE2 = self.first_ele_classifier(SE2)
            SE2 = self.ele_botten_layer(SE2)
            # SE1 = torch.tensor(np.zeros([target_ele.size(2), target_ele.size(0), 128]))
            # SE2 = torch.tensor(np.zeros([target_ele.size(2), target_ele.size(0), 128]))
            # for i in range(0, target_ele.size(2)):
            #     se1 = target_ele[:, :, i, :]
            #     se1 = se1.contiguous().view(se1.size(0), -1)  # ?1??????
            #     se1 = self.first_ele_classifier(se1)
            #     se1 = self.ele_botten_layer(se1)
            #     SE1[i, :, :] = se1
            #     SE2[i, :, :] = se1
            _t = SE2 - SE2.unsqueeze(1)  # [s2, s2, s0, s1*s3]
            _t = _t @ _t.transpose(-1, -2)  # [s2, s2, s0, s0]
            _mt = _t.mean(dim=-1).mean(dim=-1).sum()  # [s2, s2, s0, s0]
            target_ele_loss = _mt / (_t2 * _t2)

            # for i in range(0, SE1.size(0)):
            #     for j in range(0, SE2.size(0)):
            #         if i != j:
            #             target_ele_loss += mmd_linear(SE1[i], SE2[j]) / (SE1.size(0) * SE2.size(0))
            #         else:
            #             continue

            s_t_ele_loss = target_ele_loss + source_ele_loss

            # ??????????, ???????????  SE1, SE2????????

            # SE1 = torch.tensor(np.zeros([source_ele.size(2), source_ele.size(0), 128]))
            # SE2 = torch.tensor(np.zeros([target_ele.size(2), target_ele.size(0), 128]))

            # for i in range(0, source_ele.size(2)):
            #     se1 = source_ele[:, :, i, :]
            #     se1 = se1.contiguous().view(se1.size(0), -1)  # ?1??????
            #     se1 = self.first_ele_classifier(se1)
            #     se1 = self.ele_botten_layer(se1)
            #     SE1[i, :, :] = se1
            # for i in range(0, target_ele.size(2)):
            #     se1 = target_ele[:, :, i, :]
            #     se1 = se1.contiguous().view(se1.size(0), -1)  # ?1??????
            #     se1 = self.first_ele_classifier(se1)
            #     se1 = self.ele_botten_layer(se1)
            #     SE2[i, :, :] = se1

            _t = SE1 - SE2.unsqueeze(1)
            _t = _t @ _t.transpose(-1, -2)
            _mt = _t.mean(dim=-1).mean(dim=-1).sum()
            st_ele_loss = _mt / (_s2 * _t2)
            # for i in range(0, SE1.size(0)):
            #     for j in range(0, SE2.size(0)):
            #         if i != j:
            #             st_ele_loss += mmd_linear(SE1[i],SE2[j]) / (SE1.size(0) * SE2.size(0))
            #         else:
            #             continue

            target = self.depthwiseConv(target)
            target = self.separableConv(target)
            target = target.view(target.size(0), -1)
            target = self.first_classifier(target)
            target = self.ele_botten_layer(target)
            target = self.botten_classify(target)
            mmd_loss += mmd_linear(source, target)

        out = self.final_classify(source)
        return out, mmd_loss, st_ele_loss, s_t_ele_loss