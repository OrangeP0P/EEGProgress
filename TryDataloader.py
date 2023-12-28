from CrossValidation import split_data
# from DataLoader import read_model_data
# from DataLoader import read_test_data
# import scipy.io as scio
# import numpy as np
#
# data_size = 280  # 数据集样本数(包含训练集和验证集)
# te_num = 30  # 测试集样本数
# split_num = 10  # 将模型集分割的份数
# Basic_folder = 'Datasets_Transfer_Task/'  # 服务器端根目录
# Current_Datasets = '4-TL_Task_Python_Data/'  # 数据集路径
# tr, va = 9, 1  # 训练集、测试集比例
# subject_1 = 'aa'
# subject_2 = 'al'
#
# train_list, validation_list, test_list = split_data(split_num, data_size, te_num)
#
# train_list = train_list[0]
# validation_list = validation_list[0]
#
# # 读取训练模型数据
# train_source_data, train_source_label, train_target_data, train_target_label, validation_target_data, validation_target_label,\
# va_num = read_model_data(Basic_folder, Current_Datasets, subject_1, subject_2,
#                          data_size, tr, va, te_num, train_list, validation_list)
#
# # 读取测试数据
# test_target_data, test_target_label = read_test_data(Basic_folder, Current_Datasets,
#                                                      subject_2, data_size, te_num, test_list)
import random
from random import shuffle

split_num = 10
data_size = 280

datasets_list_source = list(range(0, data_size))  # 源域数据集标签序号
datasets_list_target = list(range(0, data_size))  # 目标域数据集标签序号
shuffle(datasets_list_source)  # 重排序：打乱源域数据集标签序号
shuffle(datasets_list_target)  # 重排序：打乱目标域数据集标签序号
_model_source_list = []  # 临时源域数据
model_source_list = []  # 模型集重加工，供后面进行每个Cross-Mission(折)选择
_model_target_list = []  # 临时源域数据
model_target_list = []  # 模型集重加工，供后面进行每个Cross-Mission(折)选择
model_size = data_size  # 模型集样本数

for i in range(0, data_size):  # 将原数据集分割成两部分：模型集（训练集+验证集） 和 测试集
    _model_source_list.append(datasets_list_source[i])  # 保存训练源域数据
    _model_target_list.append(datasets_list_target[i])  # 保存训练目标域数据和验证集数据
for i in range(0, model_size, int(model_size / split_num)):
    model_source_list.append(_model_source_list[i:i + int(model_size / split_num)])  # 重加工源域训练集
    model_target_list.append(_model_target_list[i:i + int(model_size / split_num)])  # 重加工目标域训练集、验证集

train_source_list = list()
train_target_list = list()
validation_target_list = list()

# 目标域数据集划分为训练集、测试集
for task in range(0, split_num):
    _train_source_list = list()
    _train_target_list = list()
    _validation_target_list = list()

    for sa in range(0, split_num):
        _train_source_list = _train_source_list + model_source_list[sa]
    train_source_list.append(_train_source_list)

    if task == split_num - 1:
        _validation_target_list = model_target_list[9]
        validation_target_list.append(_validation_target_list)
        for sp in range(0, split_num - 1):
            _train_target_list = _train_target_list + model_target_list[sp]
        train_target_list.append(_train_target_list)
    if task != split_num - 1:
        _validation_target_list = model_target_list[task]
        validation_target_list.append(_validation_target_list)
        for sp in range(0, split_num):
            if sp != task:
                _train_target_list = _train_target_list + model_target_list[sp]
        train_target_list.append(_train_target_list)
