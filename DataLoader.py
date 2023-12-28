import numpy as np
import scipy.io as scio


def read_model_data(Basic_folder, Current_Datasets, subject_1, subject_2,
                  data_size, tr, va, train_list, target_list, validation_list):

    tr_va_num = data_size  # 训练集 和 验证集 样本大小之和
    sdn = 'data_' + subject_1
    sdp = Basic_folder + Current_Datasets + sdn + '.mat'
    sln = 'label_' + subject_1
    slp = Basic_folder + Current_Datasets + sln + '.mat'
    tdn = 'data_' + subject_2
    tdp = Basic_folder + Current_Datasets + tdn + '.mat'
    tln = 'label_' + subject_2
    tlp = Basic_folder + Current_Datasets + tln + '.mat'
    tr_num = int(tr_va_num / (tr + va) * tr)  # 训练集数据大小
    va_num = int(tr_va_num - tr_num)  # 验证集数据大小

    # 读取 源域、目标域 数据集和标签
    source_data = scio.loadmat(sdp)[sdn]
    source_label = scio.loadmat(slp)[sln]
    target_data = scio.loadmat(tdp)[tdn]
    target_label = scio.loadmat(tlp)[tln]

    # 构建训练集、验证集、测试集 数据和标签
    train_source_data = source_data[train_list, :, :]
    _train_source_label = source_label[train_list, :]
    train_target_data = target_data[target_list, :]
    _train_target_label = target_label[target_list, :]
    validation_target_data = target_data[validation_list, :]
    _validation_target_label = target_label[validation_list, :]

    train_source_label = np.array(range(0, len(_train_source_label)))
    for i in range(0, len(_train_source_label)):
        train_source_label[i] = _train_source_label[i]
    train_target_label = np.array(range(0, len(_train_target_label)))
    for i in range(0, len(_train_target_label)):
        train_target_label[i] = _train_target_label[i]

    validation_target_label = np.array(range(0, len(_validation_target_label)))
    for i in range(0, len(_validation_target_label)):
        validation_target_label[i] = _validation_target_label[i]

    train_source_label = train_source_label - 1
    train_target_label = train_target_label - 1
    validation_target_label = validation_target_label - 1

    train_source_data = np.transpose(np.expand_dims(train_source_data, axis=1), (0, 1, 3, 2))
    train_target_data = np.transpose(np.expand_dims(train_target_data, axis=1), (0, 1, 3, 2))
    validation_target_data = np.transpose(np.expand_dims(validation_target_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_source_data))
    train_source_data[mask] = np.nanmean(train_source_data)
    mask = np.where(np.isnan(train_target_data))
    train_target_data[mask] = np.nanmean(train_target_data)
    mask = np.where(np.isnan(validation_target_data))
    validation_target_data[mask] = np.nanmean(validation_target_data)

    return train_source_data, train_source_label, train_target_data, train_target_label, \
           validation_target_data, validation_target_label, va_num