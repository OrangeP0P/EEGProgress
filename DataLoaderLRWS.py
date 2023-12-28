import numpy as np
import scipy.io as scio


def read_model_data(Basic_folder, Current_Datasets, subject_1, data_size,
                    tr, va, train_target_list, validation_target_list):

    tr_va_num = data_size  # 训练集 和 验证集 样本大小之和

    sdn_L = 'data_L_' + subject_1
    sdp_L = Basic_folder + Current_Datasets + sdn_L + '.mat'
    sdn_R = 'data_R_' + subject_1
    sdp_R = Basic_folder + Current_Datasets + sdn_R + '.mat'
    sln = 'label_' + subject_1
    slp = Basic_folder + Current_Datasets + sln + '.mat'

    tr_num = int(tr_va_num / (tr + va) * tr)  # 训练集数据大小
    va_num = int(tr_va_num - tr_num)  # 验证集数据大小

    # 读取 源域、目标域 数据集和标签
    train_data_L = scio.loadmat(sdp_L)[sdn_L]
    train_data_R = scio.loadmat(sdp_R)[sdn_R]
    train_label = scio.loadmat(slp)[sln]
    validation_data_L = scio.loadmat(sdp_L)[sdn_L]
    validation_data_R = scio.loadmat(sdp_R)[sdn_R]
    validation_label = scio.loadmat(slp)[sln]


    # 构建训练集、验证集、测试集 数据和标签
    train_data_L = train_data_L[train_target_list, :, :]
    train_data_R = train_data_R[train_target_list, :, :]
    _train_label = train_label[train_target_list, :]

    validation_data_L = validation_data_L[validation_target_list, :]
    validation_data_R = validation_data_R[validation_target_list, :]
    _validation_label = validation_label[validation_target_list, :]

    train_label = np.array(range(0, len(_train_label)))
    for i in range(0, len(_train_label)):
        train_label[i] = _train_label[i]

    validation_label = np.array(range(0, len(_validation_label)))
    for i in range(0, len(_validation_label)):
        validation_label[i] = _validation_label[i]

    train_label = train_label - 1
    validation_label = validation_label - 1

    train_data_L = np.transpose(np.expand_dims(train_data_L, axis=1), (0, 1, 3, 2))
    validation_data_L = np.transpose(np.expand_dims(validation_data_L, axis=1), (0, 1, 3, 2))
    train_data_R = np.transpose(np.expand_dims(train_data_R, axis=1), (0, 1, 3, 2))
    validation_data_R = np.transpose(np.expand_dims(validation_data_R, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data_L))
    train_data_L[mask] = np.nanmean(train_data_L)
    mask = np.where(np.isnan(validation_data_L))
    validation_data_L[mask] = np.nanmean(validation_data_L)
    mask = np.where(np.isnan(train_data_R))
    train_data_R[mask] = np.nanmean(train_data_R)
    mask = np.where(np.isnan(validation_data_R))
    validation_data_R[mask] = np.nanmean(validation_data_R)

    return train_data_L, train_data_R, train_label, \
           validation_data_L, validation_data_R, validation_label, va_num