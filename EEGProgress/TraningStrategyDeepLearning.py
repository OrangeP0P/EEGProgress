global EX

def tranning_strategy(EX):
    if EX == 1:  # no loss: EEGNet
        Current_Datasets = 'a22_RawData/'  # 数据集路径：原始数据集经过FIR滤波，不进行重排，不剔除参考电极
        Net_number = 'EEGNet'  # 选取网络类型：PSCNN 数据不进行重排，只区分左右脑

    if EX == 2:  # no loss: EEGNet
        Current_Datasets = 'a19_SpRaw/'  # 数据集路径：原始数据集经过FIR滤波，不进行重排，不剔除参考电极
        Net_number = 'HDAN_2'  # 选取网络类型：PSCNN 数据不进行重排，只区分左右脑

    return Current_Datasets, Net_number
