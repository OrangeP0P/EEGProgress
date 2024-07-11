from torch.autograd import Variable
import numpy as np
import torch
import torch.utils.data as Data
from EEGProgressModel import choose_net
from argparse import ArgumentParser
from DataLoaderLR import read_model_data
from EEGProgressModel import handle_param
from ProgressBar import progress_bar
from thop import profile
from thop import clever_format
import RandomSeed
import math
import time
import csv
import sys
import pandas as pd
import math

Seed_ACC = []  # 统计不同种子数时模型的最终准确率
Net_number = 'EEGProgress'  # 选取网络类型
Current_Datasets = 'a20_SpPe/'  # 数据集路径
Epoch = 150  # 训练轮数
CSV_name = 'a20_SpPe' + '-ACC'  # 设置表格名称 每次开始时设置

print('Training Model：', Net_number, '\n')
start_seed = 2023  # 起始种子数

torch.set_num_threads(1)  # 限制cpu线程数，解决占用率过高问题
SEED = start_seed
RandomSeed.setup_seed(SEED)  # 设置固定种子数 2023年
start = time.time()  # 记下开始时刻
data_size = 280  # 数据集样本数(包含训练集和验证集)
split_num = 10  # 将模型集分割的份数
tr, va = 9, 1  # 训练集、验证集比例

Basic_folder = 'Datasets_Transfer_Task/'  # 服务器端根目录

Batch_size_list = [28] * 20  # batch_size
subject_list = ['aa', 'al', 'av', 'aw', 'ay']  # 用户列表

# 固定交叉验证：便于调试
train_source_list = np.load("train_source_list.npy", allow_pickle=True)
train_target_list = np.load("train_target_list.npy", allow_pickle=True)
validation_target_list = np.load("validation_target_list.npy", allow_pickle=True)

Cross_Aver_ACC = []  # 交叉验证准确率
es_time = 0  # 预估程序总剩余时间
es_time_2 = 0  # 预估一 cross剩余时间
Start_Task = 1  # 设定起始 Task
Terminal_Task = 20  # 设定结束 Task
Start_Cross = 1  # 设定起始 Cross
Terminal_Cross = 10  # 设定结束 Cross

Result_Ans = np.zeros([Terminal_Cross-Start_Cross+1, Terminal_Task-Start_Task+1,
                       Epoch, math.ceil(28/Batch_size_list[1])+2])
Task_ACC = np.zeros([Terminal_Task-Start_Task+1, Terminal_Cross-Start_Cross+1])  # 用来存取不同 Task（用户a to 用户b）不同 Cross 的准确率
Task_LOSS = np.zeros([Terminal_Task-Start_Task+1, Terminal_Cross-Start_Cross+1])
VA_Correct = np.zeros([Terminal_Task-Start_Task+1, Terminal_Cross-Start_Cross+1])  # 存放各task分类正确的样本数

# 开始交叉验证
for Cross_Mission in range(Start_Cross-1, Terminal_Cross):
    Max_ACC = []  # 统计每轮训练最大准确率
    Task = 1  # 计数当前Task任务轮次
    # 选择subject_1, subject_2
    for k in range(0, len(subject_list)):
        for g in range(0, len(subject_list)):
            if subject_list[k] != subject_list[g] and Start_Task <= Task <= Terminal_Task:
                subject_1 = subject_list[k]
                subject_2 = subject_list[g]
                # print('source subject:', subject_1, 'to', ' target subject', subject_2, '\n')
                # 读取网络参数
                parser = ArgumentParser()
                parser.add_argument("-b", "--batch", help="batch size", type=int, default=Batch_size_list[Task-1])
                parser.add_argument("-lr", "--learning-rate", help="learning rate", type=float, default=1e-3)
                parser.add_argument("-ep", "--epochs", help="your training target", type=int, default=Epoch)
                parser.add_argument("-opt", "--optimizer", help="adam | rmsp", type=str, default='adam')
                parser.add_argument("-lf", "--loss-function", help="loss function", type=str, default='CrossEntropy')
                parser.add_argument("-act", "--activation-function", help="elu | relu | lrelu", type=str, default='relu')
                parser.add_argument("-m", "--model", help="eeg | dcn", type=str, default=Net_number)
                parser.add_argument("-load", "--load", help="your pkl file path", type=str, default='')
                args = parser.parse_args()

                # 开始训练网络
                # 读取训练模型数据
                train_source_data_L, train_source_data_R, train_source_label, train_target_data_L, train_target_data_R, \
                train_target_label, validation_target_data_L, validation_target_data_R, validation_target_label, va_num \
                    = read_model_data(Basic_folder, Current_Datasets,
                                         subject_1, subject_2, data_size, tr, va,
                                         train_source_list[Cross_Mission],
                                         train_target_list[Cross_Mission],
                                         validation_target_list[Cross_Mission])

                source_data_L = Data.TensorDataset(torch.from_numpy(train_source_data_L.astype(np.float32)),
                                                 torch.from_numpy(train_source_label.astype(np.float32)))
                target_data_L = Data.TensorDataset(torch.from_numpy(train_target_data_L.astype(np.float32)),
                                                 torch.from_numpy(train_target_label.astype(np.float32)))

                source_data_R = Data.TensorDataset(torch.from_numpy(train_source_data_R.astype(np.float32)),
                                                   torch.from_numpy(train_source_label.astype(np.float32)))
                target_data_R = Data.TensorDataset(torch.from_numpy(train_target_data_R.astype(np.float32)),
                                                   torch.from_numpy(train_target_label.astype(np.float32)))

                validation_data = Data.TensorDataset(torch.from_numpy(validation_target_data_L.astype(np.float32)),
                                                       torch.from_numpy(validation_target_data_R.astype(np.float32)),
                                                       torch.from_numpy(validation_target_label.astype(np.float32)))

                source_loader_L = Data.DataLoader(dataset=source_data_L, batch_size=args.batch, shuffle=True, drop_last=False)
                target_loader_L = Data.DataLoader(dataset=target_data_L, batch_size=args.batch, shuffle=True, drop_last=False)

                source_loader_R = Data.DataLoader(dataset=source_data_R, batch_size=args.batch, shuffle=True, drop_last=False)
                target_loader_R = Data.DataLoader(dataset=target_data_R, batch_size=args.batch, shuffle=True, drop_last=False)

                validation_loader = Data.DataLoader(dataset=validation_data, batch_size=args.batch, shuffle=True)

                epoch_num = args.epochs  # 从网络参数中读取epoch的数目
                net_dict = choose_net(args)  # 选取网络

                # 初始化损失矩阵，用来存储损失
                LOSS = []
                CLF_LOSS = []
                GLOBAL_LOSS = []
                INTRA_LOCATION_LOSS = []
                INTER_LOCATION_LOSS_LR = []
                INTER_LOCATION_LOSS_ST = []

                VA_LOSS = []
                VA_ST_GLOB_LOSS = []
                VA_CLF_LOSS = []
                VA_ST_ELE_LOSS = []
                # 训练网络
                # net[0]:model, net[1]:optimizer, net[2]:loss_function
                for key, net in net_dict.items():
                    optimizer, loss_func = handle_param(args, net[0])
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150, 250, 350, 450,
                                                                                            550, 650, 750, 850, 950],
                                                                     gamma=0.7)
                    net.extend([optimizer, loss_func, scheduler])
                    total_loss = 0
                    epoch_start = time.time()

                # '''计算模型参数量大小'''
                if Cross_Mission == 0 and Task == 1:
                    input_source_L = torch.from_numpy(train_source_data_L.astype(np.float32))
                    input_source_R = torch.from_numpy(train_source_data_R.astype(np.float32))
                    flops, t_params = profile(net[0], inputs=(input_source_L, input_source_R,), verbose=False)
                    macs, params = clever_format([flops, t_params], '%.3f')
                    print('Model MACs:', macs)
                    print('Model Paras', params)

                print('Cross:(', Cross_Mission + 1, ')', 'Transfer Task-', '(', Task, ')', subject_1, 'to',
                      subject_2)
                ACC = []  # 储存每个Batch-size的准确率
                # net[0].train()
                for epoch in range(args.epochs):
                    progress_bar(epoch, Epoch, es_time, es_time_2)
                    if (epoch+1) % 2 == 0:  # 每5个epoch更新一次时间计次
                        epoch_end = time.time()
                        # 总剩余时间
                        es_time_2 = round((epoch_end - epoch_start)/2 *
                                          (Epoch * (Terminal_Task - Start_Task + 1) *
                                          (Terminal_Cross-Start_Cross+1-Cross_Mission-1) +
                                          Epoch * ((Terminal_Task - Start_Task + 1)-Task) +
                                          Epoch-epoch) / 3600, 2)
                        # 当前轮次剩余时间
                        es_time = round((epoch_end - epoch_start)/2 *
                                        (Epoch * (Terminal_Task-Start_Task+1-Task) +
                                        Epoch-epoch) / 3600, 2)
                        epoch_start = time.time()
                    iter_source_L = iter(source_loader_L)
                    iter_target_L = iter(target_loader_L)

                    iter_source_R = iter(source_loader_R)
                    iter_target_R = iter(target_loader_R)
                    num_iter = len(source_loader_L)  # 数据长度

                    # training
                    net[0] = net[0].cuda()
                    iter_count_source = 0
                    iter_count_target = 0
                    net_count = 0

                    for i in range(0, num_iter):
                        train_source_data_L1, train_source_label_1 = next(iter_source_L)
                        train_target_data_L1, _ = next(iter_target_L)

                        train_source_data_R1, _ = next(iter_source_R)
                        train_target_data_R1, _ = next(iter_target_R)

                        if i % len(target_loader_L) == 0:  # 目标域数据用完，从头开始用
                            iter_target_L = iter(target_loader_L)  # 重新加载目标域数据
                            iter_target_R = iter(target_loader_R)  # 重新加载目标域数据
                            iter_count_target = iter_count_target + 1

                        train_source_data_L1, train_source_label_1 = train_source_data_L1.cuda(), train_source_label_1.cuda()
                        train_target_data_L1 = train_target_data_L1.cuda()
                        train_source_data_L1, train_source_label_1 = Variable(train_source_data_L1), Variable(train_source_label_1)
                        train_target_data_L1 = Variable(train_target_data_L1)

                        train_source_data_R1 = train_source_data_R1.cuda()
                        train_target_data_R1 = train_target_data_R1.cuda()
                        train_source_data_R1 = Variable(train_source_data_R1)
                        train_target_data_R1 = Variable(train_target_data_R1)

                        for key, net in net_dict.items():
                            net_count = net_count + 1
                            output = net[0](train_source_data_L1, train_source_data_R1)
                            clf_loss = net[2](output, train_source_label_1.long())
                            tr_loss = clf_loss
                            net[1].zero_grad()
                            tr_loss.backward()
                            net[1].step()
                            net[3].step()

                    list.append(LOSS, tr_loss.item())
                    list.append(CLF_LOSS, clf_loss.item())

                    cuda = torch.cuda.is_available()
                    va_clf_loss = 0  # 初始化测试损失
                    correct = 0  # 所有分类正确的样本数
                    count = 0  # 统计test中已检测样本个数

                    # net[0].eval()#开启测试模式
                    for validation_data_L, validation_data_R, validation_label in validation_loader:
                        validation_data_L, validation_data_R, validation_label \
                            = validation_data_L.cuda(),  validation_data_R.cuda(), validation_label.cuda()

                        validation_data_L, validation_data_R, validation_label \
                            = Variable(validation_data_L), Variable(validation_data_R), Variable(validation_label)

                        va_pred = net[0](validation_data_L, validation_data_R)
                        va_clf_loss += net[2](va_pred, validation_label.long())
                        va_loss = va_clf_loss
                        pred = va_pred.data.max(1)[1]  # get the index of the max log-probability
                        _correct = pred.eq(validation_label.data.view_as(pred)).cpu().sum()  # 当前epoch中’分类正确样本数‘
                        correct = correct + _correct  # 分类正确样本总数
                        # 每个batch-size中’分类正确样本个数
                        Result_Ans[Cross_Mission-Start_Cross+1, Task - Start_Task, epoch, count] = _correct.item()
                        count = count + 1  # batch-size计数

                    list.append(VA_LOSS, va_loss.item())
                    list.append(VA_CLF_LOSS, va_clf_loss.item())

                    acc = correct / va_num
                    Result_Ans[Cross_Mission-Start_Cross+1, Task - Start_Task, epoch, -2] = correct.item()  # 存入’分类正确样本总数‘
                    Result_Ans[Cross_Mission-Start_Cross+1, Task - Start_Task, epoch, -1] = acc.item()  # 存入’准确率‘
                    list.append(ACC, acc.item())

                # 记录每折中各Task中最优准确率模型的epoch数
                acc_max = 0  # 临时acc值，用来挑选所有epoch中最大的acc
                acc_max_num = 0
                loss_min = max(VA_LOSS)  # total loss中的最大loss值
                loss_min_num = 0

                for i in range(0, epoch_num):  # 计算某个Cross中每个Task的最大准确率
                    acc_current = ACC[i]
                    if acc_current >= acc_max:
                        acc_max = acc_current
                        acc_max_num = i  # 最大acc值对应的epoch序号

                list.append(Max_ACC, acc_max)  # 每个epoch的准确率列表
                Task_ACC[Task - Start_Task, Cross_Mission-Start_Cross+1] = int(acc_max_num)  # 记录每个Cross_Mission中各Task的最优模型编号

                for i in range(0, epoch_num):  # 计算某个Cross中每个Task的最大准确率
                    loss_current = ACC[i]
                    if loss_current <= loss_min:
                        loss_min = loss_current
                        loss_min_num = i
                Task_LOSS[Task - Start_Task, Cross_Mission-Start_Cross] = int(loss_min_num)

                print('  Max ACC = ', acc_max)
                torch.cuda.empty_cache()  # 释放cuda缓存
                Task = Task + 1

            else:  # 跳过重复的用户task
                continue

    AVER_ACC = np.mean(Max_ACC)
    Cross_Aver_ACC.append(AVER_ACC)
    print('Cross Validation(', Cross_Mission+1, ')', '-AVE ACC = ', AVER_ACC, '\n\n')

print('Training Over!!!\n|Average ACC = ', np.mean(Cross_Aver_ACC), '|')
end_time = time.time()
print('|Total Training time:', round((end_time-start)/3600, 3), '/Hour|\n')

print('[Compute the ave acc of Validation]')

row_list = ["Cross-1", "Cross-2", "Cross-3", "Cross-4", "Cross-5", "Cross-6", "Cross-7", "Cross-8", "Cross-9",
            "Cross-10"]
task_mean = np.zeros([Terminal_Task - Start_Task + 1])  # 存放各个task的平均准确率
cross_mean = ['Average accuracy']
for task in range(Start_Task-1, Terminal_Task):
    for cross in range(Start_Cross-1, Terminal_Cross):
        VA_Correct[task-Start_Task+1, cross-Start_Cross+1] = \
        Result_Ans[cross-Start_Cross+1, task-Start_Task+1, int(Task_ACC[task-Start_Task+1, cross-Start_Cross+1]), -2]
    acc = np.sum(VA_Correct[task-Start_Task+1, :])/(va_num*(Terminal_Cross-Start_Cross+1))  # 各个Task的准确率
    task_mean[task-Start_Task+1] = round(acc, 4)
    print('  Task(', str(task+1), ') =', acc)  # 打印每个task的平均准确率
csv_data = np.zeros([Terminal_Task - Start_Task + 1, Terminal_Cross - Start_Cross + 2])  # csv数据矩阵
csv_label = []  # csv标签列表
for cross in range(Start_Cross-1, Terminal_Cross):
    csv_data[:, cross-Start_Cross+1] = VA_Correct[:, cross-Start_Cross+1]/va_num
    list.append(cross_mean, np.mean(VA_Correct[:, cross-Start_Cross+1]/va_num))
    list.append(csv_label, row_list[cross])
list.append(csv_label, "Average accuracy")  # 为表的最后一列添加标题
list.append(cross_mean, np.mean(task_mean))
csv_data[:, -1] = task_mean
data_frame = pd.DataFrame(csv_data, columns=csv_label)
data_frame.to_csv(CSV_name + '.csv')
with open(CSV_name + '.csv', mode='a', newline='', encoding='utf8') as file:
    writer = csv.writer(file)
    writer.writerow(cross_mean)  # 写入csv表格第最后一行
acc = np.sum(VA_Correct)/(va_num * (Terminal_Cross-Start_Cross+1) * (Terminal_Task-Start_Task+1))  # Seed总准确率
print(' Validation average acc = ', acc)
list.append(Seed_ACC, acc)  # 不同种子数下的准确率
