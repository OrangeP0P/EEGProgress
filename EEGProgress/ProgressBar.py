import sys
import time

def progress_bar(finish_epoch_number,epoch_number,es_time,es_time_2):
    """
    进度条

    :param finish_tasks_number: int, 已完成的任务数
    :param tasks_number: int, 总的任务数
    :return:
    """

    percentage = round(finish_epoch_number / epoch_number * 100)
    print("\r 【本轮:{}/小时 预期{}/小时】 进度:{}%:".format(es_time,es_time_2,percentage),  end="")
    sys.stdout.flush()

# def progress_bar(finish_epoch_number,epoch_number,es_time,es_time_2):
#     """
#     进度条
#
#     :param finish_tasks_number: int, 已完成的任务数
#     :param tasks_number: int, 总的任务数
#     :return:
#     """
#
#     percentage = round(finish_epoch_number / epoch_number * 100)
#     print("\r 【本轮:{}/小时 预期{}/小时】 进度:{}%:".format(es_time,es_time_2,percentage), "▓" * (percentage // 10), end="")
#     sys.stdout.flush()


# def progress_bar(finish_epoch_number,epoch_number,Task,Cross_Mission,start,end):
#     """
#     进度条
#
#     :param finish_tasks_number: int, 已完成的任务数
#     :param tasks_number: int, 总的任务数
#     :return:
#     """
#
#     percentage = round(finish_epoch_number / epoch_number * 100)
#     es_time = round( ((end-start)/50 * epoch_number*(10-Cross_Mission-1)*20+\
#                       (end-start)/50 * epoch_number*(20-Task)+\
#                       (end-start)/50 * (epoch_number-finish_epoch_number))\
#                      /3600,3)
#     if finish_epoch_number == 0:
#         print("\r 【预期:{}/小时】 进度:{}%:".format('NAN',percentage), "▓" * (percentage // 2), end="")
#     else:
#         print("\r 【预期:{}/小时】 进度:{}%:".format(es_time,percentage), "▓" * (percentage // 2), end="")
#     sys.stdout.flush()
#     return es_time

def progress_bar_1(finish_epoch_number,epoch_number,es_time):
    """
    进度条

    :param finish_tasks_number: int, 已完成的任务数
    :param tasks_number: int, 总的任务数
    :return:
    """

    percentage = round(finish_epoch_number / epoch_number * 100)
    if finish_epoch_number == 0 or es_time < 0:
        print("\r 【预期:{}/小时】 进度:{}%:".format('NAN',percentage), "▓" * (percentage // 2), end="")
    else:
        print("\r 【预期:{}/小时】 进度:{}%:".format(es_time,percentage), "▓" * (percentage // 2), end="")
    sys.stdout.flush()
