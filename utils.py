# 这里存储一些常用的函数
# time1:5 ,time2:8 time3:10 time4: 9
import os
import pandas as pd
import torch.nn as nn
import torch
import random
import numpy as np
import csv
from config.config import TrainConfig,SqlConfig,opt
from sklearn.model_selection import train_test_split,GridSearchCV

def timeWindows(max_time, window_size):
    """
    max_time为最大时间窗口
    func: 返回windows滑窗数组
    """
    time_line = [0] + list(range(10,max_time+1))
    time_arrays = []
    length = len(time_line)
    for i in range(0, length - window_size + 1):
        subarray = time_line[i:i + window_size]
        time_arrays.append(subarray)
    return time_arrays


def trajsGet(data,time_array,playerid):
    """
    func: 返回某玩家在一个time_array的trajs
    """
    traj = []
    for time in time_array:
#         print(time) # 定位是哪个时间段
        # 如果玩家离开游戏就标识为0，方便model处理（原来处理为-1）
        area = data[(data['time_dot']==time) & (data['playerid']==playerid)]['area']
        traj.append(int(area)) if not area.empty else traj.append(int(0))# 注意，当这里出现不存在的id的时候不能转为int，会出现series报错
    return traj


def allTrajsGet(data,time_array,side):
    """
    func: 返回某方所有玩家在一个time_array的trajs，注意这是一条数据。
    """
    trajs = []
    playerids = data[(data['time_dot']==time_array[0]) &(data['side']==side) ]['playerid'].tolist() # 取每个开头时间的playerids
    for playerid in playerids:
#         print(playerid) 定位是哪个玩家
        trajs.append(trajsGet(data,time_array,playerid))
    return trajs



def imageGet(date,gameplayid,time_array):
    """
    func: 返回gameplayid的某时段图片地址
    """
    # home_o = os.listdir('./data/Img/{}/{}/home_o'.format(date,gameplayid)) 111
    home_o = os.listdir('/data/Win_Prediction/Img/{}/{}/home_o'.format(date, gameplayid))
    home_o.sort(key=lambda x: int(x.split('_')[0]))
    away_o = os.listdir('/data/Win_Prediction/Img/{}/{}/away_o'.format(date,gameplayid))
    away_o.sort(key=lambda x: int(x.split('_')[0]))
    home_d = os.listdir('/data/Win_Prediction/Img/{}/{}/home_d'.format(date,gameplayid))
    home_d.sort(key=lambda x: int(x.split('_')[0]))
    away_d = os.listdir('/data/Win_Prediction/Img/{}/{}/away_d'.format(date,gameplayid))
    away_d.sort(key=lambda x: int(x.split('_')[0]))
    home_off = []
    away_off = []
    home_def = []
    away_def = []
    for time_dot in time_array:
        if time_dot == 0:
            home_off.append(home_o[0])
            away_off.append(away_o[0])
            home_def.append(home_d[0])
            away_def.append(away_d[0])
        else:
            home_off.append(home_o[time_dot-10+1])
            away_off.append(away_o[time_dot-10+1])
            home_def.append(home_d[time_dot-10+1])
            away_def.append(away_d[time_dot-10+1])
    return home_off,away_off,home_def,away_def # 注意这个顺序



def saveIndex(gameplayid,date=SqlConfig.DATE,window_size=opt.WINDOWSIZE): # 关于数据生成都得在代码里先生成，再提交
    """
    func: 接收game_id，存储所有Data index
    """
    # 这里是超餐调整
    if os.path.exists('/data/Win_Prediction/Img/{}/{}/time4.pkl'.format(date,gameplayid)):
        print('{}已生成time4.pkl，跳过! \n'.format(gameplayid))
        return

    # data = pd.read_pickle('./data/Img/{}/{}/traj.pkl'.format(date,gameplayid)) 111
    data = pd.read_pickle('/data/Win_Prediction/Img/{}/{}/traj.pkl'.format(date, gameplayid))
    time_arrays = timeWindows(max(data['time_dot']),window_size)
    columns = ['gameplayid','trajs_1','trajs_2','image_list','static_info']
    rows = []
    for time_array in time_arrays:
        row = {
            'gameplayid': gameplayid, # 这个是id
            'trajs_1':allTrajsGet(data,time_array,'1'), # 存储动态数据
            'trajs_2':allTrajsGet(data,time_array,'2'),
            'image_list':imageGet(date,gameplayid,time_array),
            'static_info': pd.read_pickle('/data/Win_Prediction/Img/{}/{}/static.pkl'.format(date, gameplayid))
        } # 注意必加static.pkl文件
        rows.append(row)
    df = pd.DataFrame(rows,columns= columns)
    # 这里做超参数
    df.to_pickle('/data/Win_Prediction/Img/{}/{}/time4.pkl'.format(date,gameplayid)) # 复杂的序列数据格式另存
    print("{}序列存储成功\n".format(gameplayid))



def batchInfoG(date):
    '''
    func: 批量生成读取文件time.pkl
    '''
    from multiprocessing import Pool
    pool = Pool(6)

    # gameplay_ids = os.listdir('./data/Img/{}/'.format(date)) 111
    gameplay_ids = os.listdir('/data/Win_Prediction/Img/{}/'.format(date))
    for gameid in gameplay_ids:
        print("{}正在处理为{}窗口的时序数据".format(gameid,opt.WINDOWSIZE))
        pool.apply_async(func=saveIndex, args=(gameid,))
        # saveIndex(gameplayid=gameid,date=date,window_size=TrainConfig.WINDOWSIZE)
        # print("{}序列处理成功\n".format(gameid))
    pool.close()
    pool.join()
    print('ALL DONE!')


def mergeInfoG(date):
    """
    func: 合并每一个目录下面的相同time.pkl的文件，最终合并为想要的格式
    """
    list_ = []
    gameplay_ids = os.listdir('/data/Win_Prediction/Img/{}/'.format(date))
    gameplay_ids.remove('last.pkl') # 这里列出目录需要去除一部分
    gameplay_ids.remove('test.pkl')
    gameplay_ids.remove('test_bs.pkl')
    gameplay_ids.remove('train.pkl')
    gameplay_ids.remove('train_bs.pkl')
    gameplay_ids.remove('last2.pkl')
    gameplay_ids.remove('test2.pkl')
    gameplay_ids.remove('train2.pkl')
    gameplay_ids.remove('last1.pkl')
    gameplay_ids.remove('test1.pkl')
    gameplay_ids.remove('train1.pkl')
    gameplay_ids.remove('last3.pkl')
    gameplay_ids.remove('test3.pkl')
    gameplay_ids.remove('train3.pkl')
    for gameid in gameplay_ids:
        # 超参数
        file = pd.read_pickle('/data/Win_Prediction/Img/{}/{}/time4.pkl'.format(date,gameid))
        list_.append(file)
    df = pd.concat(list_,axis=0) # 合并到一个大文件中
    df = df.reset_index(drop=True) # 重置索引
    print("全部合并完成")
    # 超参数
    df.to_pickle('/data/Win_Prediction/Img/{}/last4.pkl'.format(date))
    print("已存储为/data/Win_Prediction/Img/{}/last4.pkl".format(date)) # 111


def trainTestSplit(date):
    """
    返回对应的gameids，用于合并，划分训练和测试集
    :param date:
    :return:
    """
    # 超参数
    df = pd.read_pickle('/data/Win_Prediction/Img/{}/last4.pkl'.format(date)) #111
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_pickle('/data/Win_Prediction/Img/{}/train4.pkl'.format(date))
    test_df.to_pickle('/data/Win_Prediction/Img/{}/test4.pkl'.format(date))
    print("训练测试已切分完毕！")


def initial_weight(model):
    """
    参数初始化
    :param model:
    :return:
    """
    net = model
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


def setup_seed(seed):
    '''
    func:设置随机种子
    :param seed:
    :return:
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def modelProcess(trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static,static):
    """
    func：元batch数据处理为->model接受的格式
    """
    trajs_1 = torch.LongTensor(np.concatenate(trajs_1))
    trajs_2 = torch.LongTensor(np.concatenate(trajs_2))
    h_count_num = [i[1][0] for i in seq_static]
    a_count_num = [i[3][0] for i in seq_static]
    home_o = torch.FloatTensor(np.concatenate(home_o))
    home_d = torch.FloatTensor(np.concatenate(home_d))
    away_o = torch.FloatTensor(np.concatenate(away_o))
    away_d = torch.FloatTensor(np.concatenate(away_d))
    static = torch.FloatTensor(np.stack(static))
    return trajs_1,h_count_num,trajs_2,a_count_num,home_o,away_o,home_d,away_d,static


def write_csv(results, file_name):
    # 写入test数据方便后期debug
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['gameplayid', 'h_local_prob','h_pred_glob','a_local_prob','a_pred_glob',"h_batch_attn","a_batch_attn"])
        writer.writerows(results)


def gridsearch(train_x,train_y,model,param_grid,n_jobs=-1,verbose=1,scoring = 'f1_micro'):
    '''
    网格搜索调参
    '''
    model= model
    param_grid = param_grid
    scoring = scoring
    grid_search = GridSearchCV(model,param_grid,n_jobs=n_jobs,verbose = verbose,scoring = scoring) #默认5折交叉
    grid_search.fit(train_x,train_y)
    best_params = grid_search.best_estimator_.get_params()
    print("Best params: ")
    for param,val in best_params.items():
        print(param,val)
    model = grid_search.best_estimator_
    model.fit(train_x,train_y)
    return model



if __name__ == '__main__':
    batchInfoG(SqlConfig.DATE) # 第二步，生成各场gameid的time.pkl
    mergeInfoG(SqlConfig.DATE) # 第三步，生成总的训练文件
    trainTestSplit(SqlConfig.DATE) # 第五步，生成训练测试数据