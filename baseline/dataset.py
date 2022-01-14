import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from data.dataset import collate_fn,DataLoaderX
from config.config import SqlConfig


def transDF(mode='train',ds='2021-04-23'):
    """
    dot baseline需要的格式
    :param mode:
    :param ds:
    :return:
    """
    data_info = pd.read_pickle('/data/Win_Prediction/Img/{}/{}.pkl'.format(ds,mode))
    print("数据读取成功")
    # 数据过滤，空值，空列表,比赛结果为0的字段,双方信息有空值
    data_info['trajs_1'] = data_info['trajs_1'].apply(lambda x: None if x == [] else x)
    data_info['trajs_2'] = data_info['trajs_2'].apply(lambda x: None if x == [] else x)
    data_info['image_list'] = data_info['image_list'].apply(lambda x: None if x[0][0][-5] == '0' else x)
    # 两个队伍不能有信息为空
    data_info['static_info'] = data_info['static_info'].apply(
        lambda x: None if x[x['side'] == '1'].empty or x[x['side'] == '2'].empty else x)
    data_info.dropna(axis=0, how='any', inplace=True)

    rows = []
    columns = ["gameplayid", "time_dot", "h_tower_num", "h_player_num", "res", "shiqi", "total_kill", "side_kill",
               "a_tower_num", "a_player_num", "h_skill", "h_grade", "h_totalScore", "h_equipScore", "h_skillScore",
               "h_practiceScore", "h_jingmaiScore", "h_zhuhunScore",
               "a_skill", "a_grade", "a_totalScore", "a_equipScore", "a_skillScore", "a_practiceScore",
               "a_jingmaiScore", "a_zhuhunScore", "local","finish"]
    print("开始转换")
    for row in range(0,data_info.shape[0],7): # 避免造成大量的数据重复，当时是滑窗做的，也可以pickle文件转为CSV进行预测。
        data = data_info.iloc[row]
        gameplayid = data[0]
        h_skill, h_grade, h_totalScore, h_equipScore, h_skillScore, h_practiceScore, h_jingmaiScore, h_zhuhunScore = \
        data['static_info'][data['static_info']['side'] == '1'][
            ['skill', 'grade', 'totalScore', 'equipScore', 'skillScore', 'practiceScore', 'jingmaiScore',
             'zhuhunScore']].astype('float').values.mean(axis=0)
        a_skill, a_grade, a_totalScore, a_equipScore, a_skillScore, a_practiceScore, a_jingmaiScore, a_zhuhunScore = \
        data['static_info'][data['static_info']['side'] == '2'][
            ['skill', 'grade', 'totalScore', 'equipScore', 'skillScore', 'practiceScore', 'jingmaiScore',
             'zhuhunScore']].astype('float').values.mean(axis=0)
        for i in range(7):  # 一行含有7天的数据
            time_dot, h_tower_num, h_player_num, res, shiqi, total_kill, side_kill, local, finish = map(
                lambda x: float(x) if not x.endswith('.jpg') else float(x[:-4]), data['image_list'][0][i].split('_'))
            _, a_tower_num, a_player_num, _, _, _, _, _, _ = map(
                lambda x: float(x) if not x.endswith('.jpg') else float(x[:-4]), data['image_list'][1][i].split('_'))  # 提取了10个特征
            row = {
                "gameplayid": gameplayid,
                "time_dot": time_dot,
                "h_tower_num": h_tower_num,
                "h_player_num": h_player_num, # 比重
                "res": res,
                "shiqi": shiqi,
                "total_kill": total_kill,
                "side_kill": side_kill,
                "a_tower_num": a_tower_num,
                "a_player_num": a_player_num,
                "h_skill": h_skill,
                "h_grade": h_grade,
                # 下面的是一样的每场比赛中
                "h_totalScore": h_totalScore,
                "h_equipScore": h_equipScore,
                "h_skillScore": h_skillScore,
                "h_practiceScore": h_practiceScore,
                "h_jingmaiScore": h_jingmaiScore,
                "h_zhuhunScore": h_zhuhunScore,
                "a_skill": a_skill,
                "a_grade": a_grade,
                "a_totalScore": a_totalScore,
                "a_equipScore": a_equipScore,
                "a_skillScore": a_skillScore,
                "a_practiceScore": a_practiceScore,
                "a_jingmaiScore": a_jingmaiScore,
                "a_zhuhunScore": a_zhuhunScore,
                "local":local,
                "finish": finish
            }
            rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    df.to_pickle("/data/Win_Prediction/Img/{}/{}_bs.pkl".format(ds,mode))
    print("转换成功")


def transStandard(date):
    """
    func: dot baseline standard info
    :param date:
    :return:
    """
    df_train = pd.read_pickle('/data/Win_Prediction/Img/{}/train_bs.pkl'.format(date))
    df_test = pd.read_pickle('/data/Win_Prediction/Img/{}/test_bs.pkl'.format(date))
    # df = pd.concat([df_train,df_test],ignore_index=True)
    # x = df.iloc[:, 0:-3]
    # y = df.iloc[:,-1].values.reshape(-1,1)
    # X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
    X_train = df_train.iloc[:, 12:-2]
    y_train = df_train.iloc[:, -2]+1
    X_test = df_test.iloc[:, 12:-2]
    y_test = df_test.iloc[:, -2]+1 # -2为local
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return X_train,np.asarray(y_train).reshape(-1,),X_test,np.asarray(y_test).reshape(-1,)





class myDataSetbs(Dataset):
    """
    这个dataset是改写用于seq baseline的
    """
    def __init__(self, date, repeat=1, mode="train"):
        '''
        seq dataset
        '''
        self.mode = mode
        self.feature_label = self.readFile(date)
        self.len = self.feature_label.shape[0]
        self.repeat = repeat

    def __getitem__(self, i):
        """
        func：用于返回单条数据
        """
        index = i % self.len  # repeat的情况
        # 注意，这里修正static_info为static
        gameplayid, _, _, _, static, _, _, _, _, seq_static, local_h, local_a, label = self.feature_label.iloc[
            index]  # 解包,对应列顺序
        home_tower_nums,home_player_nums,away_tower_nums,away_player_nums,ress,shiqis,total_kills,side_kills = seq_static
        seq_static = np.asarray(list(zip(home_tower_nums,home_player_nums,away_tower_nums,away_player_nums,ress,shiqis,total_kills,side_kills)))
        label = np.array([label])
        return (gameplayid, seq_static, static, local_h, local_a), label  # 注意这里的返回顺序

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = self.len * self.repeat
        return data_len

    def readFile(self, date):
        """
        func：这里需要将label一并返回
        """
        df = pd.read_pickle('/data/Win_Prediction/Img/{}/{}.pkl'.format(date, self.mode))
        # 进行数据过滤,空列表,空值,任意一方没有数据
        df['trajs_1'] = df['trajs_1'].apply(lambda x: None if x == [] else x)
        df['trajs_2'] = df['trajs_2'].apply(lambda x: None if x == [] else x)
        df['image_list'] = df['image_list'].apply(lambda x: None if x[0][0][-5] == '0' else x)  # 有的比赛以0结尾
        df['static_info'] = df['static_info'].apply(
            lambda x: None if x[x['side'] == '1'].empty or x[x['side'] == '2'].empty else x)
        df.dropna(axis=0, how='any', inplace=True)

        df['home_o'] = df['image_list'].apply(lambda row: row[0])
        df['away_o'] = df['image_list'].apply(lambda row: row[1])
        df['home_d'] = df['image_list'].apply(lambda row: row[2])
        df['away_d'] = df['image_list'].apply(lambda row: row[3])
        df['seq_static'] = df.apply(self.split_feature, axis=1)
        df['static_info'] = df.apply(self.staticProcess, axis=1)  # 优化避免batch处理，在原列上处理

        df['local_h'] = df['image_list'].apply(lambda row: int(row[0][-1][-7]))  # 注意使用最后一个时间段
        df['local_a'] = df['image_list'].apply(lambda row: int(row[1][-1][-7]))  # 这两个是局部的label，不能用作特征，防止信息泄露

        df['label'] = df['image_list'].apply(lambda row: int(row[0][0][-5]))  # 总label
        return df

    def staticProcess(self, row):
        """
        func:处理静态信息
        :param df: static[batch]中的某个df
        :return:
        """
        df = row['static_info']
        h_info = df[df['side'] == '1'][
                     ['skill', 'grade', 'totalScore', 'equipScore', 'skillScore', 'practiceScore', 'jingmaiScore',
                      'zhuhunScore']].astype('float').values.mean(axis=0) / 10 ** 4  # 都是object类型
        a_info = df[df['side'] == '2'][
                     ['skill', 'grade', 'totalScore', 'equipScore', 'skillScore', 'practiceScore', 'jingmaiScore',
                      'zhuhunScore']].astype('float').values.mean(axis=0) / 10 ** 4  # 都是object类型
        h_info[0] = h_info[0] / 1000
        a_info[0] = a_info[0] / 1000
        np.nan_to_num(h_info, copy=False)
        np.nan_to_num(a_info, copy=False)
        static_info = torch.FloatTensor(np.concatenate((h_info, a_info)))
        return static_info

    def split_feature(self, row):
        # 时刻特征
        home_tower_nums = []
        home_player_nums = []
        away_tower_nums = []
        away_player_nums = []
        # 差值
        ress = []
        shiqis = []
        total_kills = []
        side_kills = []
        home_image_list = row['image_list'][0]
        away_image_list = row['image_list'][1]
        for img in home_image_list:
            _, home_tower_num, home_player_num, res, shiqi, total_kill, side_kill, _, _ = img.split('_')
            home_tower_nums.append(int(home_tower_num))
            home_player_nums.append(int(home_player_num))
            ress.append(int(res))
            shiqis.append(int(float(shiqi)))
            total_kills.append(int(float(total_kill)))
            side_kills.append(int(float(side_kill)))
        for img in away_image_list:
            _, away_tower_num, away_player_num, _, _, _, _, _, _ = img.split('_')
            away_tower_nums.append(int(away_tower_num))
            away_player_nums.append(int(away_player_num))
        return home_tower_nums, home_player_nums, away_tower_nums, away_player_nums, ress, shiqis, total_kills, side_kills  # 注意顺序


if __name__ == '__main__':
    # 第一步，数据转换
    transDF(mode='train')
    transDF(mode='test')
    print('数据均转换成功！')
    # 第二步，数据处理
    X_train,y_train,X_test,y_test = transStandard(SqlConfig.DATE)
    print(y_train)
    # 第三步，数据封装为dataset
    # train_data = myDataSetbs(date=SqlConfig.DATE, mode='train')
    # train_loader = DataLoaderX(dataset=train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    # feature, label = next(iter(train_loader))
    # print(label)