# 该文件用于进行数据预处理，dic返回
import json
import matplotlib
import os
import matplotlib.pyplot as plt
import pandas as pd
from config.config import SqlConfig
from data.sql import Sql
# from data.datasql import sqlGameLog # 使用凯哥的环境，使用一次后注释
from multiprocessing import *
import multiprocessing.sharedctypes as sharedctypes
import ctypes


matplotlib.use('Agg')


def playerOn(data,time_dot):
    """
    func: 返回某一个时间点的所有玩家ID，list
    """
    data_player = list(json.loads(data['players'].iloc[time_dot]).keys())
    return data_player


def playerTeamOn(data,time_dot):
    """
    func: 返回双方队伍的某一时刻的player_list，信息
    """
    player_ids = playerOn(data,time_dot)
    team_1 = {}
    team_2 = {}
    for player_id in player_ids:
        player_info = playerInfo(data,player_id,time_dot)
        if player_info['side'] == '1':
            team_1[player_id] = player_info
        else:
            team_2[player_id] = player_info
    return team_1,team_2


def playerInfo(data, player_id, time_dot):
    """
    func: 返回某一个时间点的玩家ID的全部信息

    """
    player_dic = {}
    data_player = json.loads(data['players'].iloc[time_dot])[player_id]
    player_dic['coordinator'] = data_player[:3]  # [x,y,z]
    player_dic['side'] = data_player[3]
    player_dic['hp'] = data_player[4]
    player_dic['hurted'] = data_player[5]
    player_dic['cured'] = data_player[6]
    player_dic['damagePlayer'] = data_player[7]
    player_dic['damageBuilding'] = data_player[8]
    player_dic['cure'] = data_player[9]
    player_dic['reborn'] = data_player[10]
    player_dic['useSkill'] = data_player[11]
    player_dic['kill'] = data_player[12]
    player_dic['assist'] = data_player[13]
    player_dic['buff'] = data_player[14]
    # 下面的信息只在玩家首次被记录的时候才有
    player_dic['static_info'] = {}
    player_dic['static_info']['skill'] = data_player[15]
    player_dic['static_info']['class'] = data_player[16]
    player_dic['static_info']['grade'] = data_player[17]
    player_dic['static_info']['totalScore'] = data_player[18]
    player_dic['static_info']['equipScore'] = data_player[19]
    player_dic['static_info']['skillScore'] = data_player[20]
    player_dic['static_info']['practiceScore'] = data_player[21]
    player_dic['static_info']['jingmaiScore'] = data_player[22]
    player_dic['static_info']['zhuhunScore'] = data_player[23]
    #     player_dic['static_info']['role_name']= json.loads(data['players'][time_dot])[player_id][24] 有的数据没有这一条
    return player_dic


def team_Info(data, time_dot):
    """
    func:返回某一个时间点某局战场局势信息
    """
    team_dic = {}
    # 这个是整体局势的信息
    team_dic['finish'] = data['finish'].iloc[time_dot]
    team_dic['battlefield_duration'] = data["battlefield_duration"].iloc[time_dot]
    team_dic['total_kill'] = data["total_kill"].iloc[time_dot]
    # 更改存储方式
    team_dic['team_1'] = {}
    team_dic['team_2'] = {}
    # 下面是队伍双方的信息
    team_dic['team_1']['side_kill'] = data['side1_kill'].iloc[time_dot]
    team_dic['team_2']['side_kill'] = data['side2_kill'].iloc[time_dot]
    team_dic['team_1']['shiqi'] = {}
    team_dic['team_2']['shiqi'] = {}
    team_dic['team_1']['shiqi'] = data['shiqi1'].iloc[time_dot]
    team_dic['team_2']['shiqi'] = data['shiqi2'].iloc[time_dot]
    team_dic['team_1']['res'] = {}
    team_dic['team_2']['res'] = {}
    team_dic['team_1']['res'] = data['res1'].iloc[time_dot]  # 资源
    team_dic['team_2']['res'] = data['res2'].iloc[time_dot]
    team_dic['team_1']['item'] = {}
    team_dic['team_2']['item'] = {}
    team_dic['team_1']['item'] = data['item1'].iloc[time_dot].split(",")  # 道具
    team_dic['team_2']['item'] = data['item2'].iloc[time_dot].split(",")
    # 下面的部分有可能会消失
    team_dic['team_1']['jianta'] = {}
    team_dic['team_2']['jianta'] = {}
    if data['jianta1'].iloc[time_dot]:  # 防止为空
        jianta_1 = data['jianta1'].iloc[time_dot].split(",")
        jianta_1_size = len(jianta_1) // 5
        team_dic['team_1']['jianta'] = [jianta_1[i * 5:(i + 1) * 5] for i in range(jianta_1_size)]
    if data['jianta2'].iloc[time_dot]:
        jianta_2 = data['jianta2'].iloc[time_dot].split(",")
        jianta_2_size = len(jianta_2) // 5
        team_dic['team_2']['jianta'] = [jianta_2[i * 5:(i + 1) * 5] for i in range(jianta_2_size)]

    team_dic['team_1']['jidi'] = {}
    team_dic['team_2']['jidi'] = {}
    if data['jidi1'].iloc[time_dot]:
        jidi_1 = data['jidi1'].iloc[time_dot].split(",")
        team_dic['team_1']['jidi']['max_hp'] = {}
        team_dic['team_1']['jidi']['max_hp'] = jidi_1[1]
        team_dic['team_1']['jidi']['cur_hp'] = {}
        team_dic['team_1']['jidi']['cur_hp'] = jidi_1[0]
        team_dic['team_1']['jidi']['coordinator'] = {}
        team_dic['team_1']['jidi']['coordinator'] = jidi_1[2:]
    if data['jidi2'].iloc[time_dot]:
        jidi_2 = data['jidi2'].iloc[time_dot].split(",")
        team_dic['team_2']['jidi']['max_hp'] = {}
        team_dic['team_2']['jidi']['max_hp'] = jidi_2[1]
        team_dic['team_2']['jidi']['cur_hp'] = {}
        team_dic['team_2']['jidi']['cur_hp'] = jidi_2[0]
        team_dic['team_2']['jidi']['coordinator'] = {}
        team_dic['team_2']['jidi']['coordinator'] = jidi_2[2:]

    team_dic['team_1']['liangcang'] = {}
    team_dic['team_2']['liangcang'] = {}
    if data['liangcang1'].iloc[time_dot]:
        liangcang_1 = data['liangcang1'].iloc[time_dot].split(",")
        team_dic['team_1']['liangcang']['max_hp'] = {}
        team_dic['team_1']['liangcang']['max_hp'] = liangcang_1[1]
        team_dic['team_1']['liangcang']['cur_hp'] = {}
        team_dic['team_1']['liangcang']['cur_hp'] = liangcang_1[0]
        team_dic['team_1']['liangcang']['coordinator'] = {}
        team_dic['team_1']['liangcang']['coordinator'] = liangcang_1[2:]
    if data['liangcang2'].iloc[time_dot]:
        liangcang_2 = data['liangcang2'].iloc[time_dot].split(",")
        team_dic['team_2']['liangcang']['max_hp'] = {}
        team_dic['team_2']['liangcang']['max_hp'] = liangcang_2[1]
        team_dic['team_2']['liangcang']['cur_hp'] = {}
        team_dic['team_2']['liangcang']['cur_hp'] = liangcang_2[0]
        team_dic['team_2']['liangcang']['coordinator'] = {}
        team_dic['team_2']['liangcang']['coordinator'] = liangcang_2[2:]
    return team_dic


def teamInfoOn(data,time_dot):
    """
    func:返回双方建筑信息（这里其实顺带可以返回label信息）
    """
    team_dic = team_Info(data,time_dot)
    team_other = [team_dic['total_kill'],team_dic['battlefield_duration'],
                  team_dic['team_1']['res']-team_dic['team_2']['res'],
                  team_dic['team_1']['shiqi']-team_dic['team_2']['shiqi'],
                  team_dic['team_1']['side_kill']-team_dic['team_2']['side_kill'],
                  team_dic['finish']] # 注意如果加入别的信息可以在这里加入
    return team_dic['team_1'],team_dic['team_2'],team_other


def timeDot(data):
    """
    func：获取某场比赛的time_dot列表-》time_line,20s间隔
    """
    size = len(data)
    return [0] + list(range(10,size))


def timeWindows(time_line, window_size):
    """
    func: 返回windows滑窗数组
    """
    if len(time_line) < window_size:
        raise Exception("Invalid Log! Too short windows!")
    time_arrays = []
    length = len(time_line)
    for i in range(0, length - window_size + 1):
        subarray = time_line[i:i + window_size]
        time_arrays.append(subarray)
    return time_arrays


def imageInfoAll(data,time_dot):
    """
    func: 返回设计的图片的组合，防守防守，进攻进攻所有信息
    """
    team_players_1, team_players_2 = playerTeamOn(data, time_dot)
    team_builds_1, team_builds_2, team_other = teamInfoOn(data, time_dot)
    return (team_players_1, team_builds_1), (team_players_2, team_builds_2), (team_players_1, team_builds_2), (
    team_players_2, team_builds_1), team_other


def imageInfo(info_pair):
    """
    info_pair: e.g.home_defensive
    func: 接收所有信息，返回绘图所需必要信息(x_lis,y_lis,hp_lis)
    """
    x_lis = []
    y_lis = []
    hp_lis = []
    xb_lis = []
    yb_lis = []
    bhp_lis = []
    team_players, team_builds = info_pair
    for team_player in team_players.items():
        x_lis.append(team_player[1]['coordinator'][0])
        y_lis.append(team_player[1]['coordinator'][1])
        hp_lis.append(team_player[1]['hp'][0])

    # 下面的建筑很有可能为空
    if team_builds['jidi']:
        xb_lis.append(team_builds['jidi']['coordinator'][0])
        yb_lis.append(team_builds['jidi']['coordinator'][1])
        bhp_lis.append(float(team_builds['jidi']['cur_hp'])/float(team_builds['jidi']['max_hp']))# 归一化，这个因为道具问题
    if team_builds['liangcang']:
        xb_lis.append(team_builds['liangcang']['coordinator'][0])
        yb_lis.append(team_builds['liangcang']['coordinator'][1])
        bhp_lis.append(float(team_builds['liangcang']['cur_hp'])/float(team_builds['liangcang']['max_hp']))
    if team_builds['jianta']:
        for jianta in team_builds['jianta']:
            xb_lis.append(jianta[-3])
            yb_lis.append(jianta[-2])
            bhp_lis.append(float(jianta[-5])/float(jianta[-4]))
    return list(map(int, x_lis)), list(map(int, y_lis)), list(map(float, hp_lis)), list(map(int, xb_lis)), list(map(int, yb_lis)), list(map(float, bhp_lis))



def imageProduct(data,time_dot):
    """
    func: 返回绘图所需的所有信息
    """
    home_defensive,away_defensive,home_offensive,away_offensive,team_other = imageInfoAll(data,time_dot)
    home_d = imageInfo(home_defensive)
    away_d = imageInfo(away_defensive)
    home_o = imageInfo(home_offensive)
    away_o = imageInfo(away_offensive)
    return home_d,away_d,home_o,away_o,team_other


def draw(x_lis,y_lis,hp_lis,xb_lis,yb_lis,bhp_lis,colors,date,game_id,tag,time_dot,tower_num,player_num,res,shiqi,total_kill,side_kill,towernum_xo,finish):
    """
    tag标志位：存入哪个文件夹，image文件名用于解析label
    func:绘图
    """
    plt.figure(figsize=(5,5),dpi=100)
    cm1 = plt.cm.get_cmap(colors[0])
    cm2 = plt.cm.get_cmap(colors[1])
    plt.axis([0,55000,0,55000]) # 这里防止出现图片定位不一致
    plt.scatter(x_lis, y_lis,c=hp_lis,s= SqlConfig.PLAYER_SIZE,cmap=cm1,vmin=0,vmax=1) # min_hp->max_hp
    plt.scatter(xb_lis, yb_lis,c=bhp_lis,s= SqlConfig.BUILD_SIZE,cmap=cm2,vmin=0,vmax=1) # min_hp->max_hp
    plt.axis("off") # 后面是全局信息
    # plt.savefig("./Img/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.jpg".format(date,game_id,tag,time_dot,tower_num,player_num,res,shiqi,total_kill,side_kill,towernum_xo,finish),bbox_inches = 'tight',pad_inches = 0) 111
    plt.savefig(
        "/data/Win_Prediction/Img/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.jpg".format(date, game_id, tag, time_dot, tower_num, player_num, res,
                                                               shiqi, total_kill, side_kill, towernum_xo, finish),
        bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()


def imageDraw(data,time_dot,date,game_id):
    """
    func: 绘制图片并保存,这里保存全局信息到image上
    """
    home_d,away_d,home_o,away_o,team_other  = imageProduct(data,time_dot)
    # 这里可以通过time_dot+1算出局部label
    x_lis1,y_lis1,hp_lis1,xb_lis1,yb_lis1,bhp_lis1 = home_d
    x_lis2,y_lis2,hp_lis2,xb_lis2,yb_lis2,bhp_lis2 = away_d
    x_lis3,y_lis3,hp_lis3,xb_lis3,yb_lis3,bhp_lis3 = home_o
    x_lis4,y_lis4,hp_lis4,xb_lis4,yb_lis4,bhp_lis4 = away_o

    # 局部label,下一个时刻（因为下一个时间段140s有塔被攻破太正常了）
    if time_dot < len(data)-1:
        home_d_l,away_d_l,_,_,_ = imageProduct(data,time_dot+1)
        _,_,_,xb_lis1_l,_,_ = home_d_l # 下一个时刻的塔数
        _,_,_,xb_lis2_l,_,_ = away_d_l
        h_towernum_xo = 1 if (len(xb_lis2_l)-len(xb_lis2)) < 0 else 0  # 是否攻破
        a_towernum_xo = 1 if (len(xb_lis1_l)-len(xb_lis1)) < 0 else 0
    else:
        h_towernum_xo = 2 # 最后需要有个标志位【滑窗最后一位滑不到】
        a_towernum_xo = 2

    draw(x_lis1,y_lis1,hp_lis1,xb_lis1,yb_lis1,bhp_lis1,[SqlConfig.HOME_PLAYER,SqlConfig.AWAY_TOWER],date,game_id,SqlConfig.HOME_D,time_dot,len(xb_lis1),len(x_lis1),team_other[-6],team_other[-4],team_other[-3],team_other[-2],h_towernum_xo,data.iloc[-1]['finish']) # 这里用finish来表示最终的战争结果
    draw(x_lis2,y_lis2,hp_lis2,xb_lis2,yb_lis2,bhp_lis2,[SqlConfig.HOME_TOWER,SqlConfig.AWAY_PLAYER],date,game_id,SqlConfig.AWAY_D,time_dot,len(xb_lis2),len(x_lis2),team_other[-6],team_other[-4],team_other[-3],team_other[-2],a_towernum_xo,data.iloc[-1]['finish'])
    draw(x_lis3,y_lis3,hp_lis3,xb_lis3,yb_lis3,bhp_lis3,[SqlConfig.HOME_PLAYER,SqlConfig.HOME_TOWER],date,game_id,SqlConfig.HOME_O,time_dot,len(xb_lis3),len(x_lis3),team_other[-6],team_other[-4],team_other[-3],team_other[-2],h_towernum_xo,data.iloc[-1]['finish'])
    draw(x_lis4,y_lis4,hp_lis4,xb_lis4,yb_lis4,bhp_lis4,[SqlConfig.AWAY_PLAYER,SqlConfig.AWAY_TOWER],date,game_id,SqlConfig.AWAY_O,time_dot,len(xb_lis4),len(x_lis4),team_other[-6],team_other[-4],team_other[-3],team_other[-2],a_towernum_xo,data.iloc[-1]['finish'])


# 划分网格区域，生成对应区域的网格ID
# lon：经度，lat：纬度，column_num：列数，row_num：行数
def generalID(x,y,row_num,column_num):
    # 若在范围外的点，返回-1
    if x <= SqlConfig.LEFT or x >= SqlConfig.RIGHT or y <= SqlConfig.BOTTOM or y >= SqlConfig.UPPER:
        raise Exception("You need to change BOUNDS！")
    # 把经度范围根据列数等分切割
    column = (SqlConfig.RIGHT - SqlConfig.LEFT)/column_num
    # 把纬度范围根据行数数等分切割
    row = (SqlConfig.UPPER - SqlConfig.BOTTOM)/row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((x-SqlConfig.LEFT)/column)+ 1 + int((y-SqlConfig.BOTTOM)/row) * column_num



def trajectoryData(data):
    """
    func：生成轨迹数据，注意这里data是data['players‘]
    :param data:
    :return:
    """
    time_dots = len(data)
    columns = ['time_dot','playerid','x','y','side','hp']
    rows = []
    for time_dot in [0]+list(range(10,time_dots)):
        players_time_dot = json.loads(data.iloc[time_dot]) # 某一个时刻的所有玩家的信息
        for player in players_time_dot.items():
            dic = {
                'time_dot':time_dot,
                'playerid': player[0],
                'x': int(player[1][0]),
                'y': int(player[1][1]),
                'side': player[1][3], # 注意是string
                'hp': player[1][4]
            }
            rows.append(dic)
    df = pd.DataFrame(rows,columns = columns)
    df['area'] = df.apply(lambda row: generalID(row['x'], row['y'],SqlConfig.ROW_NUM,SqlConfig.COL_NUM), axis = 1)
    return df


def staticData(data,gameplayid):
    """
    func:生成静态player信息,这里接收的data['players‘]
    """
    players_static = json.loads(data.iloc[0]) # 生成静态信息数据
    columns = ['gameplayid','playerid','side','skill','class','grade','totalScore','equipScore','skillScore','practiceScore','jingmaiScore', 'zhuhunScore']
    rows = []
    for player in players_static.items():
            dic = {
                'gameplayid':gameplayid,
                'playerid': player[0],
                'side': player[1][3], # 注意是string
                'skill':player[1][15],
                'class':player[1][16],
                'grade':player[1][17],
                'totalScore':player[1][18],
                'equipScore':player[1][19],
                'skillScore':player[1][20],
                'practiceScore':player[1][21],
                'jingmaiScore':player[1][22],
                'zhuhunScore':player[1][23],
            }
            rows.append(dic)
    df = pd.DataFrame(rows,columns = columns)
    return df



# def save(gameplayid,data_all,date=SqlConfig.DATE):
#     """
#     func:用于处理多进程
#     :param gameplayid:
#     :param date:
#     :return:
#     """
#     # 创建一些东西
#     data = data_all[data_all['gameplayid'] == gameplayid]
#     if len(json.loads(data['players'].iloc[0])) <= 20:  # data['res1'].iloc[0]==data['res2'].iloc[0]
#         print("机器人对局，跳过该id")  # 人数过少进行判定
#         return
#     else:
#         if not os.path.exists('/data/Win_Prediction/Img/{}/{}'.format(date, gameplayid)):
#             # os.makedirs('./Img/{}/{}/home_d'.format(date, gameplayid)) 111仔细筛查
#             os.makedirs('/data/Win_Prediction/Img/{}/{}/home_d'.format(date, gameplayid))
#             os.makedirs('/data/Win_Prediction/Img/{}/{}/home_o'.format(date, gameplayid))
#             os.makedirs('/data/Win_Prediction/Img/{}/{}/away_d'.format(date, gameplayid))
#             os.makedirs('/data/Win_Prediction/Img/{}/{}/away_o'.format(date, gameplayid))
#         traj = trajectoryData(data['players'])  # 存储轨迹数据
#         traj.to_pickle('/data/Win_Prediction/Img/{}/{}/traj.pkl'.format(date, gameplayid))
#         print('{}-{}轨迹数据已存入'.format(date, gameplayid))
#         print('{}-{}正在生成图片数据'.format(date, gameplayid))
#         time_dot_size = timeDot(data)
#         for time_dot in time_dot_size:
#             if time_dot == 0:
#                 static_info = staticData(data['players'], gameplayid)  # 存储静态信息
#                 static_info.to_pickle('/data/Win_Prediction/Img/{}/{}/static.pkl'.format(date, gameplayid))
#                 print('{}-{}静态数据已存入'.format(date, gameplayid))
#             imageDraw(data, time_dot=time_dot, date=date, game_id=gameplayid)  # 存储图片信息
#         print('{}-{}图片数据已存入\n'.format(date, gameplayid))
#         print('{}全部数据处理完成\n'.format(gameplayid))
#
#
#
# def batchInfoD(data_all,date):
#     """
#     func: 批量生产轨迹，图片，静态信息
#     data_all: 这里传入的是所有data信息（某天）,这里改成多线程
#     """
#
#     mgr = Manager()
#     ns = mgr.Namespace()
#     ns.df = data_all # 进程之间共享大文件
#     pool = Pool(2)
#     if not os.path.exists('/data/Win_Prediction/Img/{}'.format(date)):
#         os.makedirs('/data/Win_Prediction/Img/{}'.format(date))
#         # os.makedirs('./Img/{}'.format(date)) 111
#     gameplayeid_lis = data_all['gameplayid'].unique()
#     print(len(gameplayeid_lis))
#     for gameplayid in gameplayeid_lis:
#         pool.apply_async(func=save, args=(gameplayid,ns,SqlConfig.DATE))
#         # save(gameplayid=gameplayid, data_all=ns, date=SqlConfig.DATE)
#     pool.close()
#     pool.join()
#     print("ALL DONE !")



def batchInfoD(data_all,date):
    """
    func: 批量生产轨迹，图片，静态信息
    data_all: 这里传入的是所有data信息（某天）
    """
    if not os.path.exists('/data/Win_Prediction/Img/{}'.format(date)):
        os.makedirs('/data/Win_Prediction/Img/{}'.format(date))
    gameplayeid_lis = data_all['gameplayid'].unique()
    for gameplayid in gameplayeid_lis: # 这里测试跑通代码
        # 创建一些东西
        data = data_all[data_all['gameplayid']==gameplayid]
        if len(json.loads(data['players'].iloc[0]))<=100: # data['res1'].iloc[0]==data['res2'].iloc[0]
            print("机器人对局或者比赛人数过少，跳过该id") # 人数过少进行判定
            continue # 避免机器人对局
        else:
            if not os.path.exists('/data/Win_Prediction/Img/{}/{}'.format(date,gameplayid)):
                os.makedirs('/data/Win_Prediction/Img/{}/{}/home_d'.format(date,gameplayid))
                os.makedirs('/data/Win_Prediction/Img/{}/{}/home_o'.format(date,gameplayid))
                os.makedirs('/data/Win_Prediction/Img/{}/{}/away_d'.format(date,gameplayid))
                os.makedirs('/data/Win_Prediction/Img/{}/{}/away_o'.format(date,gameplayid))
            traj = trajectoryData(data['players']) # 存储轨迹数据
            traj.to_pickle('/data/Win_Prediction/Img/{}/{}/traj.pkl'.format(date,gameplayid))
            print('{}-{}轨迹数据已存入'.format(date,gameplayid))
            print('{}-{}正在生成图片数据'.format(date,gameplayid))
            time_dot_size = timeDot(data)
            for time_dot in time_dot_size:
                if time_dot == 0:
                    static_info = staticData(data['players'],gameplayid) # 存储静态信息
                    static_info.to_pickle('/data/Win_Prediction/Img/{}/{}/static.pkl'.format(date,gameplayid))
                    print('{}-{}静态数据已存入'.format(date, gameplayid))
                imageDraw(data,time_dot = time_dot,date=date,game_id =gameplayid) # 存储图片信息
            print('{}-{}图片数据已存入\n'.format(date,gameplayid))
    print("ALL DONE !")



if __name__ == '__main__':
    sql = Sql()
    sql_config = SqlConfig()
    # data_info = sqlGameLog(sql.SQL_GAME_LOG,sql_config.DATE) # 第0步，使用凯哥的环境搞一次就行,记得注释下面语句
    print("数据库pkl文件已存储，直接读取！")
    data_info = pd.read_pickle('./down/{}.pkl'.format(sql_config.DATE))
    print("读取数据完毕")
    batchInfoD(data_info,sql_config.DATE) # 第一步，生成所有的文件,包括上面的SQL,这里可以传入2021-04-23的日期，用于合并别的日期的数据