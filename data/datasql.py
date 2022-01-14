# 数据查询
import os
import pandas as pd
from rslib.utils.datadownload import datadownload
from data.sql import Sql
from config.config import SqlConfig


def sqlTest(sql):
    '''
    测试sql语句
    :param sql:
    :return:
    '''
    df = datadownload(sql)
    print(df.shape)


def sqlGameLog(sql,ds):
    """
    拉取固定某天的战场信息
    :param sql:
    :param ds:
    :return:
    """
    if not os.path.exists('./down/{}.pkl'.format(ds)):
        print("该日期文件不存在，正在下载数据")
        df = datadownload(sql.format(ds,ds,ds))
        df = df.sort_values(by=['gameplayid','battlefield_duration']) # 排好序等待处理
        print("Download All Data Done!")
        df.to_pickle('./down/{}.pkl'.format(ds))
        print("All convert to ./down/{}.pkl".format(ds))
    else:
        print('该日期文件已下载，正在读取文件')
        df = pd.read_pickle('./down/{}.pkl'.format(ds))
        print('读取文件完毕！')
    return df


if __name__ == '__main__':
    sql = Sql()
    sql_config = SqlConfig()
    # sqlTest(sql_config.SQL_TEST)
    game_log_data = sqlGameLog(sql.SQL_GAME_LOG,SqlConfig.DATE)
    print("总数据量",len(game_log_data))

