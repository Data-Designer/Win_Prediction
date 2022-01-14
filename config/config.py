import warnings


class SqlConfig:
    # 数据拉取时间
    DATE = '2021-04-23' # 可以设置好几个日期，只在训练的时候进行concat
    # 图片
    HOME_PLAYER, AWAY_TOWER, AWAY_PLAYER, HOME_TOWER = 'Blues', 'BuGn', 'BuPu', 'OrRd'
    HOME_D, AWAY_D, HOME_O, AWAY_O = 'home_d', 'away_d', 'home_o', 'away_o'
    PLAYER_SIZE, BUILD_SIZE = 40,60
    # 轨迹
    LEFT = 0
    RIGHT = 55000
    UPPER = 55000
    BOTTOM = 0
    ROW_NUM = 10
    COL_NUM = 10


# 存放必要的训练参数
class TrainConfig:
    # 数据读取的参数
    WINDOWSIZE = 7
    LOGFILE = '/data/Win_Prediction/log2/log2.csv'
    CASEFILE = '/data/Win_Prediction/log2/case.csv'
    # TLOGFILE = '/data/Win_Prediction/log/runsgpu/' 111 这个为了tensor显示
    TLOGFILE = '/data/Win_Prediction/log2/runs01'
    WEIGHTS = '/data/Win_Prediction/log2/checkpoint/' # 只记录最好的checkpoint

    # baseline的一些必要参数，先存在log2,local存在log3
    MLPLOGFILE = '/data/Win_Prediction/log3/baselines/mlplog.csv'
    LSTMLOGFILE = '/data/Win_Prediction/log3/baselines/lstmlog.csv'
    TRANSLOGFILE = '/data/Win_Prediction/log3/baselines/translog.csv'
    TEXTCNNLOGFILE = '/data/Win_Prediction/log3/baselines/textcnnlog.csv'
    TEMPCNNLOGFILE = '/data/Win_Prediction/log3/baselines/tempcnnlog.csv'
    CONLSTMLOGFILE = '/data/Win_Prediction/log3/baselines/conlstmlog.csv'


    # 训练参数
    SEED = 40
    REPEAT = 1 # 设置为NONE表示无限循环
    BATCH_SIZE = 512
    EPOCHS = 150
    SHUFFLE = True
    LR = 1e-4
    LR_DECAY = 1e-4
    WEIGHT_DECAY = 1e-4 # 正则化
    GLOB_WEIGHT = 0.5
    LOCAL_WEIGHT = 0.5
    # 其他参数
    GPU_USED = True # gpu出问题直接调成false看一下
    PARALLEL = False
    LOG_STEP_FREQ = 50

def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数，可以自行设定一些参数
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


TrainConfig.parse = parse # 猴子补丁
opt = TrainConfig() # 包含配置项的实例
