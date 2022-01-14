# -*-coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator # 提速

import pandas as pd
from config.config import SqlConfig,TrainConfig

# 网上的不知道靠不靠谱
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def show_image(title, image):
    '''
    调用matplotlib显示RGB图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()


def cv_show_image(title, image):
    '''
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入RGB图像
    :return:
    '''
    channels = image.shape[-1]
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    cv2.imshow(title, image)
    cv2.waitKey(0)


def read_image(filename, resize_height=None, resize_width=None, normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的RGB图片数据
    '''
    # print("read_image ing open")
    bgr_image = cv2.imread(filename) # 这里好像出问题了!这里会默认多线程读取
    # bgr_image = cv2.imread(filename,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
    if bgr_image is None:
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    rgb_image = resize_image(rgb_image, resize_height, resize_width)
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    #     show_image("src resize image",rgb_image)
    # print("read_image closed")
    return rgb_image


def fast_read_image_roi(filename, orig_rect, ImreadModes=cv2.IMREAD_COLOR, normalization=False):
    '''
    快速读取图片的方法
    :param filename: 图片路径
    :param orig_rect:原始图片的感兴趣区域rect
    :param ImreadModes: IMREAD_UNCHANGED
                        IMREAD_GRAYSCALE
                        IMREAD_COLOR
                        IMREAD_ANYDEPTH
                        IMREAD_ANYCOLOR
                        IMREAD_LOAD_GDAL
                        IMREAD_REDUCED_GRAYSCALE_2
                        IMREAD_REDUCED_COLOR_2
                        IMREAD_REDUCED_GRAYSCALE_4
                        IMREAD_REDUCED_COLOR_4
                        IMREAD_REDUCED_GRAYSCALE_8
                        IMREAD_REDUCED_COLOR_8
                        IMREAD_IGNORE_ORIENTATION
    :param normalization: 是否归一化
    :return: 返回感兴趣区域ROI
    '''
    # 当采用IMREAD_REDUCED模式时，对应rect也需要缩放
    scale = 1
    if ImreadModes == cv2.IMREAD_REDUCED_COLOR_2 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_2:
        scale = 1 / 2
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_4 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_4:
        scale = 1 / 4
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_8 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_8:
        scale = 1 / 8
    rect = np.array(orig_rect) * scale
    rect = rect.astype(int).tolist()
    bgr_image = cv2.imread(filename, flags=ImreadModes)

    if bgr_image is None:
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 3:  #
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    else:
        rgb_image = bgr_image  # 若是灰度图
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    roi_image = get_rect_image(rgb_image, rect)
    # show_image_rect("src resize image",rgb_image,rect)
    # cv_show_image("reROI",roi_image)
    return roi_image


def resize_image(image, resize_height, resize_width):
    '''
    :param image:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image_shape = np.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    if (resize_height is None) and (resize_width is None):  # 错误写法：resize_height and resize_width is None
        return image
    if resize_height is None:
        resize_height = int(height * resize_width / width)
    elif resize_width is None:
        resize_width = int(width * resize_height / height)
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image


def scale_image(image, scale):
    '''
    :param image:
    :param scale: (scale_w,scale_h)
    :return:
    '''
    image = cv2.resize(image, dsize=None, fx=scale[0], fy=scale[1])
    return image


def get_rect_image(image, rect):
    '''
    :param image:
    :param rect: [x,y,w,h]
    :return:
    '''
    x, y, w, h = rect
    cut_img = image[y:(y + h), x:(x + w)]
    return cut_img


def scale_rect(orig_rect, orig_shape, dest_shape):
    '''
    对图像进行缩放时，对应的rectangle也要进行缩放
    :param orig_rect: 原始图像的rect=[x,y,w,h]
    :param orig_shape: 原始图像的维度shape=[h,w]
    :param dest_shape: 缩放后图像的维度shape=[h,w]
    :return: 经过缩放后的rectangle
    '''
    new_x = int(orig_rect[0] * dest_shape[1] / orig_shape[1])
    new_y = int(orig_rect[1] * dest_shape[0] / orig_shape[0])
    new_w = int(orig_rect[2] * dest_shape[1] / orig_shape[1])
    new_h = int(orig_rect[3] * dest_shape[0] / orig_shape[0])
    dest_rect = [new_x, new_y, new_w, new_h]
    return dest_rect


def show_image_rect(win_name, image, rect):
    '''
    :param win_name:
    :param image:
    :param rect:
    :return:
    '''
    x, y, w, h = rect
    point1 = (x, y)
    point2 = (x + w, y + h)
    cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    cv_show_image(win_name, image)


def rgb_to_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def save_image(image_path, rgb_image, toUINT8=True):
    if toUINT8:
        rgb_image = np.asanyarray(rgb_image * 255, dtype=np.uint8)
    if len(rgb_image.shape) == 2:  # 若是灰度图则转为三通道
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_image)


def combime_save_image(orig_image, dest_image, out_dir, name, prefix):
    '''
    命名标准：out_dir/name_prefix.jpg
    :param orig_image:
    :param dest_image:
    :param image_path:
    :param out_dir:
    :param prefix:
    :return:
    '''
    dest_path = os.path.join(out_dir, name + "_" + prefix + ".jpg")
    save_image(dest_path, dest_image)

    dest_image = np.hstack((orig_image, dest_image))
    save_image(os.path.join(out_dir, "{}_src_{}.jpg".format(name, prefix)), dest_image)




# 下面构建Dataset
# -*-coding: utf-8 -*-

def collate_fn(batch):
    """
    func: 重写dataloader中的batch处理函数
    """
    batch = list(zip(*batch))  # 元祖，返回get_item中的对应元祖
    labels = torch.tensor(batch[1], dtype=torch.int32)  # 注意和Tensor的区别
    features = np.array(batch[0],dtype=object) # 方便进行索引
    del batch  # 防止占用内存
    return features, labels


# -*-coding: utf-8 -*-

class myDataSet(Dataset):
    def __init__(self, date, resize_height=32, resize_width=32, repeat=1, mode='train'):
        '''
        :param filename: 数据文件TXT：内容格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''

        self.mode = mode # 注意加载顺序，下面的readfile需要依托于mode
        self.feature_label = self.readFile(date)
        # self.image_dir = './data/Img/{}/'.format(date)  111
        self.image_dir = '/data/Win_Prediction/Img/{}/'.format(date)

        self.len = self.feature_label.shape[0]
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width


        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        # self.toTensor = transforms.Compose([
        #     transforms.ToTensor(),
        # ]
        # )
        self.toTensor = transforms.ToTensor()
        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
        # self.normalize=transforms.Normalize()

    def __getitem__(self, i):
        """
        func：用于返回单条数据
        """
        # cv2.setNumThreads(0)
        # cv2.ocl.setUseOpenCL(False)
        index = i % self.len  # repeat的情况
        # 这里修改static_info为static
        gameplayid, trajs_1, trajs_2, _, static, home_o, away_o, home_d, away_d, seq_static, local_h, local_a, label = self.feature_label.iloc[index]  # 解包,对应列顺序
        home_o = self.prefix(gameplayid, home_o, tag=SqlConfig.HOME_O) # 这个图片不能在本地跑！
        away_o = self.prefix(gameplayid, away_o, tag=SqlConfig.AWAY_O)
        home_d = self.prefix(gameplayid, home_d, tag=SqlConfig.HOME_D)
        away_d = self.prefix(gameplayid, away_d, tag=SqlConfig.AWAY_D)
        label = np.array([label])
        return (gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static, local_h, local_a), label # 注意这里的返回顺序

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

        # df = pd.read_pickle('./data/Img/{}/{}.pkl'.format(date,self.mode)) 111
        df = pd.read_pickle('/data/Win_Prediction/Img/{}/{}.pkl'.format(date, self.mode)) # 这里可以进行多天拼接，在hive中处理过
        # 数据过滤，空值，空列表,比赛结果为0的字段,双方信息有空值
        df['trajs_1'] =df['trajs_1'].apply(lambda x : None if x==[] else x)
        df['trajs_2'] =df['trajs_2'].apply(lambda x : None if x==[] else x)
        df['image_list'] = df['image_list'].apply(lambda x: None if x[0][0][-5] == '0' else x)
        # 两个队伍不能有信息为空
        df['static_info'] = df['static_info'].apply(lambda x : None if x[x['side']=='1'].empty or x[x['side']=='2'].empty else x)

        df.dropna(axis=0, how='any', inplace=True)

        # 分类别提取
        df['home_o'] = df['image_list'].apply(lambda row: row[0]) # 增加列，排在static之后
        df['away_o'] = df['image_list'].apply(lambda row: row[1])
        df['home_d'] = df['image_list'].apply(lambda row: row[2])
        df['away_d'] = df['image_list'].apply(lambda row: row[3])
        df['seq_static'] = df.apply(self.split_feature, axis=1) # home_tower_nums,home_player_nums,away_tower_nums,away_player_nums,ress,shiqis,total_kills,side_kills
        df['static_info'] = df.apply(self.staticProcess, axis=1) # 优化避免batch处理，在原列上处理
        # 提取label数据
        df['local_h'] = df['image_list'].apply(lambda row: int(row[0][-1][-7])) # 注意使用最后一个时间段
        df['local_a'] = df['image_list'].apply(lambda row: int(row[1][-1][-7])) # 这两个是局部的label，不能用作特征，防止信息泄露

        df['label'] = df['image_list'].apply(lambda row: int(row[0][0][-5])) # 保证label为最后一列

        return df

    def load_data(self, path, resize_height, resize_width, normalization):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        image = read_image(path, resize_height, resize_width, normalization)
        return image

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data)
        return data

    def prefix(self, gameplayid, img_lis, tag):
        """
        func: 统一图片变为tensor
        """
        pre_tensor = []

        for img in img_lis:
            # image_path = os.path.join(self.image_dir, gameplayid, tag, img) # 问题出在这个上面
            image_path = self.image_dir+'/'+gameplayid+'/'+tag+'/'+img
            img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=True)
            img = self.data_preproccess(img)
            pre_tensor.append(img.numpy())
        return pre_tensor

    def staticProcess(self,row):
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

    def split_feature(self,row):
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
        home_image_list = row['image_list'][0] # 注意顺序
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
        return home_tower_nums, home_player_nums, away_tower_nums, away_player_nums, ress, shiqis, total_kills, side_kills # 注意顺序


class DataLoaderX(DataLoader): # 感觉用了还不如不用
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':
    train_data = myDataSet(date=SqlConfig.DATE, repeat=TrainConfig.REPEAT,mode='train')
    train_loader = DataLoaderX(dataset=train_data, batch_size=TrainConfig.BATCH_SIZE, shuffle=TrainConfig.SHUFFLE, collate_fn=collate_fn,drop_last=True)
    print("成功")
    # for features, label in train_loader:
    #     print('第一个batch', label)
    feature,label = next(iter(train_loader)) # 第五步，检验dataset和dataloader是否可用
    print(label)


