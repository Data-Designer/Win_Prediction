import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import TrainConfig,SqlConfig,opt


class ImgEmbMeta(nn.Module):
    """
    func：图片元信息提取,backbone=LeNet
    in [batch,channel,width,height]
    return [batch,32]
    """

    def __init__(self):
        super(ImgEmbMeta, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 这里比较好调参
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.bc1 = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(64,32) # 获取长度为32的embedding
        self.bc2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):  # x [batch,channel,h,w]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.bc1(self.fc1(x)))
        x = F.relu(self.bc2(self.fc2(x)))
        x = self.fc3(x)
        return x  # [batch,32]

#
# class OverallEmb(nn.Module):
#     """
#     func:单个时间点整体信息提取器
#     in 4*[batch channel width height]
#     return [batch 64]
#     """
#
#     def __init__(self):
#         super(OverallEmb, self).__init__()
#         self.offensive = ImgEmbMeta()
#         self.defensive = ImgEmbMeta()
#
#     def forward(self, home_o, home_d, away_o, away_d):
#         home_off = self.offensive(home_o)
#         away_off = self.offensive(away_o)
#         home_def = self.defensive(home_d)
#         away_def = self.defensive(away_d)
#         overall = torch.cat((home_off * away_def, home_def * away_off), axis=-1)  # [batch,64]
#         return overall


class Static(nn.Module):
    """
    func: 静态特征提取,赛前胜率预测【变换input维度】
    in [batch 16]
    return [batch 16]
    """

    def __init__(self, input_dim=16, output_dim=16):
        super(Static, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.fc(x)  # batch,output
        return out



# class SeqOverall(nn.Module):
#     """
#     func:总体整体信息提取
#     in [[batch channel width height] * 4] *7
#     """
#     def __init__(self,hidden_dim=32,n_layers=2,dropout=0.5):
#         super(SeqOverall,self).__init__()
#         self.overall_emb = OverallEmb() # 这是一个通用的特征提取器
#         self.lstm = nn.LSTM(64,
#                             hidden_dim,
#                             num_layers=n_layers,
#                             dropout=dropout,
#                             batch_first=True)
#     def forward(self,x):
#         overalls = torch.zeros((7,TrainConfig.BATCH_SIZE,64)) # window,batch,overall_hidden 循环拼接的标准做法
#         for index, imgs in enumerate(x):
#             home_o,home_d,away_o,away_d = imgs # 一次获取4个tensor
#             overall = self.overall_emb(home_o,home_d,away_o,away_d) # batch，hidden
#             overalls[index] = overall
#         overalls = overalls.permute(1,0,2) # batch.window,overall_hidden
#         out,_ = self.lstm(overalls)
#         return out[:,-1,:] # batch,hidden_dim

class OverallEmbSeq(nn.Module):
    """
    batch_size*7需要在外面加好，返回整体信息提取
    in [torch.rand((3,32,32))]*7*32,..., 4份]
    return [batch hidden_dim]
    """

    def __init__(self, hidden_dim=32, n_layers=2, dropout=0.5):
        super(OverallEmbSeq, self).__init__()
        self.offensive = ImgEmbMeta()
        self.defensive = ImgEmbMeta()
        self.lstm = nn.LSTM(64,
                            hidden_dim,
                            num_layers=n_layers,
                            dropout=dropout,
                            batch_first=True)
        # lstm参数初始化
        for name, param in self.lstm.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, home_o, home_d, away_o, away_d):
        # print(home_o)
        # home_o = torch.stack(home_o, dim=0)  # batch*7,3,32,32
        # home_d = torch.stack(home_d, dim=0)
        # away_o = torch.stack(away_o, dim=0)
        # away_d = torch.stack(away_d, dim=0)
        home_off = self.offensive(home_o)
        away_off = self.offensive(away_o)
        home_def = self.defensive(home_d)
        away_def = self.defensive(away_d)
        overall = torch.cat((home_off * away_def, home_def * away_off), axis=-1)  # [batch*7,64]
        seq_overall = torch.split(overall, [opt.WINDOWSIZE] * opt.BATCH_SIZE, dim=0)
        # seq_overall = torch.split(overall, [TrainConfig.WINDOWSIZE] * TrainConfig.BATCH_SIZE, dim=0) 111

        seq_lstm_overall = torch.stack(seq_overall, dim=0)  # 专门转为LSTM的格式 [batch,7,embed]
        out, _ = self.lstm(seq_lstm_overall)
        return out[:, -1, :]  # batch,hidden_dim




class TrajEmbMeta(nn.Module):
    """
    玩家层级轨迹信息提取器
    """

    def __init__(self, location_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 dropout):  # 注意location_size好像需要留一位给中途离开的玩家前面注意修改为0
        super(TrajEmbMeta, self).__init__()
        self.embedding = nn.Embedding(location_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # lstm参数初始化
        for name,param in self.lstm.named_parameters():
            nn.init.uniform_(param,-0.1,0.1)


    def forward(self, traj):
        '''
        traj_len用于padding，不过这里用不上traj都是7为单位进行滑窗
        '''
        embedded = self.embedding(traj)  # batch,seq_len,embed_dim
        out, _ = self.lstm(embedded)  # batch,seq_len,hidden
        out = out[:, -1, :]  # batch,hidden 最后一层
        out = self.fc(out)
        return out  # batch,output_dim


class Attention(nn.Module):
    """
    层级Attention，汇集某方玩家的轨迹为一个向量
    """
    def __init__(self):
        super(Attention,self).__init__()
        self.w_omega = nn.Parameter(torch.Tensor(32,32)) # hidden_layer
        self.u_omega = nn.Parameter(torch.Tensor(32,1))
        nn.init.uniform_(self.w_omega,-0.1,0.1)
        nn.init.uniform_(self.u_omega,-0.1,0.1)
        self.attention = nn.Softmax(dim=0) # 注意我们没有batch维度
    def forward(self,x):
        u = torch.tanh(torch.matmul(x,self.w_omega)) # k*hidden
        att = torch.matmul(u,self.u_omega) # k*1，内积
        att_score = self.attention(att) # k*1
        score_x = x * att_score # k*hidden,broadcast
        out = torch.sum(score_x,dim=0) # hidden
        return out


class TrajEmbedTeam(nn.Module):
    """
    个体信息提取，batch中的每个都是变长的，所以需要特殊处理。
    in Batch[K1+K2+..Kn]* embed_dim
    return [batch output_dim]
    """

    def __init__(self, location_size=SqlConfig.ROW_NUM*SqlConfig.COL_NUM+1, embedding_dim=8, hidden_dim=32, output_dim=32, n_layers=2, dropout=0.5):
        # 切记这里的local_size=10*10+1,需要增加中途离开embed
        super(TrajEmbedTeam, self).__init__()
        self.output_dim = output_dim
        self.traj_embed = TrajEmbMeta(location_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
        self.attention = Attention()

    def forward(self, x, count_num):  # (batch-K的和) * 7,这个k不固定,这里的count_num在static中已经提取过，直接做切片[batch*k在外面用加法合并],不过一旦传递标量就gg,不能parallel
        all_trajs = self.traj_embed(x)
        # print(all_trajs.shape)
        # print(count_num)
        all_trajs = torch.split(all_trajs, count_num, dim=0)
        batch_trajs = torch.zeros((opt.BATCH_SIZE, self.output_dim))  # bacth,output_dim
        # batch_trajs = torch.zeros((TrainConfig.BATCH_SIZE, self.output_dim))  # bacth,output_dim 111

        for index, trajs in enumerate(all_trajs):
            batch_trajs[index] = trajs.float().mean(dim=0)  # K * hiddendim,k不固定
        return batch_trajs.cuda()  # 这里存到batch里了，所以要加cuda


class LastEmb(nn.Module):
    """
    func：所有信息合并，注意局部local
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m,nn.BatchNorm2d):
            nn.init.constant_(m.weight,0)
            nn.init.constant_(m.bias,0)

    def __init__(self):
        super(LastEmb, self).__init__()
        self.h_seqtrajs = TrajEmbedTeam()
        self.a_seqtrajs = TrajEmbedTeam()
        self.overall = OverallEmbSeq()
        self.static = Static()  # 这里需要指明大小
        self.share = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 16, 32),
            nn.BatchNorm1d(32), # 注意顺序
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.local = nn.Sequential(
            nn.Linear(16, 3),
            # nn.Softmax(dim=1)  # 因为最后一个滑窗没有local指标，所以需要三分类,但是这里注意loss func！
        )
        self.glob = nn.Sequential(
            nn.Linear(16, 2),
            # nn.Softmax(dim=1) # 只有胜负两种情况
        )
        self.apply(self.weight_init)

    def forward(self, h_trajs, h_count_num, a_trajs, a_count_num, home_o,home_d,away_o,away_d, static):
        """
        参数格式示例：Batch为2
        h_trajs: torch.LongTensor([[1,2,4,5,2,1,2],[4,3,2,9,1,2,3],[2,2,3,4,5,6,1]])
        h_count_num: [1,2]
        a_trajs: torch.LongTensor([[1,2,4,5,2,1,2],[4,3,2,9,1,2,3],[2,2,3,4,5,6,1]])
        home_o:[torch.rand((3,32,32))]*BATCH_SIZE*WINDOW_SIZE
        static:torch.randn(2,16)
        """
        h_trajs_embed = self.h_seqtrajs(h_trajs, h_count_num)
        a_trajs_embed = self.a_seqtrajs(a_trajs, a_count_num)  # batch,hidden_dim
        overall_embed = self.overall(home_o,home_d,away_o,away_d)
        static_embed = self.static(static)
        # print(h_trajs_embed.shape, a_trajs_embed.shape, overall_embed.shape, static_embed.shape)
        all_embed = torch.cat((h_trajs_embed, a_trajs_embed, overall_embed, static_embed), dim=1)
        # print(all_embed.shape)
        all_embed = self.share(all_embed)
        local = self.local(all_embed)  # batch,3
        glob = self.glob(all_embed)  # batch 2
        return local, glob


if __name__ == '__main__':
    # 注意下面调试的时候Batch_Size设置为2
    model = LastEmb()
    h_trajs = torch.LongTensor([[1, 2, 4, 5, 2, 1, 2], [4, 3, 2, 9, 1, 2, 3], [2, 2, 3, 4, 5, 6, 1]])
    h_count_num = [1, 2]
    a_trajs = torch.LongTensor([[1, 2, 4, 5, 2, 1, 2], [4, 3, 2, 9, 1, 2, 3], [2, 2, 3, 4, 5, 6, 1]])
    a_count_num = [1, 2]
    home_o, home_d, away_o, away_d = torch.randn((14,3,32,32)),torch.randn((14,3,32,32)),torch.randn((14,3,32,32)),torch.randn((14,3,32,32))
    static = torch.randn(2, 16)
    local,glob = model(h_trajs, h_count_num, a_trajs, a_count_num, home_o, home_d, away_o, away_d, static)
    print(local)
