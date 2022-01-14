# baseline的model主要分为三个部分，timedot部分，seq部分
import torch
import copy
import torch.nn as nn
from torch.nn.utils import weight_norm

import numpy as np
import torch.nn.functional as F
from catboost import CatBoostClassifier, Pool
from config.config import opt

#---------------------------------dot--------------------------#
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(14,16), # 这里要注意输入维度
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.Dropout(),
        )
        self.classifier_g = nn.Linear(8,3)
        # self.classifier_l = nn.Linear(8,3) # 数据不支持

    def forward(self,x):
        out = self.mlp(x)
        glob = self.classifier_g(out)
        return glob


clf = CatBoostClassifier(iterations=2, learning_rate=1, depth=2)



#---------------------------------seq--------------------------#


class Positional_Encoding(nn.Module):
    '''
    params: embed-->word embedding dim      pad_size-->max_sequence_lenght
    Input: x
    Output: x + position_encoder
    '''

    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # 奇数cos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 单词embedding与位置编码相加，这两个张量的shape一致
        out = x + nn.Parameter(self.pe, requires_grad=False).cuda()
        out = self.dropout(out)
        return out


class Multi_Head_Attention(nn.Module):
    '''
    params: dim_model-->hidden dim      num_head
    '''

    def __init__(self, dim_model, num_head, dropout=0.5):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0  # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head  # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q，通过Linear实现张量之间的乘法，等同手动定义参数W与之相乘
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)  # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1,
                   self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # 无需mask
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale)  # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head)  # reshape 回原来的形状
        out = self.fc(context)  # 全连接
        out = self.dropout(out)
        out = out + x  # 残差连接,ADD
        out = self.layer_norm(out)  # 对应Norm
        return out


class Multi_Head_Attention(nn.Module):
    '''
    params: dim_model-->hidden dim *num_head
    '''

    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0  # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head  # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q，通过Linear实现张量之间的乘法，等同手动定义参数W与之相乘
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)  # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1,
                   self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale)  # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head)  # reshape 回原来的形状
        out = self.fc(context)  # 全连接
        out = self.dropout(out)
        out = out + x  # 残差连接,ADD
        out = self.layer_norm(out)  # 对应Norm
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)  # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])  # 多次Encoder

        self.glob = nn.Linear(config.pad_size * config.dim_model+8, config.num_classes_g)
        self.local =nn.Linear(config.pad_size * config.dim_model+8, config.num_classes_l)
        self.static_fc = nn.Linear(16,8)
    def forward(self, x,static):
        out = self.postion_embedding(x)  # batch,seqlen
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)  # 将三维张量reshape成二维，然后直接通过全连接层将高维数据映射为classes
        # out = torch.mean(out, 1)    # 也可用池化来做，但是效果并不是很好
        static = self.stati_fc(static)
        out = torch.cat((static,out),dim=1) # 维度比原来增加8
        glob = self.glob(out)
        local = self.local(out)
        return glob#,local


class ConfigTrans(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes_g = 3  # 类别数
        self.num_classes_l = 3  # 类别数
        self.num_epochs = opt.EPOCHS  # epoch数
        self.batch_size = opt.BATCH_SIZE  # mini-batch大小
        self.pad_size = opt.WINDOWSIZE  # 每句话处理成的长度(短填长切)，这个根据自己的数据集而定
        self.learning_rate = opt.LR  # 学习率
        self.embed = 64  # 字向量维度
        self.dim_model = 64  # 需要与embed一样
        self.hidden = 64
        self.last_hidden = 64
        self.num_head = 8  # 多头注意力，需要能够被embed整除！，不然会出现问题！
        self.num_encoder = 1  # 使用两个Encoder，尝试6个encoder发现存在过拟合，毕竟数据集量比较少（10000左右），可能性能还是比不过LSTM

config = ConfigTrans()



class LSTMTag(nn.Module):
    def __init__(self):
        super(LSTMTag,self).__init__() # 自动识别长度
        self.lstm = nn.LSTM(8, # embedding dim
                            32, # hidden_dim
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True)
        self.static_fc = nn.Linear(16,8)
        self.classifier_g = nn.Linear(32+8,3)
        self.classifier_l = nn.Linear(32+8,3)
    def forward(self,x,static):
        out, _ = self.lstm(x)  # batch,seq_len,hidden
        out = out[:, -1, :]  # batch,hidden 最后一层
        static = self.static_fc(static)
        out = torch.cat((out,static),dim=1)
        glob = self.classifier_g(out)
        local = self.classifier_l(out)
        return glob#,local



class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
         # return shape: (batch_size, channel, 1)
        return F.max_pool1d(x, kernel_size=x.shape[2])



class CNNTag(nn.Module):
    """
    TextcNN
    """
    def __init__(self, embed_size=8, kernel_sizes=[3,4,5], num_channels=[32,32,32]):
        super(CNNTag, self).__init__()
        # 不参与训练的嵌入层
        self.dropout = nn.Dropout(0.5)
        self.classifier_g = nn.Linear(sum(num_channels) + 8, 3) # 加上static
        self.classifier_l = nn.Linear(sum(num_channels) + 8, 3)
        self.classifiers = nn.Linear(5, 5)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.static_fc = nn.Linear(16,8)
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = embed_size,
                                        out_channels = c,
                                        kernel_size = k)) # 卷积就是不断聚合的过程


    def forward(self, inputs,static):
        static= self.static_fc(static)
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = inputs # (batch, seq_len, embed_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        encoding = torch.cat((encoding,static),dim=1)
        # 应用丢弃法后使用全连接层得到输出
        glob = self.classifier_g(self.dropout(encoding))
        local = self.classifier_l(self.dropout(encoding))
        return glob#,local


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs=8, num_channels=[16, 16], kernel_size=5, dropout=0.5):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.static_fc = nn.Linear(16,8)
        self.classifier_g = nn.Linear(120,3) # 这个地方是2个
        self.classifier_l = nn.Linear(120,3) # 这个地方是3个


    def forward(self, x,static):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        x = x.permute(0, 2, 1)
        out = self.network(x).view(opt.BATCH_SIZE, -1)
        static = self.static_fc(static)
        out = torch.cat((static, out), dim=1)  # 维度比原来增加8
        glob = self.classifier_g(out)
        local = self.classifier_l(out)
        return glob#,local


#---------------------------------CONVLSTM--------------------------#

class ConvLSTMCell(nn.Module):
    """
    func: 实现每个时刻的Conv提取层
    return : 下一个时刻的h和c
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        # input_dim是每个num_layer的第一个时刻的的输入dim，即channel
        # hidden_dim是每一个num_layer的隐藏层单元，如第一层是64，第二层是128，第三层是128
        # kernel_size是卷积核
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        #padding的目的是保持卷积之后大小不变
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,#卷积输入的尺寸
                              out_channels=4 * self.hidden_dim,#因为lstmcell有四个门，隐藏层单元是rnn的四倍
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        #input_tensor的尺寸为（batch_size，channel，weight，width），没有time_step
        #cur_state的尺寸是（batch_size,（hidden_dim）*channel，weight，width），是调用函数init_hidden返回的细胞状态
        # 感觉这里是给channel变成ID

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        #conv层的卷积不需要和linear一样，可以是多维的，只要channel数目相同即可

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        #使用split函数把输出4*hidden_dim分割成四个门
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g   #下一个细胞状态
        h_next = o * torch.tanh(c_next)  #下一个hc

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        #核对尺寸，用的函数是静态方法

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        #kernel_size==hidden_dim=num_layer的维度，因为要遍历num_layer次
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        #如果return_all_layers==true，则返回所有得到h，如果为false，则返回最后一层的最后一个h

        cell_list = []
        for i in range(0, self.num_layers):
            #判断input_dim是否是第一层的第一个输入，如果是的话则使用input_dim，否则取第i层的最后一个hidden_dim的channel数作为输入
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        #以num_layer为三层为例，则cell_list列表里的内容为[convlstmcell0（），convlstmcell1（），convlstmcell2（）]
        #Module_list把nn.module的方法作为列表存放进去，在forward的时候可以调用Module_list的东西，cell_list【0】，cell_list【1】，
        #一直到cell_list【num_layer】，
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        #第一次传入hidden_state为none
        #input_tensor的size为（batch_size,time_step,channel,height,width）
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        #在forward里开始构建模型，首先把input_tensor的维度调整，然后初始化隐藏状态
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            #调用convlstm的init_hidden方法不是lstmcell的方法
            #返回的hidden_state有num_layer个hc，cc
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)#取time_step
        cur_layer_input = input_tensor

        #初始化h之后开始前向传播
        for layer_idx in range(self.num_layers):
            #在已经初始化好了的hidden_state中取出第num_layer个状态给num_layer的h0，c0，其作为第一个输入
            h, c = hidden_state[layer_idx]
            output_inner = []
            #开始每一层的时间步传播
            for t in range(seq_len):
                #用cell_list[i]表示第i层的convlstmcell，计算每个time_step的h和c
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                #将每一次的h存放在output_inner里
                output_inner.append(h)
            #layer_output是五维向量，在dim=1的维度堆栈，和input_tensor的维度保持一致
            layer_output = torch.stack(output_inner, dim=1)
            #吧每一层输出肚饿五维向量作为下一层的输入，因为五维向量的输入没有num_layer，所以每一层的输入都要喂入五维向量
            cur_layer_input = layer_output
            #layer_output_list存放的是第一层，第二层，第三层的每一层的五维向量，这些五维向量作为input_tensor的输入
            layer_output_list.append(layer_output)
            #last_state_list里面存放的是第一层，第二层，第三次最后time_step的h和c
            last_state_list.append([h, c])

        if not self.return_all_layers:
            #如果return_all_layers==false的话，则返回每一层最后的状态，返回最后一层的五维向量，返回最后一层的h和c
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]


        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            #cell_list[i]是celllstm的单元，以调用里面的方法
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
            #返回的init_states为num_layer个hc=（batch_size,channel(hidden_dim),height,width），cc=（batch_size,channel(hidden_dim),height,width）
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLstmClassifier(nn.Module):
    def __init__(self):
        super(ConvLstmClassifier, self).__init__()
        self.offensive = ConvLSTM(3, 1, (5, 5), 1, True, True,False)
        self.defensive = ConvLSTM(3, 1, (5, 5), 1, True, True,False)
        self.classifier_g = nn.Sequential(
                                nn.Linear(2048,32),
                                nn.Linear(32,3)
        )
        self.classifier_l = nn.Sequential(
            nn.Linear(2048, 32),
            nn.Linear(32, 3)
        )

    def forward(self,home_o,away_o,home_d,away_d):
        home_off = self.offensive(home_o).view(opt.BATCH_SIZE,-1)
        home_def = self.defensive(home_d).view(opt.BATCH_SIZE,-1) # batch 1024
        away_off = self.offensive(away_o).view(opt.BATCH_SIZE,-1)
        away_def = self.defensive(away_d).view(opt.BATCH_SIZE,-1)
        overall = torch.cat((home_off * away_def, home_def * away_off), axis=-1)
        glob = self.classifier_g(overall)
        local = self.classifier_l(overall)
        return glob#,local



if __name__ == '__main__':
    # mlp
    model = MLP()
    test = torch.randn((32, 32))
    print(model(test).shape)
    # transformer
    model = Transformer()
    model.cuda()  # 这里注意位置参数必须单独进行cuda
    test = torch.randn(32, 7, 64).cuda()
    print(model(test).shape)
    # lstm
    model = LSTMTag()
    test = torch.randn((32, 7, 16))
    print(model(test).shape)
    # convlstm
    x = torch.rand((32, 7, 3, 32, 32))  # Batch, Time, Channel, Height, Width
    convlstm = ConvLSTM(3, 3, (5, 5), 1, True, True,
                        False)  # input_channel, hidden_channel, kernel_size, num_layers,batch_first=False, bias=True, return_all_layers=False
    _, last_states = convlstm(x)
    h = last_states[0][0]  # 0 for layer index, 0 for h index,1 for c index
    print(h.shape)
