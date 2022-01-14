# 主训练文件
import torch
import pandas as pd
import datetime
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from utils import modelProcess
from config.config import TrainConfig,SqlConfig,opt
from model.model import LastEmb
from data.dataset import myDataSet,DataLoaderX,collate_fn
from pytorch_lightning import seed_everything


# 设置随机种子
seed_everything(TrainConfig.SEED)
writer = SummaryWriter(TrainConfig.TLOGFILE) # 定义Tensorboard句柄




def train_step(model,features,local_labels,glob_labels):
    """
    这里注意是个多任务
    :param model:
    :param features:
    :param local_labels:
    :param labels:
    :return:
    """
    model.train()
    # 梯度清零
    model.optimizer.zero_grad()
    # 处理为model可以接受的格式
    gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static = features
    h_trajs,h_count_num,a_trajs,a_count_num,home_o,away_o,home_d,away_d,static = modelProcess(trajs_1, trajs_2,
                                                                                              home_o, away_o,
                                                                                              home_d, away_d,
                                                                                              seq_static, static)
    if TrainConfig.GPU_USED:
        h_trajs, h_count_num, a_trajs, a_count_num, home_o, away_o, home_d, away_d, static = h_trajs.cuda(),h_count_num,\
                                                                                             a_trajs.cuda(),a_count_num,home_o.cuda(),\
                                                                                             away_o.cuda(),home_d.cuda(),away_d.cuda(),static.cuda()
    # 正向计算
    pred_local,pred_glob = model(h_trajs, h_count_num, a_trajs, a_count_num, home_o, home_d, away_o, away_d, static)
    # 计算loss
    loss_local = model.criterion(pred_local,local_labels)
    loss_glob = model.criterion(pred_glob,glob_labels) # 这里应该为0和1才对！原来的label标签是1和2
    loss_total = loss_local*TrainConfig.LOCAL_WEIGHT + loss_glob*TrainConfig.GLOB_WEIGHT
    metric_local = model.metric_func(pred_local.softmax(dim=-1), local_labels)
    metric_glob = model.metric_func(pred_glob.softmax(dim=-1),glob_labels) # 这个指标需要soft
    # 反向传播
    loss_total.backward()
    model.optimizer.step()
    return loss_total.item(),loss_local.item(),loss_glob.item(),metric_local.item(),metric_glob.item()



def valid_step(model,features,local_labels,glob_labels):
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        # 处理为model可以接受的格式
        gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static = features
        h_trajs, h_count_num, a_trajs, a_count_num, home_o, away_o, home_d, away_d, static = modelProcess(trajs_1,trajs_2,
                                                                                                          home_o, away_o,
                                                                                                          home_d,away_d,
                                                                                                          seq_static,static)
        if TrainConfig.GPU_USED:
            h_trajs, h_count_num, a_trajs, a_count_num, home_o, away_o, home_d, away_d, static = h_trajs.cuda(), h_count_num, \
                                                                                                 a_trajs.cuda(), a_count_num, home_o.cuda(), \
                                                                                                 away_o.cuda(), home_d.cuda(), away_d.cuda(), static.cuda()
        # 正向计算
        pred_local,pred_glob = model(h_trajs, h_count_num, a_trajs, a_count_num, home_o, home_d, away_o, away_d, static)
        # 计算loss
        loss_local = model.criterion(pred_local, local_labels)
        loss_glob = model.criterion(pred_glob, glob_labels)
        loss_total = loss_local + loss_glob
        metric_local = model.metric_func(pred_local.softmax(dim=-1), local_labels)
        metric_glob = model.metric_func(pred_glob.softmax(dim=-1), glob_labels)
    return loss_total.item(),loss_local.item(),loss_glob.item(),metric_local.item(),metric_glob.item()




def train(model,epochs,train_loader,val_loader):
    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns=['epoch','loss_total','loss_local','loss_glob','local_'+metric_name,'glob_'+metric_name,
                                      'val_loss_total','val_loss_local','val_loss_glob','val_local_'+metric_name,'val_glob_'+metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)
    for epoch in range(1,TrainConfig.EPOCHS+1):
        # 训练循环
        loss_total_sum = 0.0
        loss_local_sum = 0.0
        loss_glob_sum = 0.0
        metric_local_sum = 0.0
        metric_glob_sum = 0.0
        # 其实这里的glob指标才是重要的
        best_glob_acc = 0.0

        step = 1 # 每个epoch从头开始
        for step,(features,labels) in enumerate(train_loader,1):
            gameplayid,trajs_1,trajs_2,home_o,away_o,home_d,away_d,seq_static,static,local_h,local_a = [features[:,i] for i in range(11)] # 解析feature
            re_features = (gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static)
            local_labels = torch.LongTensor(local_h.astype(int)).cuda() # 这里是预测home侧，预测away侧有些指标要取反！
            glob_labels = (labels.reshape(-1,).long() - 1).cuda() # 需要longtensor
            loss_total, loss_local, loss_glob, metric_local, metric_glob = train_step(model,re_features,local_labels,glob_labels)
            loss_total_sum += loss_total
            loss_local_sum += loss_local
            loss_glob_sum += loss_glob
            metric_local_sum += metric_local
            metric_glob_sum += metric_glob
            # 使用tensorboard记录数据
            writer.add_scalar('Train Total Loss', loss_total_sum / step, (epoch - 1) * len(train_loader) + step)
            writer.add_scalar('Train Local Loss', loss_local_sum / step, (epoch - 1) * len(train_loader) + step)
            writer.add_scalar('Train Local '+ metric_name, metric_local_sum / step, (epoch - 1) * len(train_loader) + step)
            writer.add_scalar('Train Global Loss', loss_glob_sum / step, (epoch - 1) * len(train_loader) + step)
            writer.add_scalar('Train Global '+ metric_name, metric_glob_sum / step, (epoch - 1) * len(train_loader) + step)
            # 打印数据
            if step % TrainConfig.LOG_STEP_FREQ == 0:
                print(("[step = %d] loss_total: %.3f, " + "local_"+metric_name + ": %.3f, "
                       + "glob_"+metric_name + ": %.3f") %
                      (step, loss_total_sum / step, metric_local_sum / step, metric_glob_sum / step))


        # 验证循环
        val_loss_total_sum = 0.0
        val_loss_local_sum = 0.0
        val_loss_glob_sum = 0.0
        val_metric_local_sum = 0.0
        val_metric_glob_sum = 0.0
        val_step = 1
        for val_step,(features,labels) in enumerate(val_loader,1):
            gameplayid,trajs_1,trajs_2,home_o,away_o,home_d,away_d,seq_static,static,local_h,local_a = [features[:,i] for i in range(11)]
            re_features = (gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static)
            local_labels = torch.LongTensor(local_h.astype(int)).cuda() # 这里是预测home侧，预测away侧有些指标要取反！
            glob_labels = (labels.reshape(-1,).long() - 1).cuda() # 分类会自动调整为one_hot格式,1,2->0,1
            loss_total, loss_local, loss_glob, metric_local, metric_glob = valid_step(model, re_features, local_labels,
                                                                                      glob_labels)
            val_loss_total_sum += loss_total
            val_loss_local_sum += loss_local
            val_loss_glob_sum += loss_glob
            val_metric_local_sum += metric_local
            val_metric_glob_sum += metric_glob
            # tensorboard记录数据
            writer.add_scalar('Val Total Loss', val_loss_total_sum / val_step, (epoch - 1) * len(val_loader) + val_step)
            writer.add_scalar('Val Local Loss', val_loss_local_sum / val_step, (epoch - 1) * len(val_loader) + val_step)
            writer.add_scalar('Val Local '+ metric_name, val_metric_local_sum / val_step, (epoch - 1) * len(val_loader) + val_step)
            writer.add_scalar('Val Global Loss', val_loss_glob_sum / val_step, (epoch - 1) * len(val_loader) + val_step)
            writer.add_scalar('Val Global '+ metric_name, val_metric_glob_sum / val_step, (epoch - 1) * len(val_loader) + val_step)

        # 记录Epoch日志
        info = (epoch,loss_total_sum/step,loss_local_sum/step,loss_glob_sum/step,metric_local_sum/step,metric_glob_sum/step,
                val_loss_total_sum/val_step,val_loss_local_sum/val_step,val_loss_glob_sum/val_step,val_metric_local_sum/val_step,val_metric_glob_sum/val_step)
        dfhistory.loc[epoch-1] = info
        # 记录最好的checkpoints
        if val_metric_glob_sum > best_glob_acc:
            torch.save(model.state_dict(), TrainConfig.WEIGHTS + "Best_{}.pth".format(epoch))

        # 打印Epoch级别日志
        info_print = (epoch,loss_total_sum/step,metric_local_sum/step,metric_glob_sum/step,
                val_loss_total_sum/val_step,val_metric_local_sum/val_step,val_metric_glob_sum/val_step)
        print(("\nEPOCH = %d, loss_total = %.3f," + 'local_'+ metric_name + " = %.3f," + "glob_"+metric_name+" = %.3f," + \
               "val_loss = %.3f," + "val_local_" + metric_name + " = %.3f," + "val_glob_"+metric_name + ' = %.3f')
              % info_print)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)

    writer.close()
    print("Finished Training...")
    return dfhistory


if __name__ == '__main__':
    model = LastEmb()
    if TrainConfig.GPU_USED:
        model.cuda()
    model.optimizer = torch.optim.Adam(model.parameters(),lr=TrainConfig.LR, weight_decay=TrainConfig.WEIGHT_DECAY)
    model.criterion = torch.nn.CrossEntropyLoss()
    model.metric_func = torchmetrics.Accuracy()
    model.metric_name = "ACC"
    if TrainConfig.GPU_USED:
        model.metric_func = model.metric_func.cuda()
    train_data = myDataSet(date=SqlConfig.DATE, repeat=TrainConfig.REPEAT,mode='train')
    print("Train Dataset运行结束！")
    train_loader = DataLoaderX(dataset=train_data, batch_size=TrainConfig.BATCH_SIZE, shuffle=TrainConfig.SHUFFLE, collate_fn=collate_fn)
    print("Train Loader运行结束！")
    val_data = myDataSet(date=SqlConfig.DATE, repeat=TrainConfig.REPEAT,mode='test')
    print("Valid Dataset运行结束！")
    val_loader = DataLoaderX(dataset=val_data, batch_size=TrainConfig.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print("All DataLoader prepared! Please wait for training !")
    dfhistory = train(model, epochs=TrainConfig.EPOCHS, train_loader=train_loader, val_loader=val_loader, log_step_freq= TrainConfig.LOG_STEP_FREQ)
    dfhistory.to_csv(TrainConfig.LOGFILE)







