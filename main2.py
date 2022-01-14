# 主训练文件
import os
import torch
import pandas as pd
import datetime
import torchmetrics
import copy
import gc
import fire
import torch.nn as nn
# from sklearn.metrics import roc_auc_score
# from torch.utils.data import DataLoader 预加载
from torch.utils.tensorboard import SummaryWriter
from utils import modelProcess,write_csv
from config.config import SqlConfig,opt
from model.model import LastEmb # 这里改变消融实验
from data.dataset import myDataSet,DataLoaderX,collate_fn
from pytorch_lightning import seed_everything
# import torch.autograd.profiler as profiler # 性能分析
import torch.multiprocessing # 不然dataloader要报错
torch.multiprocessing.set_sharing_strategy('file_system') # to many open error
# torch.multiprocessing.set_start_method('spawn', force=True) # 解决open cv bug


# 设置随机种子
seed_everything(opt.SEED)
writer = SummaryWriter(opt.TLOGFILE) # 定义Tensorboard句柄

# torch.backends.cudnn.benchmark = True # 提高卷积速度，但是会使得第一个epoch速度慢的令人发指
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.enabled = True


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
                                                                                              seq_static,static) # 去掉了static

    # print(static.shape)
    if opt.GPU_USED:
        h_trajs, h_count_num, a_trajs, a_count_num, home_o, away_o, home_d, away_d,static = h_trajs.cuda(),h_count_num,\
                                                                                             a_trajs.cuda(),a_count_num,home_o.cuda(),\
                                                                                  away_o.cuda(),home_d.cuda(),away_d.cuda(),static.cuda()

    # 正向计算
    # pred_local,pred_glob = model(h_trajs, h_count_num, a_trajs, a_count_num, home_o, home_d, away_o, away_d, static)

    # 以下是model部分，需要看att weights
    pred_local,pred_glob,_,_ = model(h_trajs, h_count_num, a_trajs, a_count_num, home_o, home_d, away_o, away_d, static)

    # 计算loss
    loss_local = model.criterion(pred_local,local_labels)
    loss_glob = model.criterion(pred_glob,glob_labels) # 这里应该为0和1才对！原来的label标签是1和2
    loss_total = loss_local*opt.LOCAL_WEIGHT + loss_glob*opt.GLOB_WEIGHT
    # metric_local = model.metric_func(pred_local.softmax(dim=-1), local_labels)
    # metric_local2 = model.metric_func2_l(pred_local.softmax(dim=-1), local_labels)
    # metric_local3 =model.metric_func3_l(pred_local.softmax(dim=-1), local_labels)
    # metric_glob = model.metric_func(pred_glob.softmax(dim=-1),glob_labels) # 这个指标需要soft
    # metric_glob2 = model.metric_func2_g(pred_glob.softmax(dim=-1),glob_labels)
    # model.metric_func3_g.update(pred_glob.softmax(dim=-1), glob_labels) # train的时候无需存储这些指标
    # metric_glob3 = model.metric_func3_g.compute()
    # 反向传播
    loss_total.backward()
    model.optimizer.step()
    return loss_total.detach().item(),loss_local.detach().item(),loss_glob.detach().item(),0,0,0,0,0,0#,metric_local.item(),metric_local2.item(),metric_local3.item() ,metric_glob.item(),metric_glob2.item(),metric_glob3.item()



def valid_step(model,features,local_labels,glob_labels):
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        # 处理为model可以接受的格式
        gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static = features
        h_trajs, h_count_num, a_trajs, a_count_num, home_o, away_o, home_d, away_d,static = modelProcess(trajs_1,trajs_2,
                                                                                                          home_o, away_o,
                                                                                                          home_d,away_d,
                                                                                                          seq_static,static)
        if opt.GPU_USED:
            h_trajs, h_count_num, a_trajs, a_count_num, home_o, away_o, home_d, away_d, static = h_trajs.cuda(), h_count_num, \
                                                                                                 a_trajs.cuda(), a_count_num, home_o.cuda(), \
                                                                                                 away_o.cuda(), home_d.cuda(), away_d.cuda(), static.cuda()
        # 正向计算
        # pred_local,pred_glob = model(h_trajs, h_count_num, a_trajs, a_count_num, home_o, home_d, away_o, away_d, static)

        pred_local,pred_glob,_,_ = model(h_trajs, h_count_num, a_trajs, a_count_num, home_o, home_d, away_o, away_d, static)
        # 计算loss
        loss_local = model.criterion(pred_local, local_labels)
        loss_glob = model.criterion(pred_glob, glob_labels)
        loss_total = loss_local*opt.LOCAL_WEIGHT + loss_glob*opt.GLOB_WEIGHT
        metric_local = model.metric_func(pred_local.softmax(dim=-1), local_labels)
        metric_local2 = model.metric_func2_l(pred_local.softmax(dim=-1), local_labels)
        metric_local3 = model.metric_func3_l(pred_local.softmax(dim=-1), local_labels)
        metric_glob = model.metric_func(pred_glob.softmax(dim=-1), glob_labels)  # 这个指标需要soft
        metric_glob2 = model.metric_func2_g(pred_glob.softmax(dim=-1), glob_labels)
        model.metric_func3_g.update(pred_glob.softmax(dim=-1), glob_labels)
        metric_glob3 = model.metric_func3_g.compute()
    return loss_total.detach().item(), loss_local.detach().item(), loss_glob.detach().item(), metric_local.item(), metric_local2.item(), metric_local3.item(), metric_glob.item(), metric_glob2.item(), metric_glob3.item()


def case(**kwargs):
    """
    其实就是测试,做case study
    :param kwargs:
    :return:
    """
    opt.parse(kwargs)
    model = LastEmb()
    results = [] # 这里存储case结果
    pth = os.listdir(opt.WEIGHTS)
    pth.sort(key=lambda x: int(x.split('_')[1][:-4]))
    model_best = pth[-1] # 取最好的一个weight load
    model_dict = torch.load(opt.WEIGHTS + model_best)
    model.load_state_dict(model_dict)  #加载参数
    if opt.GPU_USED:
        model.cuda()
    print("Start Casing...")
    case_data = myDataSet(date=SqlConfig.DATE, repeat=opt.REPEAT,mode='test') # 这里暂时使用val的数据进行case
    case_loader = DataLoaderX(dataset=case_data, batch_size=opt.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,drop_last=True)
    print("CASE LOADER FINISH! ")
    model.eval()
    with torch.no_grad():
        for step, (features, _) in enumerate(case_loader): # 只经历一个epoch
            # 获取数据
            gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static, _, _ = [features[:, i] for i in range(11)]
            h_trajs, h_count_num, a_trajs, a_count_num, home_o, away_o, home_d, away_d, static = modelProcess(trajs_1,trajs_2,
                                                                                                              home_o,away_o,home_d,
                                                                                                              away_d,seq_static,static)
            if opt.GPU_USED:
                h_trajs, h_count_num, a_trajs, a_count_num, home_o, away_o, home_d, away_d, static = h_trajs.cuda(), h_count_num, \
                                                                                                     a_trajs.cuda(), a_count_num, home_o.cuda(), \
                                                                                                     away_o.cuda(), home_d.cuda(), away_d.cuda(), static.cuda()
            # 正向计算
            pred_local, pred_glob,h_batch_attn,a_batch_attn = model(h_trajs, h_count_num, a_trajs, a_count_num, home_o, home_d, away_o, away_d, static)
            # 存储数据,home的数据！
            h_local_prob = pred_local.softmax(dim=-1)[:,0].tolist()
            h_pred_glob = pred_glob.softmax(dim=-1)[:,0].tolist()
            a_local_prob = pred_local.softmax(dim=-1)[:, 1].tolist()
            a_pred_glob = pred_glob.softmax(dim=-1)[:, 1].tolist()
            # 这里需要加上gameplayid用于区分
            batch_results = list(zip(gameplayid,h_local_prob,h_pred_glob,a_local_prob,a_pred_glob,h_batch_attn,a_batch_attn)) # 记住这个顺序
            results += batch_results
        write_csv(results, opt.CASEFILE)
    print("Finish CASING...")
    return results


def train(**kwargs):
    # 解析自定义参数
    opt.parse(kwargs)

    # 模型定义
    model = LastEmb()
    if opt.GPU_USED:
        model.cuda()
    if opt.PARALLEL:
        model = nn.DataParallel(model) # 不能使用并行
    model.optimizer = torch.optim.Adam(model.parameters(),lr=opt.LR, weight_decay=opt.WEIGHT_DECAY)
    model.criterion = torch.nn.CrossEntropyLoss()
    model.metric_func = torchmetrics.Accuracy()
    model.metric_name = "ACC"

    # 增加模型指标
    model.metric_name2_g = "F1"
    model.metric_func2_g = torchmetrics.F1(num_classes=2)
    model.metric_name3_g = "AUROC" # AUROC会缓存,这里每个epoch重新释放（但是不释放也说得通）
    # model.metric_func3_g = torchmetrics.AUROC(num_classes=2,pos_label=0) # 因为此时在判home队伍
    # model.metric_name3_g = "Recall" # AUROC会缓存
    # model.metric_func3_g = torchmetrics.Recall(average='micro', num_classes=2)

    model.metric_name2_l = "Mic-F1"
    model.metric_func2_l = torchmetrics.F1(num_classes=3)
    model.metric_name3_l = "Macro-F1"
    model.metric_func3_l = torchmetrics.F1(num_classes=3,average='macro')
    """
    glob的一些备选指标:
    model.metric_func_glob1 = torchmetrics.F1(num_classes=2)
    model.metric_func_glob2 = torchmetrics.AUROC(num_classes=2,pos_label=0) # 因为此时在判home队伍
    local的一些备选指标：
    model.metric_func_local1 = torchmetrics.F1(num_classes=3) micro
    model.metric_func_local2 = torchmetrics.F1(num_classes=3,average='macro'),macro
    """


    if opt.GPU_USED:
        model.metric_func = model.metric_func.cuda()
        model.metric_func2_g = model.metric_func2_g.cuda()
        # model.metric_func3_g = model.metric_func3_g.cuda()
        model.metric_func2_l = model.metric_func2_l.cuda()
        model.metric_func3_l = model.metric_func3_l.cuda()




    train_data = myDataSet(date=SqlConfig.DATE, repeat=opt.REPEAT,mode='train4') # 这里可以变种
    print("Train Dataset Have Done!")
    train_loader = DataLoaderX(dataset=train_data, batch_size=opt.BATCH_SIZE, shuffle=opt.SHUFFLE, collate_fn=collate_fn,drop_last=True,pin_memory=False,num_workers=10) # 这里会报错设置多个numworks,设置为10最快
    print("Train Loader Have Done,{} Loaders!".format(len(train_loader))) # dataloader和cv2的爱恨情仇，不能同时使用多进程，不然要卡死
    # print('*****************************\n{}'.format(features))
    val_data = myDataSet(date=SqlConfig.DATE, repeat=opt.REPEAT,mode='test4') # 注意这里DataLoader也重写过
    val_loader = DataLoaderX(dataset=val_data, batch_size=opt.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,drop_last=True,pin_memory=False,num_workers=10) # pin_mem为True针对大数据，小的就算了吧
    print("Valid Dataset Have Done,{} Loaders!".format(len(val_loader)))
    print("All DataLoader prepared! Please wait for training !")

    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns=['epoch','loss_total','loss_local','loss_glob',
                                      'local_'+metric_name,'local_'+model.metric_name2_l,'local_'+model.metric_name3_l,
                                      'glob_'+metric_name,'glob_'+model.metric_name2_g,'glob_'+model.metric_name3_g,
                                      'val_loss_total','val_loss_local','val_loss_glob',
                                      'val_local_'+metric_name,'val_local_'+model.metric_name2_l,'val_local_'+model.metric_name3_l,
                                      'val_glob_'+metric_name,'val_glob_'+model.metric_name2_g,'val_glob_'+model.metric_name3_g])

    # 模型训练
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)
    # 其实这里的glob指标才是重要的
    best_glob_acc = 0.0

    for epoch in range(1,opt.EPOCHS+1):
        # 重新定义auc(每个epoch清缓存)
        model.metric_func3_g = torchmetrics.AUROC(num_classes=2, pos_label=0)  # 因为此时在判home队伍
        model.metric_func3_g = model.metric_func3_g.cuda() # 千万注意这里不要让train打开，不然train的数据也会被extend进去
        # 训练循环
        loss_total_sum = 0.0
        loss_local_sum = 0.0
        loss_glob_sum = 0.0
        metric_local_sum = 0.0
        metric_local_2_sum = 0.0
        metric_local_3_sum = 0.0
        metric_glob_sum = 0.0
        metric_glob_2_sum = 0.0
        metric_glob_3_sum = 0.0

        step = 1 # 每个epoch从头开始
        for step,(features,labels) in enumerate(train_loader,1):
            gameplayid,trajs_1,trajs_2,home_o,away_o,home_d,away_d,seq_static,static,local_h,_ = [features[:,i] for i in range(11)] # 解析feature
            re_features = (gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static)

            # 不知道有没有用，解决shared memory问题
            re_feature = copy.deepcopy(re_features)
            del re_features

            local_labels = torch.LongTensor(local_h.astype(int)).cuda() # 这里是预测home侧，预测away侧有些指标要取反！
            glob_labels = (labels.reshape(-1,).long() - 1).cuda() # 需要longtensor
            loss_total, loss_local, loss_glob, metric_local,metric_local_2,metric_local_3, metric_glob,metric_glob_2,metric_glob_3 = train_step(model,re_feature,local_labels,glob_labels)
            del gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static, local_labels, glob_labels
            loss_total_sum += loss_total
            loss_local_sum += loss_local
            loss_glob_sum += loss_glob
            metric_local_sum += metric_local
            metric_local_2_sum += metric_local_2
            metric_local_3_sum += metric_local_3
            metric_glob_sum += metric_glob
            metric_glob_2_sum += metric_glob_2
            metric_glob_3_sum += metric_glob_3
            # 使用tensorboard记录数据,真的是贼鸡儿慢
            del loss_total, loss_local, loss_glob, metric_local, metric_local_2, metric_local_3, metric_glob, metric_glob_2, metric_glob_3
            writer.add_scalar('Train Total Loss', loss_total_sum / step, (epoch - 1) * len(train_loader) + step)
            writer.add_scalar('Train Local Loss', loss_local_sum / step, (epoch - 1) * len(train_loader) + step)
            writer.add_scalar('Train Local '+ metric_name, metric_local_sum / step, (epoch - 1) * len(train_loader) + step)
            writer.add_scalar('Train Global Loss', loss_glob_sum / step, (epoch - 1) * len(train_loader) + step)
            writer.add_scalar('Train Global '+ metric_name, metric_glob_sum / step, (epoch - 1) * len(train_loader) + step)
            # 打印数据
            if step % opt.LOG_STEP_FREQ == 0:
                print(("[step = %d] loss_total: %.3f, " + "local_"+metric_name + ": %.3f, "
                       + "glob_"+metric_name + ": %.3f") %
                      (step, loss_total_sum / step, metric_local_sum / step, metric_glob_sum / step))
            # time.sleep(0.003) # 不知道能不能解决互锁问题
            # torch.cuda.empty_cache() # 妄图解决OOM问题,然而这个解决的是cuda的OOM


        # 验证循环
        val_loss_total_sum = 0.0
        val_loss_local_sum = 0.0
        val_loss_glob_sum = 0.0
        val_metric_local_sum = 0.0
        val_metric_local2_sum = 0.0
        val_metric_local3_sum = 0.0
        val_metric_glob_sum = 0.0
        val_metric_glob2_sum = 0.0
        val_metric_glob3_sum = 0.0
        val_step = 1
        for val_step,(features,labels) in enumerate(val_loader,1):
            gameplayid,trajs_1,trajs_2,home_o,away_o,home_d,away_d,seq_static,static,local_h,_ = [features[:,i] for i in range(11)]
            re_features = (gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static)
            # 妄图解决shared mem不足的问题

            re_feature = copy.deepcopy(re_features)
            del re_features

            local_labels = torch.LongTensor(local_h.astype(int)).cuda() # 这里是预测home侧，预测away侧无需变动大部分指标，但是local要取反
            glob_labels = (labels.reshape(-1,).long() - 1).cuda() # 分类会自动调整为one_hot格式,1,2->0,1
            loss_total, loss_local, loss_glob, metric_local,metric_local_2,metric_local_3, metric_glob,metric_glob_2,metric_glob_3 = valid_step(model, re_feature, local_labels,
                                                                                      glob_labels)
            del gameplayid, trajs_1, trajs_2, home_o, away_o, home_d, away_d, seq_static, static,local_labels,glob_labels
            val_loss_total_sum += loss_total
            val_loss_local_sum += loss_local
            val_loss_glob_sum += loss_glob
            val_metric_local_sum += metric_local
            val_metric_local2_sum += metric_local_2
            val_metric_local3_sum+= metric_local_3
            val_metric_glob_sum += metric_glob
            val_metric_glob2_sum += metric_glob_2
            val_metric_glob3_sum += metric_glob_3
            del loss_total,loss_local,loss_glob,metric_local,metric_local_2,metric_local_3,metric_glob,metric_glob_2# auc改,metric_glob_3
            # tensorboard记录数据
            writer.add_scalar('Val Total Loss', val_loss_total_sum / val_step, (epoch - 1) * len(val_loader) + val_step)
            writer.add_scalar('Val Local Loss', val_loss_local_sum / val_step, (epoch - 1) * len(val_loader) + val_step)
            writer.add_scalar('Val Local '+ metric_name, val_metric_local_sum / val_step, (epoch - 1) * len(val_loader) + val_step)
            writer.add_scalar('Val Global Loss', val_loss_glob_sum / val_step, (epoch - 1) * len(val_loader) + val_step)
            writer.add_scalar('Val Global '+ metric_name, val_metric_glob_sum / val_step, (epoch - 1) * len(val_loader) + val_step)

        # 记录Epoch日志
        info = (epoch,loss_total_sum/step,loss_local_sum/step,loss_glob_sum/step,
                metric_local_sum/step,metric_local_2_sum/step,metric_local_3_sum/step,
                metric_glob_sum/step,metric_glob_2_sum/step,metric_glob_3_sum/step,
                val_loss_total_sum/val_step,val_loss_local_sum/val_step,val_loss_glob_sum/val_step,
                val_metric_local_sum/val_step,val_metric_local2_sum/val_step,val_metric_local3_sum/val_step,
                val_metric_glob_sum/val_step,val_metric_glob2_sum/val_step,metric_glob_3)#这里因为auc进行改进val_metric_glob3_sum/val_step
        dfhistory.loc[epoch-1] = info
        del info # 减少memory
        # 记录最好的checkpoints,这个也慢的一笔
        if val_metric_glob_sum/val_step > best_glob_acc:
            torch.save(model.state_dict(), opt.WEIGHTS + "Best_{}.pth".format(epoch))
            best_glob_acc = val_metric_glob_sum/val_step

        # 打印Epoch级别日志
        info_print = (epoch,loss_total_sum/step,metric_local_sum/step,metric_glob_sum/step,
                val_loss_total_sum/val_step,val_metric_local_sum/val_step,val_metric_glob_sum/val_step)
        print(("\nEPOCH = %d, loss_total = %.3f," + 'local_'+ metric_name + " = %.3f," + "glob_"+metric_name+" = %.3f," + \
               "val_loss = %.3f," + "val_local_" + metric_name + " = %.3f," + "val_glob_"+metric_name + ' = %.3f')
              % info_print)
        del info_print
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)
        del model.metric_func3_g # AUC进行改进
        gc.collect() # 进行垃圾回收

    writer.close()
    print("Finished Training...")
    dfhistory.to_csv(opt.LOGFILE)
    print("LOG FILE HAVE SAVED!")


if __name__ == '__main__':
    fire.Fire() # 命令行运行
    # train()





