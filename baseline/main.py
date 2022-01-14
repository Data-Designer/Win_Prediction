from baseline.model import clf,MLP,LSTMTag,Transformer,CNNTag,ConvLstmClassifier,TemporalConvNet
from catboost import CatBoostClassifier
from config.config import SqlConfig,opt
from utils import gridsearch
from sklearn.metrics import f1_score,accuracy_score
from baseline.dataset import transStandard,myDataSetbs
from data.dataset import myDataSet,DataLoaderX,collate_fn
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt
import torch
import torchmetrics
import datetime
import pandas as pd
import numpy as np
from pytorch_lightning import seed_everything
import torch.multiprocessing # 不然dataloader要报错
torch.multiprocessing.set_sharing_strategy('file_system') # to many open error,有可能open比较吃内存

# 设置随机种子
seed_everything(opt.SEED)

# 暂时使用的都是global的东西

def modelConfig():
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR, weight_decay=opt.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    metric_func = torchmetrics.Accuracy()
    metric_name = "ACC"
    # 其他指标,全局是两个指标
    metric_name2_g = "Mic-F1"
    metric_func2_g = torchmetrics.F1(num_classes=3).cuda() # 改了dataset,metric,glob,model最后输出
    metric_name3_g = "Macro-F1"
    metric_func3_g = torchmetrics.F1(num_classes=3,average='macro').cuda()
    # metric_name2_g = "F1"
    # metric_func2_g = torchmetrics.F1(num_classes=2)
    # metric_name3_g = "AUCROC"  # AUROC会缓存
    # metric_func3_g = torchmetrics.AUROC(num_classes=2,pos_label=0)
    return optimizer,criterion,metric_func,metric_name,metric_name2_g,metric_func2_g,metric_name3_g,metric_func3_g


def modelProcess(home_o, away_o, home_d, away_d):
    """
    func：元batch数据处理为->model接受的格式
    """
    home_o = torch.FloatTensor(np.stack(home_o))
    home_d = torch.FloatTensor(np.stack(home_d))
    away_o = torch.FloatTensor(np.stack(away_o))
    away_d = torch.FloatTensor(np.stack(away_d))
    return home_o,away_o,home_d,away_d


def catboost(clf,X_train,y_train,X_test,y_test,score='accuracy',grid= False):
    # 首次超参数搜索
    if grid:
        learning_rate = [0.01, 0.05, 0.1, 0.2] # 0.2
        iterations = [200]
        depth = [4,7,10] # 7
        tuned_parameters = dict(learning_rate=learning_rate, iterations=iterations, depth=depth)
        model = gridsearch(X_train, y_train, clf, tuned_parameters, scoring=score)
    else:
        model = CatBoostClassifier(learning_rate=0.2,iterations=200,depth=7)
        model.fit(X_train, y_train)
    # 查看一下特征重要性
    fea_ = model.feature_importances_
    fea_name = model.feature_names_
    plt.figure(figsize=(10, 10))
    plt.barh(fea_name, fea_, height=0.5)
    plt.show()

    model.save_model("/data/Win_Prediction/logbaseline/checkpoint/model")
    predict = model.predict(X_test)
    acc = accuracy_score(predict, y_test)
    return acc


def mlpTrainStep(model,features,labels):
    """
    mlp的train step
    :param model:
    :param feature:
    :param label:
    :return:
    """
    model.train()
    model.optimizer.zero_grad()
    predictions = model(features)
    loss = model.criterion(predictions,labels)
    metric = model.metric_func(predictions.softmax(dim=-1),labels)
    metric2 = model.metric_func2_g(predictions.softmax(dim=-1),labels)
    metric3 = model.metric_func3_g(predictions.softmax(dim=-1),labels)
    loss.backward()
    model.optimizer.step()
    return loss.item(), metric.item(),metric2.item(),metric3.item()


def mlpValidStep(model,features,labels):
    """
    mlp的valid step
    :param model:
    :param feature:
    :param label:
    :return:
    """
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        loss = model.criterion(predictions,labels)
        metric = model.metric_func(predictions.softmax(dim=-1),labels)
        metric2 = model.metric_func2_g(predictions.softmax(dim=-1), labels)
        metric3 = model.metric_func3_g(predictions.softmax(dim=-1), labels)
    return loss.item(),metric.item(),metric2.item(),metric3.item()


def lstmTrainStep(model,features,labels):
    """
    lstm和transformer公用train_step
    :param model:
    :param features:
    :param labels:
    :return:
    """
    model.train()
    model.optimizer.zero_grad()
    gameplayid, seq_static, static = features
    seq_static,static = torch.FloatTensor(np.stack(seq_static,axis=0)),torch.FloatTensor(np.stack(static,axis=0))
    if opt.GPU_USED:
        seq_static,static = seq_static.cuda(),static.cuda()
    predictions = model(seq_static,static)
    loss = model.criterion(predictions, labels)
    # print(labels.device)
    # print(predictions.device)
    metric = model.metric_func(predictions.softmax(dim=-1), labels)
    metric2 = model.metric_func2_g(predictions.softmax(dim=-1), labels)
    metric3 = model.metric_func3_g(predictions.softmax(dim=-1), labels)
    loss.backward()
    model.optimizer.step()
    return loss.item(), metric.item(),metric2.item(),metric3.item()


def lstmValidStep(model,features,labels):
    """
    lstm和transformer公用valid step
    :param model:
    :param features:
    :param labels:
    :return:
    """
    model.eval()
    with torch.no_grad():
        gameplayid, seq_static, static = features
        seq_static, static = torch.FloatTensor(np.stack(seq_static, axis=0)), torch.FloatTensor(
            np.stack(static, axis=0))
        if opt.GPU_USED:
            seq_static, static = seq_static.cuda(), static.cuda()
        predictions = model(seq_static, static)
        loss = model.criterion(predictions,labels)
        metric = model.metric_func(predictions.softmax(dim=-1),labels)
        metric2 = model.metric_func2_g(predictions.softmax(dim=-1), labels)
        metric3 = model.metric_func3_g(predictions.softmax(dim=-1), labels)
    return loss.item(), metric.item(),metric2.item(),metric3.item()


def convlstmTrainStep(model,features,labels):
    model.train()
    model.optimizer.zero_grad()
    gameplayid, home_o, away_o, home_d, away_d = features
    home_o, away_o, home_d, away_d = modelProcess(home_o, away_o, home_d, away_d)
    if opt.GPU_USED:
        home_o, away_o, home_d, away_d = home_o.cuda(), away_o.cuda(), home_d.cuda(), away_d.cuda()
    predictions = model(home_o, away_o, home_d, away_d)
    loss = model.criterion(predictions, labels)
    metric = model.metric_func(predictions.softmax(dim=-1), labels)
    metric2 = model.metric_func2_g(predictions.softmax(dim=-1),labels)
    metric3 = model.metric_func3_g(predictions.softmax(dim=-1),labels)
    loss.backward()
    model.optimizer.step()
    return loss.item(), metric.item(),metric2.item(),metric3.item()


def convlstmValidStep(model,features,labels):
    model.eval()
    with torch.no_grad():
        gameplayid, home_o, away_o, home_d, away_d = features
        home_o, away_o, home_d, away_d = modelProcess(home_o, away_o, home_d, away_d)
        if opt.GPU_USED:
            home_o, away_o, home_d, away_d = home_o.cuda(), away_o.cuda(), home_d.cuda(), away_d.cuda()
        predictions = model(home_o, away_o, home_d, away_d)
        loss = model.criterion(predictions,labels)
        metric = model.metric_func(predictions.softmax(dim=-1),labels)
        metric2 = model.metric_func2_g(predictions.softmax(dim=-1), labels)
        metric3 = model.metric_func3_g(predictions.softmax(dim=-1), labels)
    return loss.item(), metric.item(),metric2.item(),metric3.item()


if __name__ == '__main__':
    baselines = ['mlp','lstm','textcnn','tempcnn','transformer','convlstm']
    # baselines = ['convlstm']
    X_train, y_train, X_test, y_test = transStandard(SqlConfig.DATE)
    for baseline in baselines:
        # 这是dot数据
        if baseline == "catboost":
            print("Next catboost baseline experiment !")
            acc = catboost(clf,X_train,y_train,X_test,y_test)
            print("准确率为",acc)
        elif baseline == "mlp":
            print("Next MLP baseline experiment !")
            # 模型定义
            model = MLP()
            if opt.GPU_USED:
                model.cuda()
            model.optimizer, model.criterion, model.metric_func,model.metric_name,model.metric_name2_g,model.metric_func2_g,model.metric_name3_g,model.metric_func3_g=modelConfig()
            if opt.GPU_USED:
                model.metric_func =model.metric_func.cuda() # 十分重要
            dfhistory = pd.DataFrame(columns=["epoch", "loss", model.metric_name, model.metric_name2_g,model.metric_name3_g,
                                              "val_loss", "val_" + model.metric_name,"val_"+ model.metric_name2_g,"val_"+ model.metric_name3_g," best_" + model.metric_name])
            # 数据准备
            best_acc = 0.0
            X_train,y_train,X_test,y_test = torch.FloatTensor(X_train),torch.LongTensor(y_train),torch.FloatTensor(X_test),torch.LongTensor(y_test)
            train_dataset = TensorDataset(X_train,y_train)
            test_dataset = TensorDataset(X_test,y_test)
            trainloader = DataLoaderX(train_dataset,batch_size=opt.BATCH_SIZE,pin_memory=False,num_workers=8,shuffle=True)
            testloader = DataLoaderX(test_dataset,batch_size=opt.BATCH_SIZE,pin_memory=False,num_workers=8,shuffle=False)
            # 模型训练
            print("Start Training...")
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("==========" * 8 + "%s" % nowtime)
            for epoch in range(1,opt.EPOCHS+1):
                loss_sum,metric_sum,step = 0.0,0.0,1
                # 补充
                metric_2_sum,metric_3_sum = 0.0,0.0
                for step,(features,labels) in enumerate(trainloader,1):
                    labels = labels - 1
                    if opt.GPU_USED:
                        features = features.cuda()
                        labels = labels.cuda()
                    loss,metric,metric_2,metric_3 = mlpTrainStep(model,features,labels)
                    loss_sum += loss
                    metric_sum += metric
                    metric_2_sum += metric_2
                    metric_3_sum += metric_3
                    if step % opt.LOG_STEP_FREQ == 0:
                        print(("[step = %d] loss: %.3f, " + model.metric_name + ": %.3f") %
                              (step, loss_sum / step, metric_sum / step))

                val_loss_sum, val_metric_sum, val_step = 0.0, 0.0, 1
                val_metric2_sum,val_metric3_sum = 0.0,0.0
                for val_step,(features,labels) in enumerate(testloader,1):
                    labels = labels -1
                    if opt.GPU_USED:
                        labels = labels.cuda()
                        features = features.cuda()
                    val_loss,val_metric,val_metric_2,val_metric_3 = mlpValidStep(model,features,labels)
                    val_loss_sum += val_loss
                    val_metric_sum += val_metric
                    val_metric2_sum += val_metric_2
                    val_metric3_sum += val_metric_3
                # 记录日志
                if val_metric_sum/val_step > best_acc:
                    best_acc =val_metric_sum/val_step

                info = (epoch, loss_sum / step, metric_sum / step, metric_2_sum/step,metric_3_sum/step,
                        val_loss_sum / val_step, val_metric_sum / val_step,val_metric2_sum/val_step,val_metric3_sum/val_step,best_acc)
                dfhistory.loc[epoch - 1] = info
                info_print = (epoch, loss_sum / step, metric_sum / step,
                        val_loss_sum / val_step, val_metric_sum / val_step,best_acc)
                print(("\nEPOCH = %d, loss = %.3f," + model.metric_name + \
                       "  = %.3f, val_loss = %.3f, " + "val_" + model.metric_name + " = %.3f" + "best_" + model.metric_name + " = %.3f" )
                      % info_print)
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("\n" + "==========" * 8 + "%s" % nowtime)
            # 训练结束
            print('Finished Training...')
            dfhistory.to_csv(opt.MLPLOGFILE)
            print("MLP LOG FILE HAVE SAVED!")

        # 这是seq数据,lstm和trans的输入完全一致
        elif baseline == "lstm" or "transformer" or "textcnn" or "tempcnn":
            print("Next {} baseline experiment !".format(str.upper(baseline)))
            # 模型定义
            if baseline == "lstm":
                model = LSTMTag()
            elif baseline == "trannsformer":
                model = Transformer()
            elif baseline == 'tempcnn':
                model = TemporalConvNet()
            else:
                model = CNNTag()
            if opt.GPU_USED:
                model.cuda()
            model.optimizer, model.criterion, model.metric_func,model.metric_name,model.metric_name2_g,model.metric_func2_g,model.metric_name3_g,model.metric_func3_g=modelConfig()
            dfhistory = pd.DataFrame(
                columns=["epoch", "loss", model.metric_name, model.metric_name2_g, model.metric_name3_g,
                         "val_loss", "val_" + model.metric_name, "val_" + model.metric_name2_g,
                         "val_" + model.metric_name3_g, " best_" + model.metric_name])
            best_acc = 0.0
            # 数据准备
            train_dataset = myDataSetbs(SqlConfig.DATE, mode='train')
            test_dataset = myDataSetbs(SqlConfig.DATE, mode='test')
            trainloader = DataLoaderX(train_dataset, batch_size=opt.BATCH_SIZE,pin_memory=False,num_workers=0,shuffle=True,collate_fn=collate_fn,drop_last=True)
            testloader = DataLoaderX(test_dataset, batch_size=opt.BATCH_SIZE,pin_memory=False,num_workers=0,shuffle=False,collate_fn=collate_fn,drop_last=True)
            # 模型训练
            print("Start Training...")
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("==========" * 8 + "%s" % nowtime)
            for epoch in range(1,opt.EPOCHS):
                loss_sum,metric_sum,step = 0.0,0.0,1
                metric_2_sum,metric_3_sum = 0.0,0.0
                for step,(features,labels) in enumerate(trainloader,1):
                    gameplayid, seq_static, static, local_h, local_a = [features[:, i] for i in range(5)]  # 解析feature
                    re_features = (gameplayid, seq_static, static)
                    # glob_labels = (labels.reshape(-1, ).long() - 1).cuda()  # 需要longtensor
                    glob_labels =torch.LongTensor(local_h.astype(int)).cuda()
                    loss, metric,metric_2,metric_3 = lstmTrainStep(model, re_features, glob_labels)
                    loss_sum += loss
                    metric_sum += metric
                    metric_2_sum += metric_2
                    metric_3_sum += metric_3
                    if step % opt.LOG_STEP_FREQ == 0:
                        print(("[step = %d] loss: %.3f, " + model.metric_name + ": %.3f") %
                              (step, loss_sum / step, metric_sum / step))

                val_loss_sum, val_metric_sum, val_step = 0.0, 0.0, 1
                val_metric2_sum,val_metric3_sum = 0.0,0.0
                for val_step, (features, labels) in enumerate(testloader, 1):
                    gameplayid, seq_static, static, local_h, local_a = [features[:, i] for i in range(5)]  # 解析feature
                    re_features = (gameplayid, seq_static, static)
                    glob_labels = (labels.reshape(-1, ).long() - 1).cuda()  # 需要longtensor
                    val_loss, val_metric,val_metric_2,val_metric_3 = lstmValidStep(model, re_features, glob_labels)
                    val_loss_sum += val_loss
                    val_metric_sum += val_metric
                    val_metric2_sum += val_metric_2
                    val_metric3_sum += val_metric_3
                    # 记录日志
                if val_metric_sum / val_step > best_acc:
                    best_acc = val_metric_sum / val_step

                info_print = (epoch, loss_sum / step, metric_sum / step,
                        val_loss_sum / val_step, val_metric_sum / val_step, best_acc)
                info = (epoch, loss_sum / step, metric_sum / step, metric_2_sum/step,metric_3_sum/step,
                        val_loss_sum / val_step, val_metric_sum / val_step,val_metric2_sum/val_step,val_metric3_sum/val_step,best_acc)
                dfhistory.loc[epoch - 1] = info
                print(("\nEPOCH = %d, loss = %.3f," + model.metric_name + \
                       "  = %.3f, val_loss = %.3f, " + "val_" + model.metric_name + " = %.3f" + "best_" + model.metric_name + " = %.3f")
                      % info_print)
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("\n" + "==========" * 8 + "%s" % nowtime)
            # 训练结束
            print('Finished Training...')
            if baseline =="lstm":
                dfhistory.to_csv(opt.LSTMLOGFILE)
            elif baseline =="textcnn":
                dfhistory.to_csv(opt.TEXTCNNLOGFILE)
            elif baseline =='tempcnn':
                dfhistory.to_csv(opt.TEMPCNNLOGFILE)
            elif baseline =='transformer':
                dfhistory.to_csv(opt.TRANSLOGFILE)
            print(str.upper(baseline)+" LOG FILE HAVE SAVED!")

        elif baseline =='convlstm':
            print("Next CONVLSTM baseline experiment !")
            # 模型定义
            model = ConvLstmClassifier()
            if opt.GPU_USED:
                model.cuda()
            model.optimizer, model.criterion, model.metric_func,model.metric_name,model.metric_name2_g,model.metric_func2_g,model.metric_name3_g,model.metric_func3_g=modelConfig()
            dfhistory = pd.DataFrame(
                columns=["epoch", "loss", model.metric_name, model.metric_name2_g, model.metric_name3_g,
                         "val_loss", "val_" + model.metric_name, "val_" + model.metric_name2_g,
                         "val_" + model.metric_name3_g, " best_" + model.metric_name])
            best_acc = 0.0
            # 数据准备
            train_dataset = myDataSet(SqlConfig.DATE,mode='train')
            test_dataset = myDataSet(SqlConfig.DATE,mode='test')
            trainloader = DataLoaderX(train_dataset, batch_size=opt.BATCH_SIZE,collate_fn=collate_fn,num_workers=0,pin_memory=False,shuffle=True,drop_last=True)
            testloader = DataLoaderX(test_dataset, batch_size=opt.BATCH_SIZE,collate_fn=collate_fn,num_workers=0,pin_memory=False,shuffle=False,drop_last=True)
            # 模型训练
            print("Start Training...")
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("==========" * 8 + "%s" % nowtime)
            for epoch in range(1, opt.EPOCHS):
                loss_sum, metric_sum, step = 0.0, 0.0, 1
                metric_2_sum, metric_3_sum = 0.0, 0.0
                for step, (features, labels) in enumerate(trainloader, 1):
                    gameplayid, _, _, home_o, away_o, home_d, away_d, _, _, local_h, _ = [
                        features[:, i] for i in range(11)]  # 解析feature
                    re_features = (gameplayid, home_o, away_o, home_d, away_d)
                    # glob_labels = (labels.reshape(-1, ).long() - 1).cuda()  # 需要longtensor
                    glob_labels =torch.LongTensor(local_h.astype(int)).cuda()

                    loss, metric, metric_2,metric_3 = convlstmTrainStep(model, re_features, glob_labels)
                    loss_sum += loss
                    metric_sum += metric
                    metric_2_sum += metric_2
                    metric_3_sum += metric_3
                    if step % opt.LOG_STEP_FREQ == 0:
                        print(("[step = %d] loss: %.3f, " + model.metric_name + ": %.3f") %
                              (step, loss_sum / step, metric_sum / step))

                val_loss_sum, val_metric_sum, val_step = 0.0, 0.0, 1
                val_metric2_sum,val_metric3_sum = 0.0,0.0
                for val_step, (features, labels) in enumerate(testloader, 1):
                    gameplayid, _, _, home_o, away_o, home_d, away_d, _, _, _, _ = [
                        features[:, i] for i in range(11)]  # 解析feature
                    re_features = (gameplayid, home_o, away_o, home_d, away_d)
                    glob_labels = (labels.reshape(-1, ).long() - 1).cuda()  # 需要longtensor
                    val_loss, val_metric,val_metric_2,val_metric_3 = convlstmValidStep(model, re_features, labels)
                    val_loss_sum += val_loss
                    val_metric_sum += val_metric
                    val_metric2_sum += val_metric_2
                    val_metric3_sum += val_metric_3
                    # 记录日志
                if val_metric_sum / val_step > best_acc:
                    best_acc = val_metric_sum / val_step
                info = (epoch, loss_sum / step, metric_sum / step, metric_2_sum/step,metric_3_sum/step,
                        val_loss_sum / val_step, val_metric_sum / val_step,val_metric2_sum/val_step,val_metric3_sum/val_step,best_acc)
                info_print = (epoch, loss_sum / step, metric_sum / step,
                        val_loss_sum / val_step, val_metric_sum / val_step, best_acc)
                dfhistory.loc[epoch - 1] = info
                print(("\nEPOCH = %d, loss = %.3f," + model.metric_name + \
                       "  = %.3f, val_loss = %.3f, " + "val_" + model.metric_name + " = %.3f" + "best_" + model.metric_name + " = %.3f")
                      % info_print)
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("\n" + "==========" * 8 + "%s" % nowtime)
            # 训练结束
            print('Finished Training...')
            dfhistory.to_csv(opt.CONLSTMLOGFILE) # 不知道为啥就是存不了

            print("ConvLstm LOG FILE HAVE SAVED!")

    print("All baselines have been done!")










