import datetime
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import TensorDataset, Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchkeras import summary, Model

from sklearn.metrics import auc, roc_auc_score, roc_curve
from Deep_Crossing_model import get_Deep_Crossing

# auc标准的相关设置
def auc(y_pred, y_true):
    pred = y_pred.data
    y = y_true.data
    return roc_auc_score(y, pred)     # 计算AUC， 但要注意如果y只有一个类别的时候， 会报错

def data_load_train():
    # 导入数据， 数据已经处理好了 preprocess/下
    train_set = pd.read_csv('preprocessed_data/train_set.csv')
    val_set = pd.read_csv('preprocessed_data/val_set.csv')
    test_set = pd.read_csv('preprocessed_data/test.csv')
    # 这里需要把特征分成数值型和离散型， 因为后面的模型里面离散型的特征需要embedding， 而数值型的特征直接进入了stacking层， 处理方式会不一样
    data_df = pd.concat((train_set, val_set, test_set))

    dense_feas = ['I'+str(i) for i in range(1, 14)]
    sparse_feas = ['C'+str(i) for i in range(1, 27)]

    # 定义一个稀疏特征的embedding映射， 字典{key: value}, key表示每个稀疏特征， value表示数据集data_df对应列的不同取值个数， 作为embedding输入维度
    sparse_feas_map = {}
    for key in sparse_feas:
        sparse_feas_map[key] = data_df[key].nunique()
    feature_info = [dense_feas, sparse_feas, sparse_feas_map]  # 这里把特征信息进行封装， 建立模型的时候作为参数传入

    # 把数据构建成数据管道
    dl_train_dataset = TensorDataset(torch.tensor(train_set.drop(columns='Label').values).float(), torch.tensor(train_set['Label'].values).float())
    dl_val_dataset = TensorDataset(torch.tensor(val_set.drop(columns='Label').values).float(), torch.tensor(val_set['Label'].values).float())

    dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=16)
    dl_vaild = DataLoader(dl_val_dataset, shuffle=True, batch_size=16)

    net = get_Deep_Crossing(feature_info=feature_info,hidden_unit=[256, 128, 64, 32])
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    metric_func = auc
    metric_name = 'auc'

    epochs = 4
    log_step_freq = 10

    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
    print('Start Training...')
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('========='*8 + "%s" %nowtime)

    epochs = 4
    log_step_freq = 10

    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
    print('Start Training...')
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('========='*8 + "%s" %nowtime)

    best_pred_auc = 0.0
    for epoch in range(1, epochs+1):
        # 训练阶段
        net.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        
        for step, (features, labels) in enumerate(dl_train, 1):
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 正向传播
            predictions = net(features)
            loss = loss_func(predictions, labels)
            try:          # 这里就是如果当前批次里面的y只有一个类别， 跳过去
                metric = metric_func(predictions, labels)
            except ValueError:
                pass
            
            # 反向传播求梯度
            loss.backward()
            optimizer.step()
            
            # 打印batch级别日志
            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                    (step, loss_sum/step, metric_sum/step))
        
        # 验证阶段
        net.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1
        
        for val_step, (features, labels) in enumerate(dl_vaild, 1):
            with torch.no_grad():
                predictions = net(features)
                val_loss = loss_func(predictions, labels)
                try:
                    val_metric = metric_func(predictions, labels)
                except ValueError:
                    pass
            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()
        
        # 记录日志
        info = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info
        
        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
            "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
            %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)
        new_pred = val_metric_sum/val_step
        if new_pred>best_pred_auc:
            best_pred_auc = new_pred
            torch.save(net.state_dict(), './model/net_{}_parameter.pkl'.format(epoch))        
    print('Finished Training...')

def predict():
    train_set = pd.read_csv('preprocessed_data/train_set.csv')
    val_set = pd.read_csv('preprocessed_data/val_set.csv')
    test_set = pd.read_csv('preprocessed_data/test.csv')
    # 这里需要把特征分成数值型和离散型， 因为后面的模型里面离散型的特征需要embedding， 而数值型的特征直接进入了stacking层， 处理方式会不一样
    data_df = pd.concat((train_set, val_set, test_set))

    dense_feas = ['I'+str(i) for i in range(1, 14)]
    sparse_feas = ['C'+str(i) for i in range(1, 27)]

    # 定义一个稀疏特征的embedding映射， 字典{key: value}, key表示每个稀疏特征， value表示数据集data_df对应列的不同取值个数， 作为embedding输入维度
    sparse_feas_map = {}
    for key in sparse_feas:
        sparse_feas_map[key] = data_df[key].nunique()
    feature_info = [dense_feas, sparse_feas, sparse_feas_map]  # 这里把特征信息进行封装， 建立模型的时候作为参数传入

    net = get_Deep_Crossing(feature_info=feature_info,hidden_unit=[256, 128, 64, 32])
    net.load_state_dict(torch.load('./model/net_parameter.pkl'))
    y_pred_probs = net(torch.tensor(test_set.values).float())
    y_pred = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    print(y_pred)

if __name__ == "__main__":
    data__load_train()