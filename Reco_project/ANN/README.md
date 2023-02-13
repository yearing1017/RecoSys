## ANN召回

### 关于项目

#### 背景

模型召回只有离线dssm_u2i召回，离线每个小时计算好user和item的embedding，然后离线进行ann检索，将每个用户最近邻的item存在redis中，供线上使用，user的请求，去查找计算好的topk结果

- 没有实时特征，不能满足user的实时个性化需求
- 由于某些特征在新用户和低活用户上构建不了，只能计算部分用户的embedding，用户覆盖不全
- 离线计算，时延较长

#### 实时ANN召回方案

- 模型实时训练，每小时保存一个模型
- 每小时构建faiss索引（包括item样本构建、item的embedding预估、faiss搭建索引）
- 线上用户实时请求，预估user侧embedding，并实时在faiss索引中进行ANN检索，返回topk结果

### 训练样本

#### 目标
- 单目标
- 多目标

#### 特征
- user侧
- item侧

#### 采样
- in-batch负采样
- 全局负采样

### 模型训练

#### 模型
- 单目标
- 多目标

#### 离线训练
- 天级训练
- 小时训练

#### 实时训练
- FLINK任务

### 离线评估

#### recall@N

#### loss & gauc
- log_loss
- ne
- auc
- gauc
- calibration

### 索引构建

#### 保证同一模型serving

#### faiss库

#### ANN检索

### 在线serving

#### 流程

