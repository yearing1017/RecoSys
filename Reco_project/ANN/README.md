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
以经典的ctr目标，作为唯一优化目标，样本的label只有一个

- 多目标
以短视频场景为例，可分为时长、点赞、关注等多个目标，样本的label有多个

#### 特征
- 特征处理方式
    - embedding类特征
      - hash取模
        - 对id进行hash取模，ori_id转为hash_id
        - 一个group中的为同一类型的embedding矩阵；举例：视频id为一个group，标签id为一个group
    - 数值型特征
      - 归一化
      - 分桶处理
    - 类别型特征
- user侧
    - embedding类特征
    - 数值型特征
    - 类别型特征
- item侧
    - embedding类特征
    - 数值型特征
    - 类别型特征

#### 采样
- batch内负采样
    - 原理
        - 以一个batch内数据作为候选池，对于其中的每条数据，在剩余【batch_size-1】条数据中，选择N个item_id，进行负样本的拼接 
    - 流程：
        - 输入数据中的样本1：label_1(表示为正样本) + user_id + item_id + user_fea + item_fea
        - 负采样随机采到1个item_id_neg,构造负样本：label_0(表示负样本) + user_id + item_id_neg + user_fea + item_neg_fea，用户侧特征不变，拼接相应item特征
    - 优点
    - 缺点
    
- 随机mask负采样
    - 原理
        - 发现batch内负采样会对热门item打压过重，根据item的展现量，进行随机mask，使之不参与训练，减轻热门item作为负样本的概率
    - 优点
    - 缺点
- 全局负采样
    - 原理
        - 构建全局item候选池，根据item的展现量，使热门item作为负样本的概率增大
    - 流程：
        - 输入数据中的样本1：label_1(表示为正样本) + user_id + item_id + user_fea + item_fea
        - 在全局采样池中负采样，随机采到1个item_id_neg，构造负样本：label_0(表示负样本) + user_id + item_id_neg + user_fea + item_neg_fea，用户侧特征不变，拼接相应item特征
    - 优点
    - 缺点

#### 样本权重
- 正样本加权
- x目标加权
- 多目标grad

### 模型训练

#### 模型
- 单目标
模型为标准的双塔DSSM结构：user塔和item塔，两塔的输出经过【内积+sigmoid】之后，计算loss

- 多目标
在user塔或item塔的输出层，分出多个head，每个head与另外的塔进行交互，计算loss


- loss
    - cosine + fc + sigmoid
    - cosine + 温度系数 + sigmoid
        - 温度系数对loss的影响
- 模型结构
    - user & item all_one_head
    - share_bottom_only
    - share_bottom & share_dnn
    - user_multi_head & item_one_head
    - user_one_head & item_mutli_head

- 结构优化
    - senet结构

#### 离线训练
- 天级训练
- 小时训练
    - 小时级
    - n小时级
- 天级 + 小时级

#### 实时训练
- FLINK任务
    - 实时采样流程

### 离线评估

#### recall@N
- 计算方法：对每个用户来说：topk召回结果【命中】 用户最近正行为item个数 / 用户最近正行为item个数，对一批用户再求均值

#### loss & gauc
- log_loss
- ne
- auc
- gauc
- calibration

#### 人工评估case
- 是否命中主要兴趣点
- 多样性的考虑

### 索引构建
- 整体流程
  - 模型实时训练，每小时保存更新
  - 拉取最新模型，离线生成索引，连同模型一起打包到线上数据平台
    - 候选item的选取（发布天数、效率、展现量）
    - item特征拼接，生成预估样本
    - 预估item_embedding，生成faiss库


#### 保证同一模型serving
- 目的：线上实时生成user-embedding，离线生成item-embedding索引库，需保证为同一模型
- 做法：离线生成索引库时，将模型和索引库一起打包传输，供线上serving使用

#### faiss库
- 构建方式
    - item特征拼接，生成预估样本
    - 预估item_embedding，生成faiss库
    - 索引库和模型打包，供线上serving使用，保证同一模型
- 更新频率
    - 小时级
    - n小时级
    - 天级

#### ANN检索
- 检索算法
    - IndexFlatIP
    - ivf
    - HNSW


### 在线serving
- 整体流程

- 召回结果的选取
    - 蛇形merge
    - 直接截断

