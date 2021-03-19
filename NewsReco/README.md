#### 题目

- 新闻推荐场景下的用户行为预测挑战赛， 该赛题是以新闻APP中的新闻推荐为背景， 目的是**要求我们根据用户历史浏览点击新闻文章的数据信息预测用户未来的点击行为， 即用户的最后一次点击的新闻文章**

#### 数据概况

- 该数据来自某新闻APP平台的用户交互数据，包括30万用户，近300万次点击，共36万多篇不同的新闻文章，同时**每篇新闻文章有对应的embedding向量表示**。
- 从中抽取**20万用户的点击日志数据作为训练集，5万用户的点击日志数据作为测试集A，5万用户的点击日志数据作为测试集B**。具体数据表和参数，可以参考赛题说明。

#### 提交结果的评价方法

- 结合着最后的提交文件来看， 根据sample.submit.csv， 我们最后提交的格式是**针对每个用户， 我们都会给出五篇文章的推荐结果，按照点击概率从前往后排序。 而真实的每个用户最后一次点击的文章只会有一篇的真实答案， 所以我们就看我们推荐的这五篇里面是否有命中真实答案的**。比如对于user1来说， 我们的提交会是：

  > user1, article1, article2, article3, article4, article5.

  评价指标的公式如下：
  $$
  \text { score }(u s e r)=\sum_{k=1}^{5} \frac{s(u s e r, k)}{k}
  $$

- 假如article1就是真实的用户点击文章，也就是article1命中， 则s(user1,1)=1, s(user1,2-4)都是0， 如果article2是用户点击的文章， 则s(user,2)=1/2,s(user,1,3,4,5)都是0。也就是**score(user)=命中第几条的倒数。如果都没中， 则score(user1)=0。 我们希望的就是命中的结果尽量靠前， 而此时分数正好比较高。**

#### 赛题详细理解

- 目标： **根据用户历史浏览点击新闻的数据信息预测用户最后一次点击的新闻文章**。从这个目标上看， 会发现此次和之前遇到的普通的结构化比赛不太一样， 主要有两点：
  - 首先是目标上， **要预测最后一次点击的新闻文章，也就是我们给用户推荐的是新闻文章， 并不是像之前那种预测一个数或者预测数据哪一类那样的问题**
  - 数据上， 通过给出的数据我们会发现， **这种数据也不是我们之前遇到的那种特征+标签的数据，而是基于了真实的业务场景， 拿到的用户的点击日志**

- 思考方向就是结合我们的目标，**把该预测问题转成一个监督学习的问题(特征+标签)，然后我们才能进行ML，DL等建模预测**。那么我们自然而然的就应该在心里会有这么几个问题：
  - 如何转成一个监督学习问题呢？ 
  - 转成一个什么样的监督学习问题呢？
  -  我们能利用的特征又有哪些呢？ 
  - 又有哪些模型可以尝试呢？ 
  - 此次面对数万级别的文章推荐，我们又有哪些策略呢？

#### data线上线下

- 线下模式：
  - **将train+test的点击记录合并，记为click**。
  - **随机抽取5万用户from train的最后一条点击记录作为线下验证集，并将该5万条记录从原来的train中删掉，copy一份test的用户id，将所有的用户的点击记录设置为-1，此时，将线下验证集+test_copy合并，作为query**
  - 在召回的时候，query中的所有用户都存在与click中，**对query的用户都进行召回**；不同的是：**对5w的线下验证用户进行label的设置，正负标签的设置**，然后计算mrr，hitrate等线下指标。**对test_copy的用户，不设置label，设置为空**
  - 在排序模型的训练时，**此时的训练集为召回的数据中label有值的数据，即5万条数据；测试集为之前label为空的数据；**训练好模型时，**对test的召回结果进行排序，最后取前5条**作为结果返回

- 线上模式：
  - 取消了验证集的设置，**click与线下的click一致，但query只有test_copy了**
  - **召回的时候还是和离线时一样召回，排序的时候使用线下训练的模型进行排序**

#### 改进版的itemCF

- 原始的 itemcf 将用户点击过的新闻看做一个无序的集合，但在实际应用中，**应该考虑点击次序带来的影响**。在**计算同一序列中两个新闻的相似度时，不仅需要考虑共现次数，也需要考虑两个新闻之间的次序关系。同一点击序列中两个新闻位置越远，相关性应该减小。新闻对顺序和逆序的权重也不同，在点击序列A，B，C中，"BC"这样的正序权重应该大于"BA"这样的逆序权重**。更多关于 itemcf 的改进可以看之前 KDD CUP 的相关方案总结。
- 建立新闻的相似度关系后，进入到**召回阶段**，根据用户的历史点击新闻，结合相似度选择 TOP100 关联新闻。**选取关联新闻时，除了考虑和历史点击新闻的相似度，还要加入位置距离衰减。新闻点击是强热点相关，所以历史点击新闻对下一次点击预测的影响传播不会太远。在实际测试中，利用所有历史点击新闻做召回，hitrate_5 指标只有0.20，限定只用最近点击的两个新闻来做召回的话，可以大幅提升至0.33。**

- 考虑顺序权重和共现次数一起建立该共现矩阵：
  - 考虑：**可以加入时间戳作为正向点击和反向点击的依据**

```python
# 依次对每个用户点击的新闻序列两两+1，得到所有用户点击过的所有新闻的共现矩阵 值为同时喜欢两个新闻的用户数
    for _, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_dict.setdefault(item, {})
            
            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue

                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重
                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 离得越近且为正向点击的话 权重就越大
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))

                sim_dict[item][relate_item] += loc_weight  / \
                    math.log(1 + len(items))
    # 根据上面得到的共现数 算相似矩阵 C矩阵归一化可以得到物品之间的余弦相似度矩阵W。
    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / \
                math.sqrt(item_cnt[item] * item_cnt[relate_item])

    return sim_dict, user_item_dict

```

- 召回阶段：**依据每个用户历史点击数据中最后两个点击新闻做召回，在相似矩阵中找到与其相似的新闻，并依据距离最后一次点击的距离做相似评分，召回100**
- 召回完之后，构建正负样本：召回的文章 若 和 之前读过的文章（此处的df_query中的用户的读过文章是之前在训练集中随机选的5w用户的最后的点击数据）一样， 则为正样本

```python
for user_id, item_id in tqdm(df_query.values):
        rank = {}
        # 此句意思是给验证集的用户召回 
        if user_id not in user_item_dict:
            continue
        # 针对每个用户 取出相应的历史点击新闻序列
        interacted_items = user_item_dict[user_id]
        # 倒序排列用户的历史点击新闻 取出最后点击的两个
        interacted_items = interacted_items[::-1][:2]
        #此处的loc代表的使用户最后点击的序列 loc越大 代表离最后一次越远
        for loc, item in enumerate(interacted_items):
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij * (0.7**loc)

        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]
        # 将验证集中每个用户的召回结果保存下来
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        # 构建 正负样本
        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            # 召回的文章 若 和 之前读过的文章一样 则为正样本
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')
```

#### word2vec召回

- 数据的生成：通过将每个用户的历史点击序列变成字符串 进行训练

- 通过gensim去训练模型，训练完模型后，通过模型去生成每条新闻id的embedding向量

```python
# df_click, 'user_id', 'click_article_id'
def word2vec(df_, f1, f2, model_path):
    df = df_.copy()
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})
    # sentenses 是将每个用户的点击历史点击新闻id按照从前往后的顺序存到一个list中  所以的用户形成一个总的list
    # 以下是5个用户的
    # [[300470, 16129, 160974], [160974], [183665, 181686], [160974, 202557], [160974, 160417]]
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    words = []
    #将每个用户的历史列表转为字符串
    for i in range(len(sentences)):
        #  只有转换成字符串才可以进行训练
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        model = Word2Vec(sentences=sentences,
                         size=256, # size: 表示词向量的维度。
                         window=3, #window：决定了目标词会与多远距离的上下文产生关系。
                         min_count=1, # min_count: 设置最小的
                         sg=1, #sg: 如果是0，则是CBOW模型，是1则是Skip-Gram模型。
                         hs=0,
                         seed=seed,
                         negative=5,
                         workers=10, # workers: 表示训练时候的线程数量
                         iter=1) # iter: 训练时遍历整个数据集的次数
        model.save(f'{model_path}/w2v.m')

    article_vec_map = {}
    for word in set(words):
        if word in model:
            # key是新闻id，value是训练好的相应的新闻id的词向量 存起来
            article_vec_map[int(word)] = model[word]

    return article_vec_map
```

- embedding词向量表生成之后，通过annoy.AnnoIndex去进行快速最近邻检索
- 首先建立item_id和词向量的对应关系

```python
# 得到每个文章id对应的词向量
    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',model_path)
    f = open(w2v_file, 'wb')
    # 将得到的词向量 保存至文件中
    pickle.dump(article_vec_map, f)
    f.close()
    # 说白了就是先将加载进来的vector进行相似临近计算，然后生成一个树形结构的索引，这样查找速度会变得很快，只不过会牺牲一定的近似精度。
    # 将 embedding 建立索引
    article_index = AnnoyIndex(256, 'angular') #metric='angular'表示使用 angular（余弦）距离度量来计算簇和哈希。
    article_index.set_seed(2020)
    ##加载article_id和向量映射，添加到annoyIndex
    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)
    # tree_num设置为100，在内存允许的情况下，越大越好
    article_index.build(100)
```

- 在召回的时候，通过AnnoIndex去快速召回要查询向量的最近邻100，然后通过返回的100个最近邻的向量和要查询向量的distances来构建相似度

```python
# 给验证集的用户召回
    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)
        # 验证集的用户曾经点过的新闻
        interacted_items = user_item_dict[user_id]
        
        interacted_items = interacted_items[-1:]

        for item in interacted_items:
            article_vec = article_vec_map[item]
#If you set include_distances to True, it will return a 2 element tuple with two lists in it: 
#the second one containing all corresponding distances.
            # anno通过快速最近邻检索  通过余弦距离找到和当前向量最接近的100个相应的id
            # distances包含了这100个检索的 和 查询的向量 的余弦距离
            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 100, include_distances=True)
            # 通过该距离计算相似度
            sim_scores = [2 - distance for distance in distances]

            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij
```

#### 验证集的离线指标MRR和HitRate

![image-20210314225538365](/Users/yearing1017/Library/Application Support/typora-user-images/image-20210314225538365.png)

- 由此可知：**MRR的计算：找到验证集中每个用户的推荐结果里，和真实最后点击新闻一致的排名，取倒数，将所有用户的相加，除以用户数，越大越好**

![image-20210314225937269](/Users/yearing1017/Library/Application Support/typora-user-images/image-20210314225937269.png)

- 有公式可得：**hitrate_N是在前N个rank里命中click元素的用户数 / 总用户数**

```python
def evaluate(df, total):
    hitrate_5 = 0
    mrr_5 = 0

    hitrate_10 = 0
    mrr_10 = 0

    hitrate_20 = 0
    mrr_20 = 0

    hitrate_40 = 0
    mrr_40 = 0

    hitrate_50 = 0
    mrr_50 = 0

    gg = df.groupby(['user_id'])
    # g是每个用户的所有数据 很多行
    for _, g in tqdm(gg):
        try:
            item_id = g[g['label'] == 1]['article_id'].values[0]
        except Exception as e:
            continue
        # 用户召回的结果
        predictions = g['article_id'].values.tolist()

        rank = 0
        # 找到召回结果中和真实的最后一次点击一样的排名
        while predictions[rank] != item_id:
            rank += 1
        # 前5个击中 目标的用户数
        if rank < 5:
            mrr_5 += 1.0 / (rank + 1)
            hitrate_5 += 1

        if rank < 10:
            mrr_10 += 1.0 / (rank + 1)
            hitrate_10 += 1

        if rank < 20:
            mrr_20 += 1.0 / (rank + 1)
            hitrate_20 += 1

        if rank < 40:
            mrr_40 += 1.0 / (rank + 1)
            hitrate_40 += 1

        if rank < 50:
            mrr_50 += 1.0 / (rank + 1)
            hitrate_50 += 1

    hitrate_5 /= total
    mrr_5 /= total

    hitrate_10 /= total
    mrr_10 /= total

    hitrate_20 /= total
    mrr_20 /= total

    hitrate_40 /= total
    mrr_40 /= total

    hitrate_50 /= total
    mrr_50 /= total

    return hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50
```

#### 召回合并

- 读取多路召回的结果，因为sim_score是排序的标准，所以需要进行得分的合并；
- 由于各路召回得分计算方式的不同，所以需要先进行归一化，这里针对每个具体的召回方法，针对每个用户，对其多个召回结果的score进行**最大值最小值归一化**

```python
def mms(df):
    user_score_max = {}
    user_score_min = {}

    # 获取用户下的相似度的最大值和最小值
    for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
        scores = g['sim_score'].values.tolist()
        # 之前召回时已按相似度大小排好序  
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    # 每个用户召回了n个文章，对每个用户的n个召回结果的相似度进行最大最小值归一化 下面是公式
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans
```

- 对召回方法不同的得分，进行加权求和

```python
for recall_method in recall_methods:
        recall_result = pd.read_pickle(
            f'{recall_path}/recall_{recall_method}.pkl')
        weight = weights[recall_method]
        # 重复被召回的新闻在多个召回方式中的得分是不一样的，得分是排序时的强特，需要进行得分合并。
        # 对各个召回结果，以用户为单位进行相似度的最大最小归一化。
        recall_result['sim_score'] = mms(recall_result)
        # 召回方法对应其权重
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result

    # 求相似度
    for recall_method1, recall_method2 in permutations(recall_methods, 2):
        score = recall_result_sim(recall_dict[recall_method1],
                                  recall_dict[recall_method2])
        log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score}')

    # 合并召回结果
    recall_final = pd.concat(recall_list, sort=False)
    # 分数合并测试了sum，mean和max，效果对比见下表。max丢失的消息较多，mean对重复次数多的新闻不公平。
    recall_score = recall_final[['user_id', 'article_id',
                                 'sim_score']].groupby([
                                     'user_id', 'article_id'
                                 ])['sim_score'].sum().reset_index()
```

- 将多路召回结果合并之后去重

```python
# 去重 用户id和新闻id都相同的召回结果删除掉
    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])
```

- 计算合并后的指标

```python
# 删除没有召回到真实点击的验证集用户，减少了无用负样本的数量，也提高了后续排序模型的效果，
    # 但是在计算线下指标的时候这部分用户的数量要包括进去，否则指标会虚高。
    # 删除无正样本的训练集用户
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            label_sum = g['label'].sum()
            if label_sum > 1:
                print('error', user_id)
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)
    log.debug(f'df_useful_recall: {df_useful_recall.head()}')

    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'sim_score'], ascending=[True,
                                             False]).reset_index(drop=True)

    # 计算相关指标
    if mode == 'valid':
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_useful_recall[df_useful_recall['label'].notnull()], total)
        
        log.debug(
            f'召回合并后指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
```

#### 特征工程

- 这部分的作用：**在之前合并的召回结果的特征基础上，添加一些自己构建的特征，以利于排序模型的训练**

- 特征工程分为3类：**新闻特征，用户特征，用户-新闻交互特征**。数据集本身给出的属性信息较少，所以特征工程主要围绕交互属性展开。
- 新闻特征包括：
  - 新闻字数
  - 新闻创建时间
  - 新闻被阅读数量

- 用户特征包括：
  - 用户点击新闻的创建时间差的平均值
  - 用户点击新闻的点击时间差的平均值
  - 用户点击新闻的点击-创建时间差的统计值：mean，std
  - 用户点击新闻的 click_datetime_hour 统计值
  - 用户点击新闻的字数统计值
  - 用户点击新闻的创建时间统计值
  - 用户点击新闻的点击时间统计值
  - 用户新闻阅读数量
  - 用户某种类新闻阅读数量

- **交互特征主要基于之前的召回策略进行，通过保存召回阶段的新闻相似度信息或向量，我们能够间接或直接得到用户对待预测新闻的评分。**
- **基于 itemcf， 网络关系和 w2v 的召回得到的只是新闻之间的相似度，需要和用户的历史点击新闻计算间接得到用户-新闻评分，采用如下方式**：
  - 待预测新闻和用户所有历史点击新闻相似度按次序加权求和
  - 待预测新闻和用户最近一次点击新闻相似度

- 在计算上述用户-新闻评分时，因为itemcf和w2v的评分计算方式不一样，所以两种方法的计算方法不同：

```python
# itemCF  待预测新闻和用户所有历史点击新闻相似度按次序加权求和
def func_if_sum(x):
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += item_sim[i][article_id] * (0.7**loc)
        except Exception as e:
            pass
    return sim_sum

# itemCF 待预测新闻和用户最近一次点击新闻相似度
def func_if_last(x):
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = item_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim

# 两个向量之间的余弦距离
def consine_distance(vector1, vector2):
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray:
        return -1
    distance = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return distance

# w2v 待预测新闻和用户倒数num条历史点击新闻相似度按次序加权求和
def func_w2w_sum(x, num):
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1][:num]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += consine_distance(article_vec_map[article_id],
                                        article_vec_map[i])
        except Exception as e:
            pass
    return sim_sum

# w2v 待预测新闻和用户最近一次点击新闻相似度
def func_w2w_last_sim(x):
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = consine_distance(article_vec_map[article_id],
                               article_vec_map[last_item])
    except Exception as e:
        pass
    return sim
```

#### 排序模型的训练

- 先划分数据：
  - 训练的数据是**经过召回+特征工程后 label不为空的数据，进行二分类的训练**
  - 测试的数据是**经过召回+特征工程后 label为空的数据，利用模型进行召回结果的排序**

```python
# 现在的训练样本:最初的5w验证集+test  ->  经过召回+特征工程之后 -> label不为空的
    df_train = df_feature[df_feature['label'].notnull()]
    # 现在的测试样本就是之前的test，label设为null的行，但是这些用户也召回新闻了
    df_test = df_feature[df_feature['label'].isnull()]
```

- lgb分类模型的搭建
  - LGBM、XGBoost都是基于GBDT的，这三者可以说是同属于一类，XGBoost先作为GBDT的工程实现，后续LGBM又在此基础上进行优化，都属于GBDT原理的工程实现库
- k折交叉验证的训练
- 线下验证集的指标计算
  - 题目里给的指标其实就是MRR@5
- 测试集的召回结果排序，返回前5条