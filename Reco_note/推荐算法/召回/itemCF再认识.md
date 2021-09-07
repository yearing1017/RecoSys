#### ItemCF原理

ItemCF是基于Item的协同过滤（Collaboration Filter）算法，它是通过分析用户的行为来计算Item的相似度。与基于内容计算相似、一些embedding方法相比，itemcf中增加了用户的行为，在线上效果表现会比较好。

该算法认为物品A和物品B相似的依据是因为喜欢物品A的用户也喜欢物品B。

##### 共现矩阵

- 基于的原始数据：user对item的行为序列数据 (user, item1, item2, item3 ...)

- 针对所有的item建立共现矩阵，**矩阵中的元素$C[i][j]$是同时喜欢item_i 和 item_j 的用户数**

- 计算物品 a 与物品 b 的相似度:
  $$
  W_{a b}=\frac{C[a][b]}{\sqrt{N(a) * N(b)}}=\frac{1}{\sqrt{6}} \approx 0.41
  $$

  - 分子表示的是物品 a 与物品 b 的共现次数
  - 分母中 N(a) 表示的是对喜欢物品 a 的人数，N(b) 表示喜欢物品 b 的人数

##### 基于用户对item的打分进行协同

- 给定数据：user对item的打分数据  (user, (item,  score))
- 计算 (item, score^2_sum)  每个item对应的score的平方和
- 得到每个user的两两共现的item的信息 
  - (user, ((item1, score1, item1_counts), (item1, score1, item1_counts)))  
  - RDD[(String, ((Int, Double, Int), (Int, Double, Int)))]
- 计算两两item的得分乘积和 和 共现次数((item1, item2), （sum(score1 * score2), 共现次数))
- 相似度的计算  两个item分数乘积的平方和 / 根号下 item1的score平方和 * item2的score平方和

```scala
/**
   * Item CF ：基于item协同
   * @param userRatings   : (user,(item, score))
   * @param similarity    : 计算相似度的方法 loglikelihood 或  cosine
   * @param minFreq       : 共现次数阈值
   * @param topSimItems   : 相似商品截断阈值
   * @param numPartitions : join时的分片数
   * @return
   */
  def calculateItemSim(userRatings: RDD[(String, (Int, Double))],
                       similarity: String = "loglikelihood",
                       minFreq: Int = 2,
                       topSimItems: Int = 100,
                       numPartitions: Int = 5000): RDD[(Int, Array[(Int, Double)])] = {
    val sc = userRatings.sparkContext
    // 取出top100活跃用户的打分
    val topFreqUser = userRatings.map(l => (l._1, 1)).reduceByKey(_ + _).top(100)(Ordering[Int].on(_._2))
    val topFreqUserSet = sc.broadcast(topFreqUser.map(_._1).toSet)
    // 去除掉最常用用户top100打分
    var filteredUserRatings = userRatings.filter(l => !topFreqUserSet.value.contains(l._1))
    // 每种item的个数统计 
    val itemCounts = filteredUserRatings.map(l => (l._2._1.toInt, 1)).reduceByKey(_ + _)
    val itemCountsMap = sc.broadcast(itemCounts.collectAsMap())
    //(uid,(gid,rating,gidCount))
    val filteredUserRatingsAndCount = filteredUserRatings.mapValues(l => (l._1, l._2, itemCountsMap.value(l._1)))

    // (uid,(gid,rating,gidCount),(gid,rating,gidCount)) 
    // 所有用户Rating数据只保留第一个gid比第二个gid小的情况（相当于只取矩阵的下三角或上三角）
    val joinResult = filteredUserRatingsAndCount.join(filteredUserRatingsAndCount, numPartitions = numPartitions)
      .filter {
        case (_, ((item1, _, _), (item2, _, _))) =>
          item1 < item2
      }

    // loglikelihood or cosine  
    // RDD[(Int, (Int, Double))]
    val itemRawSims = similarity match {
      case "loglikelihood" =>
        val all_event_num = sc.broadcast(filteredUserRatings.count())
        joinResult.map {
          case (_, ((item1, _, count1), (item2, _, count2))) =>
            (((item1, count1), (item2, count2)), 1)
        }  // 计算所有用户Rating数据中 item1 和 item2的共现次数
          .reduceByKey(_ + _)
          .filter(_._2 >= minFreq)
          .flatMap {
            // 根据两个item的共现次数 计算相似度
            case (((item1, count1), (item2, count2)), freq) =>
              // loglikelihood ratio
              val sim = LogLikelihood.logLike(count1, count2, freq, all_event_num.value)
              List((item1, (item2, sim)), (item2, (item1, sim)))
          }
      case "cosine" =>
        // (item, score^2_sum) 每个item对应的score的平方和
        val itemRatingProduct = filteredUserRatings
          .map(l => (l._2._1, l._2._2 * l._2._2)).reduceByKey(_ + _).collectAsMap()
        val itemRatingProductBC = sc.broadcast(itemRatingProduct)
        // joinResult: RDD[(String, ((Int, Double, Int), (Int, Double, Int)))]
        joinResult.map {
          case (_, ((item1, rating1, _), (item2, rating2, _))) =>
            ((item1, item2), (rating1 * rating2, 1))
        }
          // ((item1, item2), （sum(score1 * score2), 共现次数))
          .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
          .filter(_._2._2 >= minFreq)
          .flatMap {
            case ((item1, item2), (dotProduct, freq)) =>
              val powSumRating1 = itemRatingProductBC.value(item1)
              val powSumRating2 = itemRatingProductBC.value(item2)
              // 相似度的计算  两个item分数乘积的平方和 / 根号下 item1的score平方和 * item2的score平方和
              val sim = dotProduct / math.sqrt(powSumRating1 * powSumRating2)
              List((item1, (item2, sim)), (item2, (item1, sim)))
          }
    }
    //  RDD[(Int, (Int, Double))]
    itemRawSims
      .topByKey(topSimItems)(Ordering[Double].on(_._2))
      .mapValues(arr => {
        // 归一化
        val maxSim = arr.map(_._2).max + 1e-7
        arr.map(x => (x._1, x._2 / maxSim))
      })
  }
```

#### ItemCF优化

##### IUF参数惩罚

在协同过滤中两个物品产生相似度是因为它们共同出现在很多用户的兴趣列表中。换句话说，**每个用户的兴趣列表都对物品的相似度产生贡献。那么是不是每个用户的贡献都相同呢?**

John S. Breese在论文【Empirical Analysis of Predictive Algorithms for Collaborative Filtering】中提出了**IUF(Inverse User Frequence)，即用户活跃度对数的倒数的参数**，他认为**活跃用户对物品相似度的贡献应该小于不活跃的用户**，他提出应该增加IUF 参数来修正物品相似度的计算公式（这个思想和TF-IDF中的IDF思想是一致的）：
$$
W_{a b}=\frac{\sum_{u \in N(a) \cap N(b)} \frac{1}{\log (1+N(u))}}{\sqrt{N(a) * N(b)}}
$$

##### 相似度归一化

George Karypis 在论文【Evaluation of Item-based Top-N Recommendation Algorithms】中提到如果**将ItemCF的相似度矩阵按最大值归一化，可以提高推荐的准确率**，注意这里的 $max_j$ 表达的是其相似item j 最大值的下标：
$$
w_{i j}^{\prime}=\frac{w_{i j}}{\max _{j} w_{i j}}
$$
归一化的好处不仅仅在于增加推荐的准确度，它还可以提高推荐的覆盖率和多样性。 

一般来说，物品总是属于很多不同的类，每一类中的物品联系比较紧密。**假设物品分为两类**——A和B，A类物品之间的相似度为0.5，B类物品之间的相似度为0.6，而A类物品和B类物品之间的相似度是0.2。在这种情况下， 如果一个用户喜欢了5个A类物品和5个B类物品，用ItemCF给他进行推荐，推荐的就都是B类物品， 因为B类物品之间的相似度大。但如果归一化之后，A类物品之间的相似度变成了1，B类物品之 间的相似度也是1，那么这种情况下，用户如果喜欢5个A类物品和5个B类物品，那么他的推荐列 表中A类物品和B类物品的数目也应该是大致相等的。从这个例子可以看出，相似度的归一化可 以提高推荐的多样性。

##### 基于时间权重的优化

用户对Item产生行为的时间是不一样的，通常情况下使用itemcf并没有对不同时间行为进行时间衰减。因此一种常见的思路就是：**在计算item 相似度时进行时间衰减；也可理解为距离衰减，在行为序列中，距离越远的两个item相似性应该低一些，例如ABCDE，A和B的相似度 > A和E的相似度**
$$
W_{a b}=\frac{\sum_{u \in N(a) \cap N(b)} f\left(\left|t_{u a}-t_{u b}\right|\right)}{\sqrt{N(a) * N(b)}}
$$
其中 **f 为衰减权重**，下式中t_ua代表用户u对a的行为时间
$$
f\left(\left|t_{u a}-t_{u b}\right|\right)=\frac{1}{1+\alpha\left|t_{u a}-t_{u b}\right|}
$$

##### 基于session的优化

基于session的优化主要**考虑的是用户行为的连贯性，因为用户可能在不同的时间段内有不同的偏好行为**，比如用户在 T 时间段搜索的是书籍相关的东西，而在 T+1 时间段可能考虑的是电子数码相关的物品

session可以根据具体的业务场景进行不同的定义，常见的定义方式有以下几种：

- 用户一次会话称为一次session，即用户的一次会话过程
- 按照固定的时间段去划分，比如按天划分，用户一天内的行为是一个session
- 按照间断时间去划分，比如用户的一次会话中，如果两次行为的时间超过30分钟，可以划分为两个session
- 按照行为划分，比如用户构建用户点击+下单序列，按照时间排序后， 可以根据下单进行截断

session在一些graph、dnn、序列算法中应用的比较多，比如：

- 2018年阿里发表的论文：Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba中在**构建用户的行为序列时，就是以两次行为时间超过1小时进行截断**
- **论文里对做cf的启发点还包括：**
  - 点击之后用户停留时间小于1s，这可能是用户的误点击，需要过滤
  - 太过活跃的用户进行过滤，比如三个月内购买了1000件以上的商品，点击了3500个以上的商品（这个要根据实际情况去定）
  - 同一个ID，但是发生变化的商品需要过滤
- 2018年Airbnb发表的论文：Real-time Personalization using Embeddings for Search Ranking at Airbnb中**将用户的行为分为了两类**：
  - 经过一系列点击之后有预定房源的行为
  - 经过一系列点击之后没有预定房源的行为
  - 用户短期内预定房源的行为其实在一定程度上和之前的搜索点击行为都有强相关的关系，因此将预定房源的行为作为一个全局的动作加入到每个session中，这一点对于我们的启发：**用户行为中包含的偏好信息会随着某一强偏好行为而终结或者发生变化**

##### 对于Match的优化

match的优化主要集中在user-to-item，主要有两种方式

- **行为选择**
  - 比如基于用户过去一周内的行为进行item的相似召回，但是用户对item的行为包括：曝光、点击、提单、下单、分享、收藏等，**针对不同的行为其偏好强度是不一样的**，这里假设用户的行为强度大小关系为：下单 > 提单 > 分享 > 收藏 > 点击 > 曝光
  - 那么可以选取用户过去一周内的除曝光外的行为，**按照时间排序，同样spu去除权重低的行为**。比如用户分享、收藏、提单、下单的spu，肯定是发生过点击，可以从点击行为中进行删除
  - **不同行为赋予不同的权重，然后在进行行为相似召回时，进行相应的加权**

- **时间衰减**
  - 用户行为越接近于当前，其权重越大，在使用时对用户过去一周内行为的商品进行时间的降权。

