### 前言

- 本篇论文笔记是2017年的[Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)，由新加坡国立大学何向南教授完成的一篇关于基于神经网络方法的推荐系统论文；
- 参考：《深度学习推荐系统-王喆》

### 论文主要内容

- 这篇文章使用神经网络（NN, Neural Network）方法完成了推荐系统中的协同过滤，不同于其他算法中使用神经网络提取辅助特征，user和item仍然使用矩阵内积来计算；这篇文章使用神经网络直接计算user和item之间的关系
  - 通过神经网络代替矩阵分解算法（形成模型**GMF**）来提取user和item之间的**线性关系**
  - 搭建3层的**MLP**来提取user和item之间的**非线性关系**
  - 组合MLP和GMF形成最终的**NeuMF**模型。

### Introduction

- 推荐系统中主流方法是协同过滤CF（Collaborative Filtering）方法，协同过滤中最流行的就是矩阵分解MF（Matrix Factorization）算法。但是矩阵分解算法由于在隐空间中只是将user和item的矩阵进行简单内积运算，由此导致了很大程度的信息丢失。
- 对于矩阵$U \in R^{m \times k}$ 和矩阵$V \in R^{k \times n}$，矩阵相乘，在结果矩阵$W \in R^{m \times n}$，每个位置得到的结果$W_{ij}$相当于左矩阵U的每一行$u_i^{T}$与右矩阵V的每一列$v_j$内积运算的结果：$W_{ij} = <u_i, v_j> = u_i^T \times v_j$ 。由此进行拓展，对于矩阵分解算法，每一个$W_{ij}$其实就是和对应位置线性相乘之和

- 所以作者提出将简单的矩阵相乘改由深层神经网络模型解决的思路，并指出以往的深度学习方法大多是在辅助信息（Auxiliary Information）上整活儿，但是这篇文章是直接将向量内积的部分改掉。

### Preliminaries

#### **从隐式数据中学习**

- 令M和N分别表示用户数和物品数。定义用户-物品交互矩阵Y，从用户的隐式反馈来看：

$$
y_{u i}=\left\{\begin{array}{ll}1, & \text { if interaction (user } u, \text { item } i) \text { is observed; } \\ 0, & \text { otherwise }\end{array}\right.
$$

- 这里，  $y_{ui}$的值为1，表示用户u与物品i之间存在交互; 然而，这并不意味着用户u真的喜欢物品i。类似地，$y_{ui}$的值为0并不一定意味着用户u不喜欢物品i，因为也有可能是用户不知道该物品。这对从隐式数据中进行学习提出了挑战，因为它只提供关于用户偏好的噪声信号。虽然观察到的交互数据反映了用户对物品的兴趣，但未观察到的交互数据可能只是缺少数据，而且自然缺乏负面反馈。
- 隐式反馈的推荐问题被表述为估计用户-物品交互矩阵Y中未观察物品的分数的问题，该分数用于对物品进行排序。基于模型的方法假设可以生成数据 (或由底层模型描述)。在形式上,他们可以抽象为 

$$
\hat{y}_{u i}=f(u, i \mid \Theta)
$$



-  $\hat{y}_{u i}$表示用户u和物品i的预测得分，Θ表示模型的参数，f表示将模型参数映射到预测分数的函数(我们称之为交互函数)。为了估计参数 Θ ，现有的方法通常遵循优化目标函数的机器学习范式。文献中最常用的目标函数有两种——pointwise loss和pairwise loss。

- 给定${y}_{u i}$，point-wise loss致力于缩小每一个${y}_{u i}$和$\hat{y}_{u i}$之间的距离，这样做的前提是认为用户不喜欢每个${y}_{u i} = 0$的item，但这是不对的（和我们日常经验保持一致，我们去超市买东西，买到的东西认为我们喜欢这没问题，但是没买到的东西都是不喜欢的这就问题太大了，因为更常见的情况是我们没买只是因为我们不知道这个东西的存在）;pair-wise loss则假定一个排序，认为和user有互动的item比没有互动的item优先级更高，我们应该拉大两者之间的距离。

#### Matrix Factorization

- 这里作者举了一个例子来证明矩阵分解算法的局限性。作者首先将矩阵分解是线性运算进行了解释，也就是我们在Introduction中说明的内容，公式化如下： 这里的大写K表示隐空间维度，p和q分别代表用户和物品的隐向量

$$
\hat{y}_{u i}=f\left(u, i \mid \mathbf{p}_{u}, \mathbf{q}_{i}\right)=\mathbf{p}_{u}^{T} \mathbf{q}_{i}=\sum_{k=1}^{K} p_{u k} q_{i k}
$$

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/9-1.jpg)

- 这里作者用上图来说明即使实际中看起来更相似（Jaccard系数更近）的两个向量，在坐标系中的内积并不一定更小（单位向量的夹角不一定更小）。
- 这个例子说明了在低维隐空间中，使用一个简单而固定的内积来估计复杂的用户-项目交互可能会导致MF的局限性。我们注意到，解决这一问题的一个办法是利用大量的潜在因素。然而，这可能会对模型的泛化造成不利影响(例如，数据的过度拟合)，特别是在稀疏环境中。在本文的工作中，我们通过使用深度神经网络从数据中进行学习构造相互作用函数来解决这个问题。

### NeuralCF

#### 整体框架

- 下图展示了本文的通用神经网络框架。 
  - Input Layer：最下层的两个输入分别表示user和item，分别使用user和item的编号的one-hot编码格式作为输入向量。  
  - Embedding Layer：全连接层，将one-hot编码转换为一个较短的向量形式。 
  -  Neural CF Layers：神经协同过滤层，包含多层，本文使用三层的格式，每层神经元个数减半，全连接前馈神经网络形式。 
  - Output Layer：最终输出只有一个值，离1越近表示user可能越喜欢这个item。与进行比较后，将损失进行反向传播，优化整个模型

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/9-2.jpg)

- 为了允许对协同过滤进行完整的神经处理，我们采用多层表示对用户-物品交互$y_{ui}$进行建模，如图2所示，其中一层的输出作为下一层的输入。底层输入层由两个特征向量 $u_u^U$ 和 $v_i^V$ 组成分别描述用户u和物品i的特征，它们可以被定制以支持用户和项目的建模。由于本文工作的重点是纯粹的协同过滤设置，我们只使用一个用户的身份和一个物品作为输入特征，将其转换为一个具有one-hot编码的二值化稀疏向量。注意，有了这样一个用于输入的通用特征表示，我们的方法可以很容易地进行调整，通过使用内容特征来表示用户和项目，从而解决冷启动问题。
- Embedding及以上层论文如下所述：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/9-3.jpg)

#### NCF的学习

- 为了学习模型参数，现有的pointwise方法大多使用了平方损失的回归：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/9-4.jpg)

- 考虑到隐式反馈的一类性质，我们可以将 $y_{ui}$ 的值看作一个标签—1表示物品i与用户u相关，否则为0。预测得分 $\hat y_{ui}$ 表示了用户u和物品i之间相关-的可能性。赋予NCF这种概率解释,我们需要限制输出 $\hat y_{ui}$ 在[0,1]之间,它可以很容易地通过使用概率函数(例如,Logistic或Probit函数)作为输出层 激活函数。在上述介绍的环境下，定义概率函数如下所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/9-5.jpg)

#### Generalized Matrix Factorization (GMF)

- 作者用神经网络实现了矩阵分解功能，并在最终的$\hat y_{ui}$的输出上多加了一个sigmoid实现非线性变换。作者将这一用神经网络泛化后的矩阵分解算法称为泛化矩阵分解（GMF，Generalized Matrix Factorization）模型。,具体变换如下:

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/9-6.jpg)

#### Multi-Layer Perceptron (MLP)

- 一个n层的神经网络，每层神经元个数递减，每一层的激活函数援用ReLU函数加快收敛。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/9-7.jpg)

#### Fusion of GMF and MLP

- 到目前为止，我们已经开发了两个NCF实例：GMF采用线性核函数对潜在特征交互进行建模，MLP采用非线性核函数从数据中学习交互函数。接下来的问题是:我们如何在NCF框架下融合GMF和MLP，使得它们可以相互增强，从而更好地对复杂的用户-物品矩阵迭代交互进行建模。
- 为了给融合模型提供更大的灵活性，我们允许GMF和MLP学习独立的Embedding，并通过连接它们的最后一个隐藏层来组合这两个模型。下图展示了所提出的方案。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/9-8.jpg)

- 论文原文中关于上图的解释：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/9-9.jpg)

