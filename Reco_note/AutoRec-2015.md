### 前言

- 本文学习的模型是2015年由澳大利亚国立大学提出的[AutoRec](https://link.zhihu.com/?target=https%3A//www.researchgate.net/publication/311491420_AutoRec_Autoencoders_Meet_Collaborative_Filtering%3Fev%3Dauth_pub)，参考：《深度学习推荐系统--王喆》

- 协同过滤模型的目的是利用用户对商品的偏好信息来提供个性化的推荐。
- AutoRec是一个新型的**基于自编码器的协同过滤模型**。论文作者认为AutoRec与现有的将玻尔兹曼机用于协同过滤的神经方法相比具有表征和计算上的优势，并从经验上证明了AutoRec优于当前最先进的方法。

### 1. 自编码器模型

- 在基于评分的协同过滤中，假设有m个用户，n个物品，并且有一个用户-物品的评分矩阵$R \in R^{m \times n}$ 
- 每个用户向量$u \in U=\{1 ... m\}$，可以被向量表示为$\mathbf{r}^{(u)}=\left(R_{u 1}, \ldots R_{u n}\right) \in \mathbb{R}^{n}$ 
- 类似的，每个物品$i \in I=\{1 ... n\}$也可被向量表示为$\mathbf{r}^{(i)}=\left(R_{1 i}, \ldots R_{m i}\right) \in \mathbb{R}^{m}$

- 论文的目的就在于**设计一个基于物品（用户）的自编码器**，能够将输入的$r^{(i)}$或$r^{(u)}$映射到一个低维空间，并且在输出中重新构建$r^{(i)}$或$r^{(u)}$来预测缺失的评分进而进行推荐。
- 自编码器的目标函数为：

$$
\min _{\theta} \sum_{\mathbf{r} \in \mathbf{S}}\|\mathbf{r}-h(\mathbf{r} ; \theta)\|_{2}^{2}
$$

- AutoRec要解决的问题是**构建一个重建函数h(r;θ)，使所有该重建函数生成的评分向量与原评分向量的平方残差和最小**，如上式所示。其中，S是所有数据向量的集合。在完成自编码器的训练后，就相当于在重新函数h(r;θ)中存储了所有数据向量的“精华”。**自编码器相当于完成了数据压缩和降维的工作。**

### 2. AutoRec模型结构

- 下图为AutoRec的结构图，从图中可以看出，网络的输入层是物品的评分向量r,输出层是一个多分类层、蓝色的神经元代表模型的k维单隐层，其中k<<m

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/7-1.png)

- V和W分别是输入层到隐层，以及隐层到输出层的参数矩阵。该模型结构代表的重建函数的具体形式为：

$$
h(\mathbf{r} ; \theta)=f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{r}+\boldsymbol{\mu})+\mathbf{b})
$$

- 其中，f(.),g(.)分别为输出层神经元和隐层神经元的激活函数。参数θ通过反向传播来进行学习

- 为防止重构函数的过拟合，在加入L2正则化项后，AutoRec目标函数的具体形式如下

$$
\left.\min _{\theta} \sum_{i=1}^{n} \| \mathbf{r}^{(i)}-h\left(\mathbf{r}^{(i)} ; \theta\right)\right) \|_{\mathcal{O}}^{2}+\frac{\lambda}{2} \cdot\left(\|\mathbf{W}\|_{F}^{2}+\|\mathbf{V}\|_{F}^{2}\right)
$$

- 当输入物品i的评分向量为$r^{(i)}$时，模型的输出向量$h(r^i;\theta)$，就是对所有用户对物品i的评分预测。那么，其中的第u维就是用户u对物品i的预测$\hat{R}_{u i}$：

$$
\hat{R}_{u i}=\left(h\left(\mathbf{r}^{(i)} ; \hat{\theta}\right)\right)_{u}
$$

- 相比之下，矩阵分解学习的是线性的潜在表示，而自动编码器可以通过激活函数g(.)学习到非线性的潜在表示。以上介绍的AutoRec输入向量是物品的评分向量，因此可称为I-AutoRec(Item based AutoRec)

### 3. 优点及局限性

- AutoRec模型使用一个单隐层的AutoEncoder泛化用户或物品评分，使模型具有一定的泛化和表达能力。由于AutoRec模型的结构比较简单，使其存在一定的表达能力不足的问题。
- 从深度学习的角度来说，AutoRec模型的提出，拉开了使用深度学习的思想解决推荐问题的序幕，为复杂深度学习网络的构建提供了思路。