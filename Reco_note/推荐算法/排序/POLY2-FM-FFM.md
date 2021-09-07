#### POLY2模型

##### 模型简介

`LR` 模型只考虑特征之间的线性关系，而`POLY2` 模型考虑了特征之间的非线性关系

- **捕获非线性特征的一个常用方法是采用核技巧，如高斯核`RBF`，将原始特征映射到一个更高维空间。在这个高维空间模型是线性可分的，即：只需要考虑新特征之间的线性关系。但是核技巧存在计算量大、内存需求大的问题**
- **论文 `Training and Testing Low-degree Polynomial Data Mappings via Linear SVM` 提出多项式映射 `polynomially mapping` 数据的方式来提供非线性特征，在达到接近核技巧效果的情况下大幅度降低内存和计算量**

`LR`模型
$$
\begin{array}{c}
z(\overrightarrow{\mathbf{x}})=w_{0}+\sum_{i=1}^{n} w_{i} \times x_{i} \\
y(\overrightarrow{\mathbf{x}})=\frac{1}{1+\exp (-z(\overrightarrow{\mathbf{x}}))}
\end{array}
$$
`POLY2`模型公式：
$$
\begin{array}{c}
z(\overrightarrow{\mathbf{x}})=w_{0}+\sum_{i=1}^{n} w_{i} \times x_{i}+\sum_{i=1}^{n} \sum_{j=i}^{n} w_{i, j} \times x_{i} \times x_{j} \\
y(\overrightarrow{\mathbf{x}})=\frac{1}{1+\exp (-z(\overrightarrow{\mathbf{x}}))}
\end{array}
$$
由此可见新增的组合特征数：$\frac{n \times (n-1)}{2}$

##### 优缺点

优点：除了线性特征之外，还能够**通过特征组合自动捕获二阶特征交叉产生的非线性特征**。

缺点：

- 参数太多导致计算量和内存需求发生爆炸性增长。如计算广告场景中，原始样本特征可能达到上万甚至百万级别，则特征的交叉组合达到上亿甚至上万亿。
- 数据稀疏导致二次项参数训练困难，非常容易过拟合。参数$w_{i,j}$ 的训练需要大量的$x_i,x_j$都非零的样本。而大多数应用场景下，原始特征本来就稀疏（非零的样本数很少），特征交叉之后更为稀疏（非零的样本数更少）。这使得训练 $w_{i,j}$ 的样本明显不足，很容易发生过拟合。

#### FM模型

##### 模型简介

FM模型也直接引入任意两个特征的二阶特征组合，和POLY2模型最大的不同，在于特征组合权重的计算方法。

**FM对于每个特征，学习一个大小为k的一维向量**，于是，两个特征 $x_i$ 和 $x_j$ 的特征组合的权重值，通过特征对应的向量 $v_i$ 和 $v_j$ 的内积 $<v_i,v_j>$ 来表示。**这本质上是在对特征进行embedding化表征**

模型公式
$$
\hat{y}(\overrightarrow{\mathbf{x}})=w_{0}+\sum_{i=1}^{K} w_{i} \times x_{i}+\sum_{i=1}^{K} \sum_{j=i+1}^{K} \hat{w}_{i, j} \times x_{i} \times x_{j}
$$
其中 $\hat{w}_{i, j}$ 是交叉特征的参数，它由一组参数定义：
$$
\hat{w}_{i, j}=<\overrightarrow{\mathbf{v}}_{i}, \overrightarrow{\mathbf{v}}_{j}>=\sum_{l=1}^{d} v_{i, l} \times v_{j, l} \\
\hat{\mathbf{W}}=\left[\begin{array}{cccc}
\hat{w}_{1,1} & \hat{w}_{1,2} & \cdots & \hat{w}_{1, K} \\
\hat{w}_{2,1} & \hat{w}_{2,2} & \cdots & \hat{w}_{2, K} \\
\vdots & \vdots & \ddots & \vdots \\
\hat{w}_{K, 1} & \hat{w}_{K, 2} & \cdots & \hat{w}_{K, K}
\end{array}\right]=\mathbf{V}^{T} \mathbf{V}=\left[\begin{array}{c}
\overrightarrow{\mathbf{v}}_{1}^{T} \\
\overrightarrow{\mathbf{v}}_{2}^{T} \\
\vdots \\
\overrightarrow{\mathbf{v}}_{K}^{T}
\end{array}\right]\left[\begin{array}{llll}
\overrightarrow{\mathbf{v}}_{1} & \overrightarrow{\mathbf{v}}_{2} & \cdots & \overrightarrow{\mathbf{v}}_{K}
\end{array}\right]
$$
模型待求解的参数：
$$
\begin{array}{r}
w_{0} \in \mathbb{R}, \overrightarrow{\mathbf{w}} \in \mathbb{R}^{n} \\
\mathbf{V}=\left(\overrightarrow{\mathbf{v}}_{1}, \cdots, \overrightarrow{\mathbf{v}}_{K}\right) \in \mathbb{R}^{d \times K}
\end{array}
$$
其中：

-  $w_0$表示全局偏差
-  $w_i$用于捕捉第$i$个特征和目标之间的关系
-  $\hat{w}_{i,j}$用于捕捉 二路交叉特征和目标之间的关系
-  $v_{K}$代表特征的`representation vector`，它是 V 的第 i 列

##### 模型复杂度

https://zhuanlan.zhihu.com/p/58160982   另外一种复杂度优化化简方法

`FM`  模型的计算复杂度为：$O(K \times K \times d) = O(K^2d)$，但是经过数学转换之后其计算复杂度可以降低到 $O(Kd)$

**利用类比的转换**：$(a+b)^2 = a^2 + b^2 + 2ab  => ab = [\frac{1}{2}[(a+b)^2] - (a^2 + b^2)] $
$$
\begin{aligned} \sum_{i=1}^{K} \sum_{j=i+1}^{K} \hat{w}_{i, j} \times x_{i} \times x_{j}=\sum_{i=1}^{K} \sum_{j=i+1}^{K} \sum_{l=1}^{d} v_{i, l} \times v_{j, l} \times x_{i} \times x_{j} \\=\sum_{l=1}^{d}\left(\sum_{i=1}^{K} \sum_{j=i+1}^{K}\left(v_{i, l} \times x_{i}\right) \times\left(v_{j, l} \times x_{j}\right)\right) \\= \sum_{l=1}^{d} \frac{1}{2}\left(\left(\sum_{i=1}^{K} v_{i, l} \times x_{i}\right)^{2}-\sum_{i=1}^{K} v_{i, l}^{2} \times x_{i}^{2}\right) \end{aligned}
$$
因此有：
$$
\hat{y}(\overrightarrow{\mathbf{x}})=w_{0}+\sum_{i=1}^{K} w_{i} \times x_{i}+\frac{1}{2} \sum_{l=1}^{d}\left(\left(\sum_{i=1}^{K} v_{i, l} \times x_{i}\right)^{2}-\sum_{i=1}^{K} v_{i, l}^{2} \times x_{i}^{2}\right)
$$
其计算复杂度为$O(Kd)$

##### 特征类型

`FM` 模型可以处理不同类型的特征：

- 离散型特征 `categorical`：`FM` 对离散型特征执行 `one-hot` 编码

  如，性别特征：“男” 编码为 `(0,1)`，“女” 编码为 `(1,0)` 

- 离散集合特征 `categorical set`：`FM` 对离散集合特征执行类似 `one-hot` 的形式，但是执行样本级别的归一化

  如，看过的历史电影。假设电影集合为：“速度激情9，战狼，泰囧，流浪地球”。如果一个人看过 “战狼，泰囧，流浪地球”， 则编码为 `(0,0.33333,0.33333,0.33333)` 

- 数值型特征 `real valued`：`FM`直接使用数值型特征，不做任何编码转换

##### 优势

- 给定特征 `representation` 向量的维度时，预测期间计算复杂度是线性的。

- 在**交叉特征高度稀疏的情况下，参数仍然能够估计。** 

  因为交叉特征的参数不仅仅依赖于这个交叉特征，还依赖于所有相关的交叉特征。这相当于增强了有效的学习数据。

- 能够泛化到未被观察到的交叉特征。

  设交叉特征 `“看过电影 A 且 年龄等于20”` 从未在训练集中出现，但出现了 `“看过电影 A”`相关的交叉特征、以及 `“年龄等于20”`相关的交叉特征。于是可以从这些交叉特征中分别学习 `“看过电影 A”` 的 `representation` 、`“年龄等于20”` 的 `representation`，最终泛化到这个未被观察到的交叉特征。

#### FFM模型

##### 模型简介

使用例子引入：POLY2 ==> FM ==> FFM

考虑一组特征：“性别、年龄、城市”。为简化讨论，假设：“年龄”取值集合为 `[18,19,20]`， “城市” 取值集合为 `[北京,上海,广州,深圳]` 。

把离散特征 `one-hot` 编码，设各 `binary` 特征分别记作：`male,female,age18,age19,age20,bj,sh,gz,sz`， `y` 表示样本标签（`-1` 表示不感兴趣，`+1` 表示感兴趣）。数据如下表所示：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/32-3.jpg" style="zoom:50%;" />

- **POLY2模型**

$$
\hat{y}(\overrightarrow{\mathbf{x}})=w_{0}+\sum_{i=1}^{K} w_{i} \times x_{i}+\sum_{i=1}^{K} \sum_{j=i+1}^{K} w_{i, j} \times x_{i} \times x_{j}
$$

参数个数为 $O(K^2)$，计算复杂度为 $O(K^2)$

- **FM模型**

$$
\begin{array}{c}\hat{y}(\overrightarrow{\mathbf{x}})=w_{0}+\sum_{i=1}^{K} w_{i} \times x_{i}+\sum_{i=1}^{K} \sum_{j=i+1}^{K} \hat{w}_{i, j} \times x_{i} \times x_{j} \\ \hat{w}_{i, j}=<\overrightarrow{\mathbf{v}}_{i}, \overrightarrow{\mathbf{v}}_{j}>=\sum_{l=1}^{d} v_{i, l} \times v_{j, l}\end{array}
$$

参数个数为 $O(K \times d)$，计算复杂度为 $O(K \times d)$

- `FM` 要优于 `POLY2` ，原因是：

  - **交叉特征非零的样本过于稀疏使得无法很好的估计** ；但是在 `FM`中，**交叉特征的参数可以从很多其它交叉特征中学习，使得参数估计更准确**

    如：交叉特征 $(male=1, age19=1)$从未出现过，因此在 `POLY2` 模型中参数$w_{male=1, age19=1}$ 根本无法学习。而在 `FM` 模型中 `male=1` 的 `representation` 向量可以从以下交叉特征的样本中学习：$(male=1, age18=1)$ $(male=1, age20=1)$ $(male=1, sh=1)$ $(male=1,sz=1)$  `age19=1` 的 `representation` 向量可以从交叉特征 的样本中学习

  - **另外 `FM` 还可以泛化到没有见过的交叉特征**。如：交叉特征 $(male=1, age19=1)$从未在训练样本中出现过，但是在预测阶段 `FM` 模型能够较好的预测该交叉特征的测试样本

**在 `FM` 模型中，每个特征的 `representation` 向量只有一个**。$\hat{w}_{\text {male }=1, \text { age }=18}, \hat{w}_{\text {male }=1, \text { age }=20}, \hat{w}_{\text {male }=1, \mathrm{sh}=1}$用到的是同一个向量$v_{male}$

论文`Field-aware Factorization Machines for CTR Prediction` 提出的 `FFM` 算法认为：`age=18` 和 `sh=1` 之间的区别，远远大于 `age=18` 和 `age=20` 之间的区别。

因此，**`FFM` 算法将特征划分为不同的域`field`**。其中：

- 特征 $(male=1, female=1)$ 属于性别域 `gender field` 。
- 特征$(age18=1,age19=1,age20=1)$ 属于年龄域 `age field` 。
- 特征$(bj=1,sh=1,gz=1,sz=1)$ 属于城市域 `city field` 。

**`FFM` 中每个特征的 `representation` 向量有多个，用于捕捉该特征在不同`field` 中的含义**

如：特征 `male=1` 具有两个 `representation` 向量：

- 当用于计算 `age field` 域的交叉特征时，采用 $v_{male=1,age}$
- 当用于计算 `city field` 域的交叉特征时，采用 $v_{male=1,city}$

其它特征依次类推。

注意：**`male=1` 没有 `gender field` 域的交叉特征。因为 `one-hot` 编码的原因 ，交叉特征`male=1,female=1` 一定不能同时存在，所以不用计算 $v_{male=1,female=1}$**

##### 模型

`FFM` 模型用数学语言描述为：
$$
\begin{array}{c}\hat{y}(\overrightarrow{\mathbf{x}})=w_{0}+\sum_{i=1}^{K} w_{i} \times x_{i}+\sum_{i=1}^{K} \sum_{j=i+1}^{K} \hat{w}_{i, j} \times x_{i} \times x_{j} \\ \hat{w}_{i, j}=<\overrightarrow{\mathbf{v}}_{i, f_{j}}, \overrightarrow{\mathbf{v}}_{j, f_{i}}>=\sum_{l-1}^{d} v_{i, f_{j} l} \times v_{j, f_{i}, l}\end{array}
$$
其中：$f_i$ 表示第 $i$ 个特征所属的 `field` ，一共有 $F$ 个`field` $(1<=F<=K)$ ，参数数量为 $O(K \times d \times F)$

**和 `FM` 相比，通常 `FFM` 中 `representation` 向量的维度要低的多**

`FFM` 每个`representation` 向量的学习只需要特定 `field` 中的样本。

如：学习  $v_{male=1,age}$ 时，只需要考虑交叉特征 $(male=1, age18=1)$ $(male=1, age20=1)$  的样本，而不需要考虑交叉特征 $(male=1, sh=1)$ $(male=1,sz=1)$ 的样本

> 因为上面的示例中，交叉特征 $(male=1, age19=1)$ 未出现；如果该交叉特征出现，则也需要考虑该交叉特征的样本

