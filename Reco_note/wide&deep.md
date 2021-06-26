#### 记忆和泛化

`memorization`：学到 `item` 或者 `feature` 共现关系，并基于历史数据中的这种相关性来推荐。基于`memorization` 的推荐通常更具有话题性，并且和用户已经发生行为的 `item` 直接关联

`generalization`：根据 `item` 或者 `feature` 的共现关系，探索过去从未发生或者很少发生的新特征组合。基于 `generalization` 的推荐通常更具有多样性

广义**线性模型**通常对特征执行 `one-hot` 编码。如 `性别=男` 表示特征：如果用户是 `男性`，则该特征为 1 

- **通过特征交叉可有效达到 `memorization`**。如特征交叉 `AND(性别=男，曾经购买汽车=奇瑞QQ)` ：当用户是男性、且曾经购买奇瑞 `QQ` 汽车时该交叉特征为 1
- 如果希望**提升泛化能力，则可以提升特征的粒度**。如： `AND(性别=男，曾经购买汽车=10万以下汽车)` 这种方式的限制是无法推广到训练集中没有出现过的 `query-item` 或者 `feature pair` 

**基于 `embedding` 的模型（如 `FM` 或者 `DNN` ）为每个 `query` 和 `item` 学习一个低维的 `dense embedding` 向量，通过 `embedding` 向量来泛化到训练集中未见过的 `query-item feature pair` ，同时也缓解了特征工程的代价**

- 但当 `query-item` 矩阵非常稀疏且矩阵的秩较高时（如：用户具有特定偏好，产品非常小众），很难学到有效的 `query/item` 的低维表达。

  此时大多数 `query-item pair` 之间不应该存在任何交互，但是 `dense embedding` 仍然给出了非零的预测结果。这会导致严重的过拟合，并给出一些不怎么相关的推荐结果。在这种场景下，基于特征交叉的广义线性模型能够记住这些特定偏好或者小众产品的 `exception rule` 。

广义线性模型（称为 `wide` 模型）可以通过大量交叉特征来记住特征交互 `feature interaction` ，即 `memorization` 。其优点是可解释性强，缺点是：为了提升泛化能力，需要人工执行大量的特征工程

深度神经网络模型（称为 `deep` 模型）只需要执行较少的特征工程即可泛化到未出现的特征组合，即 `generalization` 。其优点是泛化能力强，缺点是容易陷入过拟合

即：广义线性模型表达能力不强，容易欠拟合；深度神经网络模型表达能力太强，容易过拟合。二者结合就能取得平衡

#### 模型

`Wide & Deep` 模型主要用于 `ranking` 精排模块，它包含一个 `linear model:LM` 部分和一个 `neural network:NN` 部分

设模型的输入特征向量为$\overrightarrow{\mathbf{x}}=\left(x_{1}, \cdots, x_{d}\right)^{T}$ 是一个$d$维的特征向量（经过one-hot后），$\phi(.)$ 表示特征交叉转换函数，$\phi(\overrightarrow{\mathbf{x}})$ 包含转换后的特征

- `LM` 部分：即左侧的 `wide` 子模型，它是一个线性模型

$$
y=\overrightarrow{\mathbf{w}} \cdot<\overrightarrow{\mathbf{x}}, \phi(\overrightarrow{\mathbf{x}})>+b
$$

​	其中 $<>$ 表示特征拼接，$\overrightarrow{\mathbf{w}}$ 是模型参数（ 表示交叉特征的数量）， $b$ 为偏置。

- `NN` 部分：即右侧的 `deep` 子模型，它是一个 `DNN` 模型
  - 输入层：为了缓解模型的输入大小，`DNN` 的所有离散特征的输入都是原始特征，而没有经过 `one-hot` 编码转换
  - 第一层 `embedding` 层：将高维稀疏的 `categorical` 特征转换为低维的 `embedding` 向量。论文中的`embedding` 向量维度为 32 维
  - 第二层特征拼接层：将所有的 `embedding` 向量拼接成一个 `dense feature` 向量。论文中该向量维度为 1200维
  - 后续每一层都是全连接层：

$$
\overrightarrow{\mathbf{h}}^{(l+1)}=\sigma\left(\mathbf{W}^{(l)} \overrightarrow{\mathbf{h}}^{(l)}+\overrightarrow{\mathbf{b}}^{(l)}\right)
$$

​			其中 $l$ 为层的编号， $\sigma()$为激活函数

模型联合了 `wide` 和 `deep` 的输出：
$$
\hat{y}=p(y=1 \mid \overrightarrow{\mathbf{x}})=\operatorname{sigmoid}\left(\overrightarrow{\mathbf{w}}_{w i d e} \cdot<\overrightarrow{\mathbf{x}}, \phi(\overrightarrow{\mathbf{x}})>+\overrightarrow{\mathbf{w}}_{\text {deep }} \cdot \overrightarrow{\mathbf{h}}^{(L)}+b\right)
$$
其中$\overrightarrow{\mathbf{w}}_{w i d e}$ 为 `wide` 部分的权重， $\overrightarrow{\mathbf{w}}_{deep}$为 `deep` 部分的权重， $b$ 为全局偏置

模型的损失函数为负的对数似然，并通过随机梯度下降来训练：
$$
\mathcal{L}=-\frac{1}{N} \sum_{i=1}^{N}\left(y_{i} \log \hat{y}_{i}+\left(1-y_{i}\right) \log \left(1-\hat{y}_{i}\right)\right)
$$
![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/31-8.png)

`Wide&Deep` 模型与 `LM & DNN` 的 `ensemble` 集成模型不同。

- 在集成模型中，每个子模型都是独立训练的，只有预测时才将二者结合在一起。在 `Wide&Deep` 模型中，每个子模型在训练期间就结合在一起，共同训练。
- 在集成模型中，每个子模型必须足够大从而足够健壮，使得子模型集成之后整体的 `accuracy` 等性能足够高。在 `Wide&Deep` 模型中，每个子模型都可以比较小，尤其是 `wide` 部分只需要少量的特征交叉即可

#### 模型Pipeline



