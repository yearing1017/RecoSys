### 1. 问题背景

- Focal Loss的引入主要是为了解决**难易样本数量不平衡（注意，有区别于正负样本数量不平衡）**的问题；

### 2. Focal Loss 原理分析

- 在**二分类**的情况下，模型最后需要预测的结果只有两种情况，对于每个类别我们的预测得到的概率为p和1-p，交叉熵的公式如下：

$$
L=\frac{1}{N} \sum_{i} L_{i}=\frac{1}{N} \sum_{i}-\left[y_{i} \cdot \log \left(p_{i}\right)+\left(1-y_{i}\right) \cdot \log \left(1-p_{i}\right)\right]
$$

- 即可转型为：

$$
C E=\left\{\begin{aligned}-\log (p), & \text { if } y=1 \\-\log (1-p), & \text { if } y=0 \end{aligned}\right.
$$

- 其中，**y表示样本的label，正为1，负为0；p表示样本预测为正类的概率**

- 为了**解决正负样本不平衡**的问题，我们通常会在交叉熵损失的前面加上一个参数 $\alpha$，如下：

$$
C E=\left\{\begin{aligned}-\alpha \log (p), & \text { if } y=1 \\-(1-\alpha) \log (1-p), & \text { if } y=0 \end{aligned}\right.
$$

- **对于负样本数量多，给负样本的权重小一些，对于正样本数量少，给正样本的权重大一些**
- 但这并不能解决全部问题。根据正、负、难、易，样本一共可以分为以下四类：

|      |  难  |  易  |
| :--: | :--: | :--: |
|  正  | 正难 | 正易 |
|  负  | 负难 | 负易 |

- **尽管** $\alpha$ **平衡了正负样本，但对难易样本的不平衡没有任何帮助**。而实际上，目标检测中大量的候选目标都是像下图一样的易分样本。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/CNN/19-13.jpg" style="zoom:50%;" />

- 这些样本的损失很低，但是由于数量极不平衡，易分样本的数量相对来讲太多，最终主导了总的损失。而本文的作者认为，**易分样本（即，置信度高的样本）对模型的提升效果非常小，模型应该主要关注那些难分样本**（**这个假设是有问题的，是GHM损失的主要改进对象**）

- 这个时候，就有了Focal Loss，一个简单的思想：**把高置信度(p)样本的损失再降低一些！**

$$
F L=\left\{\begin{array}{cl}-(1-p)^{\gamma} \log (p), & \text { if } y=1 \\ -p^{\gamma} \log (1-p), & \text { if } y=0\end{array}\right.
$$

- $(1-p)^{\gamma}$ 称为**调制系数**，**p趋于0时，调制系数趋于1，对总的loss贡献大；p趋于1时，调制系数趋于0，对总的loss减小**
- 举个例：$\gamma = 2$ 时，p = 0.968，则 $(1-0.968)^2 = 0.001$，**损失也随之衰减了1000倍**。
- Focal Loss的最终形式结合了上面的公式（2）。这很好理解，公式(3)解决了难易样本的不平衡，公式(2)解决了正负样本的不平衡，将公式（2）与（3）结合使用，同时解决正负难易2个问题！如下：

$$
F L=\left\{\begin{aligned}-\alpha(1-p)^{\gamma} \log (p), & \text { if } \quad y=1 \\-(1-\alpha) p^{\gamma} \log (1-p), & \text { if } \quad y=0 \end{aligned}\right.
$$

- 实验表明：$\gamma=2; \alpha=0.25$ 时效果最好，实现代码如下：