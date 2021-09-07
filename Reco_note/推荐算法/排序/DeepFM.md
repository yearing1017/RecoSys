#### 特征交叉的重要性

理解用户点击行为背后隐藏的交叉特征对于 `CTR` 预估非常重要。例如，对 `app store` 的研究表明：人们经常在用餐时间下载送餐 `app`。这说明：`app` 类别和时间戳构成的交叉特征可以作为 `CTR` 预估的信号

通常用户点击行为背后的特征之间的各种交互非常复杂，其中的低阶交叉特征和高阶交叉特征都能发挥重要作用。根据 `Wide&Deep` 的理解，与仅考虑其中一种情况相比，同时考虑低阶交叉特征和高阶交叉特征能够带来额外的收益

但是这个问题的挑战在于：**如何有效的构建交叉特征**

- 有些交叉特征很容易理解，可以由业务专家人工设计。如上面的 `app store` 例子
- 大多数交叉特征都隐藏在数据中，很难事先预知，只能通过机器学习自动识别。如，经典的关联规则 `“啤酒 & 尿布”` 是从数据中提取的，而不是专家人工设计的

论文`《DeepFM: A Factorization-Machine based Neural Network for CTR Prediction》` 提出了 `DeepFM` 模型，该模型结合了 `FM` 的推荐能力和 `DNN` 的特征学习能力，综合利用了低阶交叉特征和高阶交叉特征。其特点有：

- **`DeepFM` 集成了 `FM` 和 `DNN`，可以对低阶交叉特征和高阶交叉特征建模，同时无需对原始输入执行任何特征工程**
- **`DeepFM` 的 `wide` 二阶特征交叉部分和 `deep` 部分共享输入及`embedding`** 

#### DeepFM

`DeepFM` 模型由两种组件构成：`FM` 组件、`deep` 组件，它们**共享输入。这种共享输入使得`DeepFM` 可以同时从原始特征中学习低阶特征交互和高阶特征交互，完全不需要执行特征工程**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/32-1.jpg" style="zoom:50%;" />

假设**输入**包含 `sparse` 特征和 `dense` 特征。设输入向量 $\overrightarrow{\mathbf{x}}$，其中：
$$
\overrightarrow{\mathbf{x}}=<\overrightarrow{\mathbf{x}}_{\text {sparse }}^{(1)}, \cdots, \overrightarrow{\mathbf{x}}_{\text {sparse }}^{(K)}, \overrightarrow{\mathbf{x}}_{\text {dense }}>\in \mathbb{R}^{d}
$$
其中 $\overrightarrow{\mathbf{x}}_{\text {sparse }}^{(1)}$ 为 `field i` 的 `one-hot` 向量， $\overrightarrow{\mathbf{x}}_{\text {dense}}$ 为原始的 `dense` 特征， $<.>$为向量拼接，对于特征 `j`（即$x_j$ ）：

- **标量 $w_j$ 用于对它的一阶特征重要性进行建模，即 `FM` 组件左侧的 `+` 部分**
- **向量 $\overrightarrow{\mathbf{v_j}}$ 用于对它的二阶特征重要性进行建模，即 `FM` 组件右侧的 `x` 部分**
- **<u>向量 $\overrightarrow{\mathbf{v_j}}$  也作为 `deep` 组件的输入，从而对更高阶特征交互进行建模，即 `deep` 组件</u>**

最终模型联合了 `FM` 组件和 `deep` 组件的输出：
$$
\hat{y}=\operatorname{sigmoid}\left(\hat{y}_{F M}+\hat{y}_{D N N}\right)
$$
其中 $\hat{y} \in (0,1)$ 为模型预测的`CTR`，$\hat{y}_{FM}$ 为 `FM` 组件的输出，$\hat{y}_{DNN}$ 为 `deep` 组件的输出。

#### FM

`FM` 组件：该部分是一个 `FM` ，**用于学习一阶特征和二阶交叉特征**

**`FM` 组件由两种操作组成：加法 `Addition` 和内积 `Inner Product`：**
$$
\hat{y}_{F M}=\sum_{i=1}^{d}\left(w_{i} \times x_{i}\right)+\sum_{i=1}^{d} \sum_{i=j+1}^{d}\left(\overrightarrow{\mathbf{v}}_{i} \cdot \overrightarrow{\mathbf{v}}_{j}\right) \times x_{i} \times x_{j}
$$

- 第一项 `Addition Unit` 用于对一阶特征重要性建模
- 第二项 `Inner Product` 用于对二阶特征重要性建模

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/32-2.jpg" style="zoom:50%;" />

#### Deep

**`deep` 组件：该部分是一个全连接的前馈神经网络，用于学习高阶特征交互**

假设 `embedding` 层的输出为：$\overrightarrow{\mathbf{h}}^{(0)}=\left[\overrightarrow{\mathbf{e}}_{1}, \cdots, \overrightarrow{\mathbf{e}}_{m}\right]$,其中 $\overrightarrow{\mathbf{e}}_{i}$ 为`field i` 的 `embedding` 向量， $\overrightarrow{\mathbf{h}}^{(0)}$ 为前馈神经网络的输入。则有：
$$
\overrightarrow{\mathbf{h}}^{(l+1)}=\sigma\left(\mathbf{W}^{(l)} \overrightarrow{\mathbf{h}}^{(l)}+\overrightarrow{\mathbf{b}}^{(l)}\right)
$$
其中 $l$ 为第 $l$ 层，$\sigma(.)$ 为激活函数 最终有如下的输出，L为deep部分的网络深度
$$
y_{D N N}=\sigma\left(\overrightarrow{\mathbf{w}}_{d n n} \cdot \overrightarrow{\mathbf{h}}^{(L)}+b_{d n n}\right)
$$
<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/24-6.png" style="zoom:50%;" />

#### 问题

##### 两端的输入

FM分为两部分

- 线性部分，即LR部分，直接将原始的高维稀疏的one-hot 输入
- **二阶交叉部分与DNN共用Embedding化之后的低位稠密特征**

##### 学习参数的更新

- FM的二阶交叉部分，在学完Embedding之后，就不会再学了，直接拿来做内积使用，图中红色线
- 但是DNN连接到Embedding层的参数和FM的LR部分的参数是一直更新学习的，图中黑色线

##### 训练时

- Deep部分的参数大多集中在Embedding到DNN的连接处，模型的更新大部分时间花在这个部分；
- 可使用预训练Embedding的方法来解决

