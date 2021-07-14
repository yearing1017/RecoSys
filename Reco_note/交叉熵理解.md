#### 信息量

**一条信息的信息量大小和它的不确定性有很大的关系。一句话如果需要很多外部信息才能确定，我们就称这句话的信息量比较大。**

- 比如你听到“云南西双版纳下雪了”，那你需要去看天气预报、问当地人等等查证（因为云南西双版纳从没下过雪）。相反，如果和你说“人一天要吃三顿饭”，那这条信息的信息量就很小，因为条信息的确定性很高。

将事件 $x_0$ 的信息量定义如下（其中 $p(x_0)$ 表示事件 $x_0$ 发生的概率）：
$$
I\left(x_{0}\right)=-\log \left(p\left(x_{0}\right)\right)
$$

#### 熵

信息量是对于单个事件来说的，但是实际情况一件事有很多种发生的可能，比如掷骰子有可能出现6种情况，明天的天气可能晴、多云或者下雨等等。

**熵是表示随机变量不确定的度量，是对所有可能发生的事件产生的信息量的期望**。公式如下：
$$
H(X)=-\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(p\left(x_{i}\right)\right)
$$
其中一种比较特殊的情况就是掷硬币，只有正、反两种情况，该种情况（二项分布或者0-1分布）熵的计算可以简化如下：
$$
\begin{aligned} H(X) &=-\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(p\left(x_{i}\right)\right) \\ &=-p(x) \log (p(x))-(1-p(x)) \log(1-p(x)) \end{aligned}
$$

#### 相对熵 （KL散度）

**相对熵又称KL散度，用于衡量对于同一个随机变量x的两个分布p(x)和q(x)之间的差异**，KL散度的值越小表示两个分布越接近。

在机器学习中，**p(x)常用于描述样本的真实分布**，例如[1,0,0,0]表示样本属于第一类，**而q(x)则常常用于表示预测的分布**，例如[0.7,0.1,0.1,0.1]。显然使用q(x)来描述样本不如p(x)准确，q(x)需要不断地学习来拟合准确的分布p(x)

**KL散度的公式如下：**
$$
D_{K L}(p \| q)=\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(\frac{p\left(x_{i}\right)}{q\left(x_{i}\right)}\right)
$$

#### 交叉熵

将KL散度的公式进行log除法变形，得到如下：
$$
\begin{aligned} D_{K L}(p \| q) &=\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(p\left(x_{i}\right)\right)-\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(q\left(x_{i}\right)\right) \\ &=-H(p(x))+\left[-\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(q\left(x_{i}\right)\right)\right] \end{aligned}
$$
**前半部分就是p(x)的熵，后半部分就是交叉熵：**
$$
H(p, q)=-\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(q\left(x_{i}\right)\right)
$$
**机器学习中，我们常常使用KL散度来评估predict和label之间的差别，但是由于KL散度的前半部分是一个常量，所以我们常常将后半部分的交叉熵作为损失函数**

#### 机器学习中的交叉熵损失函数

二分类：
$$
L=-\frac{1}{N}\sum_{i=1}^{N} y^{(i)} \log \hat{y}^{(i)}+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right)
$$
多分类：
$$
L=-\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{(i,k)} \log \hat{y}_{(i,k)}
$$
**信息论里的交叉熵仅仅是针对一个样本的交叉熵**，在机器学习进行优化时，会把**所有样本的交叉熵值做一个平均**，即机器学习的交叉熵损失函数定义如下，假设有N个样本：
$$
J(w)=\frac{1}{N} \sum_{n=1}^{N} H\left(p_{n}, q_{n}\right)
$$
因为**交叉熵常用于解决分类问题**，而分类问题(我们一般说分类问题，是指单标签多分类问题)的概率本质是计算类别变量的广义的伯努利分布，所以**机器学习采用的是交叉熵的离散形式**：
$$
C E=-\sum_{i=1}^{K} t_{i} \log f(s_i)
$$
其中，$t_i$是类别标签，$s_i$ 是模型对第 $i$ 个类别计算得到的$score$， $f(s_i)$ 指的是 激活函数($sigmoid, softmax$) 对$score$的转换

综合以上两点，得到机器学习的交叉熵损失函数：
$$
J(w)=-\frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{K} t_{i} \log s_{i}
$$

#### 最小化交叉熵损失与极大似然

机器学习中，**二分类的交叉熵损失函数**
$$
J(w)=-\frac{1}{N} \sum_{n=1}^{N}\left[y_{n} \log \hat{y_{n}}+\left(1-y_{n}\right) \log \left(1-\hat{y_{n}}\right)\right]
$$
去掉 $\frac{1}{N}$ 并不影响函数的单调性，机器学习任务的也可以是最小化下面的交叉熵损失：
$$
J(w)= -\sum_{n=1}^{N}\left[y_{n} \log \hat{y_{n}}+\left(1-y_{n}\right) \log \left(1-\hat{y_{n}}\right)\right]
$$
去掉负号，等价于最大化下面这个函数：
$$
J(w)= \sum_{n=1}^{N}\left[y_{n} \log \hat{y_{n}}+\left(1-y_{n}\right) \log \left(1-\hat{y_{n}}\right)\right]
$$
上式就是**对伯努利分布求极大似然中的对数似然函数(log-likelihood)**

#### 伯努利分布的极大似然

有二元随机变量 $Y \in \{0, 1\}$ (例如：抛硬币实验)，设 $p(Y=1) = \beta$ ，那么它的**概率质量函数**(Probability Mass Function，PMF)为：
$$
P(Y \mid \beta)=\beta^{Y}(1-\beta)^{1-Y}
$$
现有$D = \{y_1,y_2,...y_n\}$是来自 $Y$ 的、数量为N的一个样本集，元素是0或1，似然函数为：
$$
P(D \mid \beta)=\prod_{i=1}^{N} P\left(Y=y_{i} \mid \beta\right)=\prod_{i=1}^{N} \beta^{y_{i}}(1-\beta)^{1-y_{i}}
$$
在机器学习模型中，对上述 $\beta$ 的定义为：
$$
\beta=p_{\theta}\left(Y=1 \mid x_{i}\right)
$$
其中，$X = \{x_1,x_2,...x_n\}$ ， $x_i \in X$ , $X$ 是 $D$ 中的**每个样本点对应类别的特征的集合**。**即给定模型参数 $\theta$ 和随机变量的样本点 $Y=1$ 的属性特征 $x_i$ ( $x_i$ 可以是一个向量)，让模型估计出事件 $Y=1$ 的概率(同时也是当前伯努利分布的参数)。**

于是，将似然函数的参数 $\beta$ 替换为$\theta$，所得：
$$
P(D \mid \theta, X)=\prod_{i=1}^{N} \beta^{y_{i}}(1-\beta)^{1-y_{i}} = \prod_{i=1}^{N} p_{\theta}\left(Y=1 \mid x_{i}\right)^{y_{i}}\left(1-p_{\theta}\left(Y=1 \mid x_{i}\right)\right)^{1-y_{i}}
$$
易得**对数似然函数**(log-likelihood)：
$$
\begin{array}{l}\mathcal{L}(\theta ; X, D)=\log \prod_{i=1}^{N} p_{\theta}\left(Y=1 \mid x_{i}\right)^{y_{i}}\left(1-p_{\theta}\left(Y=1 \mid x_{i}\right)\right)^{1-y_{i}} \\ =\sum_{i=1}^{N} \log p_{\theta}\left(Y=1 \mid x_{i}\right)^{y_{i}}\left(1-p_{\theta}\left(Y=1 \mid x_{i}\right)\right)^{1-y_{i}} \\ =\sum_{i=1}^{N} \log p_{\theta}\left(Y=1 \mid x_{i}\right)^{y_{i}}+\log \left(1-p_{\theta}\left(Y=1 \mid x_{i}\right)\right)^{1-y_{i}} \\ =\sum_{i=1}^{N} y_{i} \log p_{\theta}\left(Y=1 \mid x_{i}\right)+\left(1-y_{i}\right) \log \left(1-p_{\theta}\left(Y=1 \mid x_{i}\right)\right)\end{array}
$$

#### 极大似然估计与最小化交叉熵损失的转化过程

说明在伯努利分布下，**极大似然估计与最小化交叉熵损失其实是同一回事**：
$$
\begin{array}{l}\theta_{p}=\arg \max _{\theta} \sum_{i=1}^{N} y_{i} \log p_{\theta}\left(Y=1 \mid x_{i}\right)+\left(1-y_{i}\right) \log \left(1-p_{\theta}\left(Y=1 \mid x_{i}\right)\right) \\ =\arg \max _{\theta} \sum_{i=1}^{N} y_{i} \log \hat{y}_{i}+\left(1-y_{i}\right) \log \left(1-\hat{y}_{i}\right) \\ =\arg \min _{\theta}-\sum_{i=1}^{N} y_{i} \log \hat{y}_{i}+\left(1-y_{i}\right) \log \left(1-\hat{y}_{i}\right) \\ =\arg \min _{\theta} \sum_{i=1}^{N} H\left(y_{i}, \hat{y}_{i}\right)\end{array}
$$
