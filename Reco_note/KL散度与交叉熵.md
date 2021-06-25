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

