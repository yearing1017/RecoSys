### 前言

- 本篇文章前面大篇幅**介绍SVM的概念及目标函数的推导。**
- 没有对SMO算法进行推导，已从入门推导到放弃...文章大概写了两周（之后应该会重新看下的）
- 主要内容为：**线性可分SVM，线性不可分SVM，软间隔支持向量机。**

### 1. 什么是SVM

- 先看一个线性分类问题，如下图所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/9-1.jpg)

- 如图(a)所示，为了将红色与蓝色球分开，我们需要画一条分隔线。**SVM就是试图将线画到一个最佳位置，即使得线的两侧尽量有更大的间隙，这个间隙就是球到所划线的距离。**
- 图(b)、(c)给出了两组方案，其中黑色实线为分界线，术语称为”决策面“。在效果上看，两组方案差不多，但是在性能上看，就有差距了。如下所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/9-2.jpg)

- 在"决策面"不变的情况下，在图上又添加了一个红点。可以看到，分类器B依然能很好的分类结果，而分类器C则出现了分类错误。
- 显然分类器B的"决策面"放置的位置优于分类器C的"决策面"放置的位置，SVM算法也是这么认为的，它的依据就是分类器B的分类间隔比分类器C的分类间隔大。
- 这里涉及到第一个SVM独有的概念**"分类间隔"**：
  - 在保证决策面方向不变且不会出现错分样本的情况下移动决策面，会在原来的决策面两侧找到两个极限位置（越过该位置就会产生错分现象），如虚线所示。
  - 虚线的位置由决策面的方向和距离原决策面最近的几个样本的位置决定。而这两条平行虚线正中间的分界线就是在保持当前决策面方向不变的前提下的最优决策面。
  - 两条虚线之间的垂直距离就是这个最优决策面对应的分类间隔。

- 显然每一个可能把数据集正确分开的方向都有一个最优决策面（有些方向无论如何移动决策面的位置也不可能将两类样本完全分开），而不同方向的最优决策面的分类间隔通常是不同的，那个**具有“最大间隔”的决策面**就是SVM要寻找的**最优解**。
- 这个真正的最优解对应的**两侧虚线所穿过的样本点**（也可说为离决策面最近的那些点），就是SVM中的支持样本点，称为**"支持向量"**。

### 2. SVM数学建模

#### 2.1 建模分析

- 求解上述决策面的过程，就是最优化，最优化包括两个因素：
  - 目标函数：希望什么东西的什么指标达到最好，SVM算法中，目标函数是**”分类间隔“**。
  - 优化对象：希望改变什么因素使得目标函数最优，SVM算法中，优化对象就是**”决策面“**。
- 数学建模，先在二维建模，后推广到多维。

#### 2.2  决策面

- 在二维空间下的一条直线方程如下所示：

$$
y = ax +b
$$

- 现在让x轴变为x1，y轴变为x2：

$$
x_2 = ax_1 + b
$$

- 移项得：

$$
ax_1 - x_2 + b = 0
$$

- 将公式向量化得：

$$
\left[\begin{array}{ll}{a} & {-1}\end{array}\right]\left[\begin{array}{l}{x_{1}} \\ {x_{2}}\end{array}\right]+b=0
$$

- 进一步向量化，使用w列向量和x列向量以及标量r：

$$
\boldsymbol{\omega}^{T} \boldsymbol{x}+\gamma=0
$$

- 其中，向量w和x为：

$$
\boldsymbol{w} = {[w_1 , w_2]}^T,\boldsymbol{x} = {[x_1,x_2]}^T
$$

- 这里w1=a，w2=-1。我们都知道，最初的那个直线方程a和b的几何意义，a表示直线的斜率，b表示截距，a决定了直线与x轴正方向的夹角，b决定了直线与y轴交点位置。那么向量化后的直线的w和r的几何意义是什么呢？
- 现在假设$a=\sqrt{3}, b=0$，则$w = {[\sqrt{3},-1]}^T$，在坐标轴上画出直线与向量w：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/9-3.jpg)

- 蓝色的线代表向量w，红色的线代表直线y。我们可以看到向量w和直线的关系为垂直关系。这说明了向量w也控制这直线的方向，只不过是与这个直线的方向是垂直的。标量$\gamma$的作用也没有变，依然决定了直线的截距。此时，我们称w为直线的法向量。
- 二维空间的直线方程已经推导完成，将其推广到n维空间，就变成了超平面方程。(一个超平面，在二维空间的例子就是一个直线)但是它的公式没变，依然是：

$$
\boldsymbol{\omega}^{T} \boldsymbol{x}+\gamma=0
$$

- 其中向量为：

$$
\begin{array}{l}{\boldsymbol{\omega}=\left[\omega_{1}, \omega_{2}, \cdots, \omega_{n}\right]^{T}} \\ {\boldsymbol{x}=\left[x_{1}, x_{2}, \cdots, x_{n}\right]^{T}}\end{array}
$$

- 以上方程即为**“决策面”方程**，这就是所求的超平面方程。

#### 2.3 分类间隔

- 还是从二维的例子出发，进行理解推导：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/9-4.jpg)

- 间隔的大小实际上就是支持向量对应的样本点到决策面的距离的二倍。那么图中的距离d我们怎么求？我们高中都学过，点到直线的距离距离公式如下：

$$
d=\left|\frac{A x_{0}+B y_{0}+C}{\sqrt{A^{2}+B^{2}}}\right|
$$

- 其中公式中的直线方程为$Ax_0+By_0+C=0$，点P的坐标为$(x_0,y_0)$。
- 现在将直线推展到多维来推导距离大小：

$$
d=\frac{\left|\boldsymbol{\omega}^{T} \boldsymbol{x}+\gamma\right|}{\|\boldsymbol{\omega}\|}
$$

- 这个d就是"分类间隔"。其中$||w||$表示w的二范数，求所有元素的平方和，然后再开方。比如对于二维平面：

- $$
  \begin{array}{c}{\boldsymbol{\omega}=\left[\omega_{1}, \omega_{2}\right]^{T}} \\ {\|\omega\|=\sqrt[2]{\omega_{1}^{2}+\omega_{2}^{2}}}\end{array}
  $$

- 我们目的是为了**找出一个分类效果好的超平面作为分类器。分类器的好坏的评定依据是分类间隔W=2d的大小，即分类间隔w越大，我们认为这个超平面的分类效果越好。此时，求解超平面的问题就变成了求解分类间隔W最大化的为题。**

#### 2.4 约束条件

- 在获得目标函数的形式之后，为了求解最大间隔，我们需要考虑下面的问题：
  - 如何判断超平面是否正确将样本点分类？
  - 怎样找到**支持向量**上的点？

- 上述问题就是在优化过程中遇到的限制和约束，在SVM算法中，如下解决：

- 平面上有两种点，分别如下进行标记：

  - 红颜色的圆点标记为1，我们人为规定其为正样本；
  - 蓝颜色的五角星标记为-1，我们人为规定其为负样本。
  - 对样本点$x_i$加上标签$y_i$:

  $$
  y_i = +1:红色点 or -1:蓝色点
  $$

- 超平面若成功分类，则会满足下列不等式：

$$
\left\{\begin{array}{c}{\boldsymbol{\omega}^{T} x_{i}+\gamma>0 \quad y_{i}=1} \\ {\boldsymbol{\omega}^{T} x_{i}+\gamma<0 \quad y_{i}=-1}\end{array}\right.
$$

- 假设决策面正好处于间隔区域的中轴线上，并且相应的支持向量对应的样本点到决策面的距离为d，那么公式进一步写成：

$$
\left\{\begin{array}{ll}{\frac{\boldsymbol{\omega}^{T} x_{i}+\gamma}{\|\boldsymbol{\omega}\|} \geq \mathrm{d}} & {\forall y_{i}=1} \\ {\frac{\boldsymbol{\omega}^{T} x_{i}+\gamma}{\|\boldsymbol{\omega}\|} \leq-\mathrm{d}} & {\forall y_{i}=-1}\end{array}\right.
$$

- 上述不等式的含义为：对于所有分类标签为1和-1样本点，它们到直线的距离都大于等于d**(支持向量上的样本点到超平面的距离)**。公式两边都除以d，就可以得到：

$$
\left\{\begin{array}{ll}{\boldsymbol{\omega}_{d}^{T} x_{i}+\gamma_{d} \geq 1} & {\forall y_{i}=1} \\ {\boldsymbol{\omega}_{d}^{T} x_{i}+\gamma_{d} \leq-1} & {\forall y_{i}=-1}\end{array}\right.
$$

- 其中$w_d 、\gamma_d$如下：

$$
\omega_{d}=\frac{\omega}{\|\omega\| d}, \quad \gamma_{d}=\frac{\gamma}{\|\omega\| d}
$$

- 因为$||w||$和$d$都是标量。所以上述公式的两个矢量，依然描述一条直线的法向量和截距。

$$
\begin{array}{l}{\boldsymbol{\omega}_{d}^{T} \boldsymbol{x}+\gamma_{d}=0} \\ {\boldsymbol{\omega}^{T} \boldsymbol{x}+\gamma=0}\end{array}
$$

- 上述两个公式，都是描述一条直线，数学模型代表的意义是一样的。现在，让我们对$w_d$和$γ_d$重新起个名字，就叫它们$w$和$\gamma $。
- 因此，我们就可以说："对于存在分类间隔的两类样本点，我们一定可以找到一些超平面，使其对于所有的样本点均满足下面的条件："

$$
\left\{\begin{array}{ll}{\boldsymbol{\omega}^{T} x_{i}+\gamma \geq 1} & {\forall y_{i}=1} \\ {\boldsymbol{\omega}^{T} x_{i}+\gamma \leq-1} & {\forall y_{i}=-1}\end{array}\right.
$$

- 上述方程给出了SVM最优化问题的约束条件，之所以标记为1和-1，是方便将约束条件变为如下方程：

$$
y_{i}\left(\boldsymbol{\omega}^{T} x_{i}+\gamma\right) \geq 1  ：\forall x_{i}
$$

#### 2.5 线性SVM优化问题思路

- 首先已经得到了目标函数：

$$
d=\frac{\left|\boldsymbol{\omega}^{T} \boldsymbol{x}+\gamma\right|}{\|\boldsymbol{\omega}\|}
$$

- 我们的优化目标是是d最大化。我们已经说过，我们是**用支持向量上的样本点求解d的最大化**的问题的。那么支持向量上的样本点有什么特点呢？

$$
\left(\boldsymbol{\omega}^{T} x_{i}+\gamma\right) = 1  : \forall 支持向量上的点x_{i}
$$

- 正如2.4推导所示，上述方程成立。
- 有了该方程，便可简化间隔大小d：$d = \frac{1}{||w||}$
- 在求距离最大时，可等效为$min \frac{1}{2} {||w||}^2$
- 之所以如上等效，是为了在最优化过程中对目标函数求导比较方便，但这不会影响最后的结果，如下所示：我们需要将目标函数和约束条件放一起进行描述：

$$
\begin{array}{l}{\min \frac{1}{2}\|\omega\|^{2}} \\ {\text { s.t. } y_{i}\left(\omega^{T} x_{i}+b\right) \geq 1, i=1,2, \ldots, n}\end{array}
$$

- 这里n是样本点的总个数，缩写s.t.表示"Subject to"，是"服从某某条件"的意思。上述公式描述的是一个典型的不等式约束条件下的二次型函数优化问题，同时也是支持向量机的基本数学模型。

#### 2.6 求最优解

- 通常求解的最优化问题分为以下几类：

  - 无约束优化问题：

  $$
  min f(x)
  $$

  - 有等式约束的优化问题：

  $$
  \begin{array}{l}{\min f(x)} \\ {\text { s.t. } h_{i(x)}=0, \quad i=1,2, \ldots, n}\end{array}
  $$

  - 有不等式约束的问题：

  $$
  \begin{array}{l}{\min f(x)} \\ {\text { s.t. } g_{i(x)} \leq 0, \quad i=1,2, \ldots, n} \\ {h_{j(x)}=0, \quad j=1,2, \ldots, m}\end{array}
  $$

- 对于第(a)类的优化问题，尝试使用的方法就是**费马大定理(Fermat)**，即使用求取函数f(x)的导数，然后令其为零，可以求得候选最优值，再在这些候选值中验证；如果是凸函数，可以保证是最优解。这也就是我们高中经常使用的求函数的极值的方法。

- 对于第(b)类的优化问题，常常使用的方法就是**拉格朗日乘子法（Lagrange Multiplier)** ，即把等式约束h_i(x)用一个系数与f(x)写为一个式子，称为拉格朗日函数，而系数称为拉格朗日乘子。通过拉格朗日函数对各个变量求导，令其为零，可以求得候选值集合，然后验证求得最优值。

- 对于第(c)类的优化问题，常常使用的方法就是**KKT条件**。同样地，我们把所有的等式、不等式约束与f(x)写为一个式子，也叫拉格朗日函数，系数也称拉格朗日乘子，通过一些条件，可以求出最优值的**必要条件(即最优值可推出的理论)**，这个条件称为KKT条件。

- 很明显，SVM的最优化问题属于第三类，所以要先学习一下拉格朗日函数和KKT条件。

#### 2.7 拉格朗日函数

- 拉格朗日方程的目的：把约束条件放入目标函数中，从而将**有约束优化问题转换为无约束优化问题**。

- 对于使用拉格朗日获得的函数，使用求导的方法求解依然困难，于是需要对问题再进行一次转换，使用一个数学技巧——**拉格朗日对偶**

- 第一步：**将有约束的原始目标函数转换为无约束的新构造的拉格朗日目标函数**：

  - 得拉格朗日函数如下：

  $$
  \mathcal{L}(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2}\|\boldsymbol{w}\|^{2}-\sum_{i=1}^{n} \alpha_{i}\left(y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)-1\right)
  $$

  - 其中$a_i$是拉格朗日乘子，大于等于0，是构造新目标函数引入的变量。
  - 现在写一个新的目标函数：

  $$
  \theta(w)=\max _{\alpha_{i} \geq 0} \mathcal{L}(w, b, \alpha)
  $$

  - 对于上面的函数，若样本点不满足约束条件，即**在可行解区域之外**：$y_{i}\left(\boldsymbol{\omega}^{T} x_{i}+b\right)<1$
     	- 此时，将$a_i$设置为正无穷，显然此时$\theta(w)$也是正无穷。 
  - 对于上面的函数，若样本点满足约束条件，即**在可行解区域之内**：$y_{i}\left(\boldsymbol{\omega}^{T} x_{i}+b\right) \geq 1$
    - 此时，由于减去的后半部分是一个正数部分，所以最大值仍是$\frac{1}{2}{||w||}^2$
  - 将上述情况结合一下，得到：

  $$
  \theta(\boldsymbol{\omega})=\left\{\begin{array}{ll}{\frac{1}{2}\|\boldsymbol{\omega}\|^{2}} & {x \in 可行解区域内} \\ {+\infty} & {x \in 非可行解区域}\end{array}\right.
  $$

  - 使用拉格朗日函数的初衷就是：建立一个在可行解区域内与原目标函数相同，在可行解区域外函数值趋近于无穷大的新函数。
  - 又因为之前的目标就是为了求$min \frac{1}{2}{||w||}^2$，所以此时可转换为求：

  $$
  \min _{w, b} \theta(w)=\min _{w, b} \max _{\alpha_{i} \geq 0} \mathcal{L}(w, b, \alpha)=p^{*}
  $$

  - 这里用p*表示这个问题的最优值，且和最初的问题是等价的。观察上面新的目标函数，若先求有不等式约束的max值，这样比较麻烦。所以这时利用**拉格朗日函数的对偶性**，将最小和最大的位置交换一下，这样就变成了如下所示：

  $$
  \max _{\alpha_{i} \geq 0} \min _{w, b}  \mathcal{L}(w, b, \alpha)=d^{*}
  $$

  - 交换之后的新问题就变成了原始问题的对偶问题，这个新问题的最优值用$d^*$来表示。从形式可看出：$d^* <= p^*$。而我们需要知道当两者相等的时候，这才是我们想要的解。需要什么条件使其相等呢？
    - 首先必须满足这个优化问题是凸优化问题。
    - 其次需要满足KKT条件。
  - 凸优化问题的定义是：**求取最小值的目标函数为凸函数的一类优化问题。**目标函数是凸函数我们已经知道，这个优化问题又是求最小值。所以我们的最优化问题就是凸优化问题。
  - 下一步就是探讨是否满足KKT条件

#### 2.8 KKT条件

- 首先是KKT条件下的最优化模型的标准形式如下：

$$
\begin{aligned} \min f(\boldsymbol{x}) & \\ \text {s.t.} & h_{j}(\boldsymbol{x})=0, j=1,2, \cdots, p \\ & g_{k}(\boldsymbol{x}) \leq 0, k=1,2, \cdots, q \\ & \boldsymbol{x} \in X \subset \mathbb{R}^{n} \end{aligned}
$$

- KKT条件的全称是Karush-Kuhn-Tucker条件，KKT条件是说最优值条件必须满足以下条件：

  - 1. 经过拉格朗日函数处理之后的新目标函数$L(x,\alpha,\beta)$对x求导为0。
  - 2. $h(x)=0$
  - 3. $\alpha \times g(x) = 0$

- 针对我们之前的目标函数对上述条件进行分析：
  $$
  \begin{array}{l}{\min \frac{1}{2}\|\omega\|^{2}} \\ {\text { s.t. } y_{i}\left(\omega^{T} x_{i}+b\right) \geq 1, i=1,2, \ldots, n}\end{array}
  $$

- 首先看条件2，因为没有等式约束，所以满足条件2
- 对于条件1，先看一个简单的例子：设想我们的目标函数$z = f(x)$, x是向量, z取不同的值，相当于可以投影在x构成的平面（曲面）上，即成为等高线，如下图，目标函数是$f(x, y)$，这里x是标量，虚线是等高线，现在假设我们的约束$g(x)=0$，x是向量，在x构成的平面或者曲面上是一条曲线，假设g(x)与等高线相交，交点就是同时满足等式约束条件和目标函数的可行域的值，但肯定不是最优值，因为相交意味着肯定还存在其它的等高线在该条等高线的内部或者外部，使得新的等高线与目标函数的交点的值更大或者更小，只有到等高线与目标函数的曲线相切的时候，可能取得最优值。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/12-1.jpg)

- 如上图所示，我们可以联系SVM中的图，可行域是$y_{i}\left(\omega^{T} x_{i}+b\right) \geq 1$，在两条边界往外扩张（箭头向外），而我们需要求的目标函数是min最小值，它是指向内方向的箭头，在两者于边界处相交时，取得在可行域之内的最小值。此时，上例中的$f(x)$与$g(x)$梯度大小一致，只不过两者方向不同，即$\frac{\partial \mathcal{f(x)}}{\partial \boldsymbol{x}}-\frac{\partial {\mathcal{\alpha \times g(x)}}}{\partial \boldsymbol{x}} = 0 $，所以在新目标函数中，我们需满足的KKT条件:
  $$
  \frac{\partial \mathcal{L}}{\partial \boldsymbol{w}} = 0\\\frac{\partial \mathcal{L}}{\partial \boldsymbol{b}} = 0
  $$
  也可将L分为前后两部分，然后可证得上述成立。

- 条件3，g(x)即不等式约束条件，由以上证明条件得知，在取得极小值时，在边界取得，所以g(x) = 0。

- 有关拉格朗日乘子法及KKT条件，下面的博文可以参考：[拉格朗日乘子法](https://blog.csdn.net/xianlingmao/article/details/7919597)，[KKT](https://www.cnblogs.com/liaohuiqiang/p/7805954.html)

#### 2.9 对偶问题求解

- 凸优化问题和KKT都满足了，问题转换成了对偶问题。而**求解这个对偶学习问题，可以分为三个步骤：首先要让L(w,b,α)关于w和b最小化，然后求对α的极大，最后利用SMO算法求解对偶问题中的拉格朗日乘子**。
- **第一步**：求L关于w、b的最小值

$$
\begin{array}{l}{\max _{\alpha_{i} \geq 0} \min _{w, b} \mathcal{L}(w, b, \alpha)=d^{*}} \\ {\mathcal{L}(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2}\|\boldsymbol{w}\|^{2}-\sum_{i=1}^{n} \alpha_{i}\left(y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)-1\right)}\end{array}
$$

   -  首先固定α，要让L(w,b,α)关于w和b最小化，我们分别对w和b偏导数，令其等于0，即：
     $$
     \begin{aligned} \frac{\partial \mathcal{L}}{\partial \boldsymbol{w}} &=0 \Rightarrow \boldsymbol{w}=\sum_{i=1}^{n} \alpha_{i} y_{i} \boldsymbol{x}_{i} \\ \frac{\partial \mathcal{L}}{\partial b} &=0 \Rightarrow \sum_{i=1}^{n} \alpha_{i} y_{i}=0 \end{aligned}
     $$

- 将上面结果带回至L中：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/9-5.jpg)

- 从上面的最后一个式子，我们可以看出，此时的L(w,b,α)函数只含有一个变量，即α。

- **第二步**：现在内侧的最小值求解完成，我们求解外侧的最大值，从上面的式子得到：

$$
\begin{array}{l}{\max _{\boldsymbol{\alpha}} \sum_{i=1}^{n} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}} \\ {\text { s.t. } \alpha_{i} \geq 0, i=1,2, \cdots, n} \\ {\quad \sum_{i=1}^{n} \alpha_{i} y_{i}=0}\end{array}
$$

- 现在优化问题变成了如上的形式。
- 对于这个问题，有更高效的优化算法，即**序列最小优化（SMO）算法**。我们通过这个优化算法能得到α，再根据α，我们就可以求解出w和b，进而求得我们最初的目的：找到超平面，即"决策平面"。
- 此处不对SMO算法进行深入。

### 3. SMO算法

#### 3.1 smo算法简介

- 1996年，John Platt发布了一个称为SMO的强大算法，用于训练SVM。SMO表示序列最小化(Sequential Minimal Optimizaion)。Platt的SMO算法是**将大优化问题分解为多个小优化问题来求解**的。这些小优化问题往往很容易求解，并且对它们进行顺序求解的结果与将它们作为整体来求解的结果完全一致的。在结果完全相同的同时，SMO算法的求解时间短很多。
- **SMO算法的目标**是求出一系列alpha和b，一旦求出了这些alpha，就很容易计算出权重向量w并得到分隔超平面。
- **SMO算法的工作原理**是：每次循环中选择两个alpha进行优化处理。一旦找到了一对合适的alpha，那么就增大其中一个同时减小另一个。这里所谓的"合适"就是指两个alpha必须符合以下两个条件，条件之一就是两个alpha必须要在间隔边界之外，而且第二个条件则是这两个alpha还没有进行过区间化处理或者不在边界上。
- 最后便可以根据求解出的α，计算出w和b，从而得到分类超平面函数。

$$
\begin{array}{l}{w^{*}=\sum_{i=1}^{n} \alpha_{i} y_{i} x_{i}} \\ {b^{*}=-\frac{\max _{i: y_{i}=-1} w^{* T} x_{i}+\min _{i: y_{i}=1} w^{* T} x_{i}}{2}}\end{array}
$$

- 在对新的点进行预测时，实际上就是将数据点x*代入分类函$f(x)=w^Tx+b$中，若f(x)>0，则为正类，f(x)<0，则为负类。

$$
\begin{aligned} f(x) &=\left(\sum_{i=1}^{n} \alpha_{i} y_{i} x_{i}\right)^{T} x+b \\ &=\sum_{i=1}^{n} \alpha_{i} y_{i}\left\langle x_{i}, x\right\rangle+ b \end{aligned}
$$

### 4. 核函数

- 上述的超平面只能解决线性可分的问题，对于线性不可分的问题，例如：异或问题，我们需要使用核函数将其进行推广。

- 一般地，解决线性不可分问题时，常常采用**映射**的方式，**将低维原始空间映射到高维特征空间，使得数据集在高维空间中变得线性可分，从而再使用线性学习器分类**。

- 如果原始空间为有限维，即属性数有限，那么总是存在一个高维特征空间使得样本线性可分。若∅代表一个映射，令$\phi(x)$表示将x映射后的特征向量，则在特征空间中的划分函数变为：

  $f(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T}} \phi(\boldsymbol{x})+b$

- 按照同样的方法，先写出新目标函数的拉格朗日函数，接着写出其对偶问题，求L关于w和b的极大，最后运用SOM求解α。可以得出：

- 原对偶问题变为：

$$
\begin{aligned} \max _{\alpha} & \sum_{i=1}^{n} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j}\left\langle\phi\left(x_{i}\right), \phi\left(x_{j}\right)\right\rangle \\ \text { s.t. } & \alpha_{i} \geq 0, i=1, \ldots, n \\ & \sum_{i=1}^{n} \alpha_{i} y_{i}=0 \end{aligned}
$$

- 原分类函数变为：

$$
f(x)=\sum_{i=1}^{n} \alpha_{i} y_{i}\left\langle\phi\left(x_{i}\right), \phi(x)\right\rangle+ b
$$

- 求解的过程中，只涉及到了高维特征空间中的内积运算，由于特征空间的维数可能会非常大，
  - 例如：若原始空间为二维，映射后的特征空间为5维，若原始空间为三维，映射后的特征空间将是19维，之后甚至可能出现无穷维，根本无法进行内积运算了，此时便引出了**核函数**（Kernel）的概念。

- **核是一个函数K，对所有的$x,z\in Y,$ 满足$K(x,z)=<\phi(x),\phi(z)>$**，这里的$\phi$是从Y到内积特征空间F的映射
- 因此，核函数可以直接计算隐式映射到高维特征空间后的向量内积，而不需要显式地写出映射后的结果，它虽然完成了将特征从低维到高维的转换，但最终却是在低维空间中完成向量内积计算，与高维特征空间中的计算等效**（低维计算，高维表现）**，从而避免了直接在高维空间无法计算的问题。引入核函数后，原来的对偶问题与分类函数则变为：

$$
\begin{aligned} \max _{\alpha} & \sum_{i=1}^{n} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} K\left(x_{i}, x_{j}\right) \\ \text { s.t. } & \alpha_{i} \geq 0, i=1, \ldots, n \\ & \sum_{i=1}^{n} \alpha_{i} y_{i}=0 \end{aligned}
$$

- 分类函数：

$$
f(x)=\sum_{i=1}^{n} \alpha_{i} y_{i} K\left(x_{i}, x\right)+b
$$

- 因此，在线性不可分问题中，核函数的选择成了支持向量机的最大变数，若选择了不合适的核函数，则意味着将样本映射到了一个不合适的特征空间，则极可能导致性能不佳。同时，核函数需要满足以下这个必要条件：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/12-8.jpg)

### 5. 软间隔支持向量机

- 当数据线性可分时，直接使用最大间隔的超平面划分；当数据线性不可分时，则通过核函数将数据映射到高维特征空间，使之线性可分。
- 然而在现实问题中，对于某些情形还是很难处理，例如数据中有**噪声**的情形，噪声数据（**outlier**）本身就偏离了正常位置，但是在前面的SVM模型中，我们要求所有的样本数据都必须满足约束，如果不要这些噪声数据还好，当加入这些outlier后导致划分超平面被挤歪了，如下图所示，对支持向量机的泛化性能造成很大的影响：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/12-9.jpg)

- 为了解决这一问题，我们需要允许某一些数据点不满足约束，即可以在一定程度上偏移超平面，同时使得不满足约束的数据点尽可能少，这便引出了**“软间隔”支持向量机**的概念
  - 允许某些数据点不满足约束$y(w'x+b)≥1$；
  - 同时又使得不满足约束的样本尽可能少。
- 这样优化目标变为：

$$
\begin{array}{l}{\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \ell_{0 / 1}\left(y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)-1\right)} \\ {\ell_{0 / 1}(z)=\left\{\begin{array}{ll}{1,} & {\text { if } z<0 ;} \\ {0,} & {\text { otherwise }}\end{array}\right.}\end{array}
$$

- 其中$l_{0/1}$为损失函数，虽然表示效果最好，但是数学性质不佳。因此常用其它函数作为“替代损失函数”。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/12-10.jpg)

- 支持向量机中的损失函数为**hinge损失**，引入**“松弛变量”**，目标函数与约束条件可以写为：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/12-11.jpg)

- 其中C为一个参数，控制着目标函数与新引入正则项之间的权重，这样显然每个样本数据都有一个对应的松弛变量，用以表示该样本不满足约束的程度，将新的目标函数转化为拉格朗日函数得到：

$$
\mathcal{L}(w, b, \xi, \alpha, r)=\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{n} \xi_{i}-\sum_{i=1}^{n} \alpha_{i}\left(y_{i}\left(w^{T} x_{i}+b\right)-1+\xi_{i}\right)-\sum_{i=1}^{n} r_{i} \xi_{i}
$$

- 按照与之前相同的方法，先让L求关于w，b以及松弛变量的极小，再使用SMO求出α，有：

$$
\begin{array}{l}{\frac{\partial L}{\partial w}=0 \Rightarrow w=\sum_{i=1}^{n} \alpha_{i} y_{i} x_{i}} \\ {\frac{\partial L}{\partial b}=0 \Rightarrow \sum_{i=1}^{n} \alpha_{i} y_{i}=0} \\ {\frac{\partial L}{\partial \xi_{i}}=0 \Rightarrow C-\alpha_{i}-r_{i}=0, \quad i=1, \ldots, n}\end{array}
$$

- 将w代入L化简，便得到其对偶问题：

$$
\begin{aligned} \max _{\alpha} & \sum_{i=1}^{n} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j} \quad \text { 与原对偶问题相同 } \\ \text { s.t. } & 0 \leq \alpha_{i} \leq C, i=1, \ldots, n \\ & \sum_{i=1}^{n} \alpha_{i} y_{i}=0 \end{aligned}
$$

- 将“软间隔”下产生的对偶问题与原对偶问题对比可以发现：**新的对偶问题只是约束条件中的α多出了一个上限C，其它的完全相同，因此在引入核函数处理线性不可分问题时，便能使用与“硬间隔”支持向量机完全相同的方法。**