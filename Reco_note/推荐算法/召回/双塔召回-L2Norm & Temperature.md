### L2-Norm

#### 欧式距离、余弦距离、L2-Norm

在n维空间中，有两个向量分别是**x**与**y**

欧式距离（全面衡量向量的差异性，既考虑方向，也考虑尺度。其结果的范围不固定，受到向量长度以及向量维度的影响）
$$
|x-y|=\sqrt{(x-y) \cdot(x-y)}=\|x-y\|_2
$$

$$

$$

余弦距离（衡量两向量**x**，**y**方向的一致性。其结果的范围固定，与向量的长度无关，与向量的维度无关）
$$
1- cosine(x,y) = 1- \frac{x.y}{|x|.|y|}
$$


如果向量**x**，**y**进行了L2-Norm，则有：
$$
|x| = |y| = ||x||_{2} = ||y||_{2} = 1
$$
此时：
$$
\begin{aligned}
& |x-y|=\sqrt{(x-y) \cdot(x-y)}=\sqrt{x \cdot x+y \cdot y-2 x \cdot y} \\
& =\sqrt{|x|^2+|y|^2-2|x||y| \cos <x, y>}=\sqrt{2(1-\cos <x, y>)}
\end{aligned}
$$
即：优化欧式距离**等效于**优化余弦距离

例如：假设二维空间两个点， $A(x_1,y_1), B(x_2,y2)$

归一化为单位向量：
$$
A\left(\frac{x_1}{\sqrt{x_1^2+y_1^2}}, \frac{y_1}{\sqrt{x_1^2+y_1^2}}\right), B\left(\frac{x_2}{\sqrt{x_2^2+y_2^2}}, \frac{y_2}{\sqrt{x_2^2+y_2^2}}\right)
$$
余弦相似度，由于归一化后，向量模长为1，分母是1，省略了
$$
\cos =\frac{x_1}{\sqrt{x_1^2+y_1^2}} \times \frac{x_2}{\sqrt{x_2^2+y_2^2}}+\frac{y_1}{\sqrt{x_1^2+y_1^2}} \times \frac{y_2}{\sqrt{x_2^2+y_2^2}}
$$
欧式距离就是：
$$
e u c=\sqrt{\left(\frac{x_1}{\sqrt{x_1^2+y_1^2}}-\frac{x_2}{\sqrt{x_2^2+y_2^2}}\right)^2+\left(\frac{y_1}{\sqrt{x_1^2+y_1^2}}-\frac{y_2}{\sqrt{x_2^2+y_2^2}}\right)^2} =\sqrt{2- 2\times cos}
$$


####双塔召回最后一层输出需要L2-Norm

向量化召回一般都会接ANN serving，HNSW就是一种常用的算法

> 在hnswlib项目（[https://github.com/nmslib/hnswlib](https://link.zhihu.com/?target=https%3A//github.com/nmslib/hnswlib)）中可以看到HNSW 算法在点乘距离的数据集上效果差。本质是因为点乘距离非度量空间，不满足三角不等式 ，距离比较没有传递性。更通俗的说内积不**保序**，假设有三个点ABC，点击意义下|A,B|<|A,C|，但是欧式距离下不一定有|A,B|<|A,C|，比如A=(100,0),B=(0,100),C=(101,0)

一般HNSW采用**欧式距离**构建检索图，而**归一化**则巧妙地**将双塔点积行为转化为了欧式距离**，总结：**双塔召回需要ANN，点积不保序一般使用欧式距离，双塔最上层归一化能将输入映射欧式空间，保证了训练检索的一致性，提高了效果**

#### 温度系数temperate

向量归一化之后，向量模长为1，余弦相似度即为两向量内积，所以取值范围为[-1,1]（cos的取值范围为[-1,1]），此时网络的最终输出logit的取值范围为[-1,1]，**logit要经过sigmoid之后，再进入交叉熵损失函数**

- 此时logit和交叉熵的关系，第一列为logit，第二列为交叉熵(**未加负号前**)

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/2023/0403/1.jpg" style="zoom:33%;" />

首先我们要认识到：**模型如果把正样本分对了，loss应该是极小值才对**。而由于**归一化点积值域必在[-1,1]，导致模型预估点击概率为1时模型loss仍然很大**，temperature的作用其实就是**放大logit，让模型容易学习**。

##### 以softmax激活函数

内积限制在[-1,1]，我们假设正样本预估logits是1，负样本都是-1，来10个负样本，这个计算出来的概率是0.4249，极端情况尚且如此，根本拟合不了样本：
$$
\frac{math.exp(1)}{(math.exp(1)+math.exp(-1)*10)} = 0.4249256576603398
$$
给logits统一除以一个超参T，比如0.2，这个概率就变成了0.99了
$$
\frac{math.exp(\frac{1}{0.2})}{(math.exp(\frac{1}{0.2})+math.exp(\frac{-1}{0.2})*10)} = 0.9995462067242036
$$

