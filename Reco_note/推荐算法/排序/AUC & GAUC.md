###背景

给出一个模型α和一组测试数据，如图A，α对测试数据的预测概率值如图B，如果我们使用几个不同的阈值来得到分类结果，如图c，同样的模型，在不同阈值下，分类正确的正负样本数不同

- “假正”指的是本来是负样本，但被预测为正样本了

- “真正”就是本来是正样本，正确地预测为正样本了

所以对于C(1)，假正为2，真正为5

所以对于C(2)，假正为1，真正为4

所以对于C(3)，假正为0，真正为3

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/2023/0321/1.jpg" style="zoom: 50%;" />

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/2023/0321/2.jpg" style="zoom:50%;" />

### 图解ROC

先给所有的样本预测值进行排序，然后用所有的阈值（0.9→0.1）来分割这些样本的时候时，得到的“假正”和“真正”的变化分别是

- 假正：0 0 0 1 2 3 4 5 

- 真正：1 2 3 4 5 5 5 5

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/2023/0321/3.jpg" style="zoom:50%;" />

将数据放到坐标轴中，横坐标为**假正数**、纵坐标为**真正数**，得到ROC曲线

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/2023/0321/4.jpg" style="zoom:50%;" />

AUC的含义是Area Under Curve，也就是曲线下的面积，哪个曲线呢，就是ROC曲线咯。所以AUC其实就是ROC面积了。因为上面的**横纵坐标含义是假正数，纵坐标为真正数**，我们先把它分别转化为**假正率**和**真正率**，然后标记出曲线下的图形，如下，面积为0.88

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/2023/0321/5.jpg" style="zoom:50%;" />

### AUC含义

AUC，本质上是一个积分的过程呢，**积分的区域就是正样本和负样本的交叉区域，积分的内容是每个正样本在多少个负样本的前面**，比如下图，有5个正样本，因此横坐标的取值范围是[1,2,...,5 ]，在他们后面，分别有5、5、5、4、3个负样本，所以：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/2023/0321/6.jpg" style="zoom: 67%;" />
$$
AUC = \frac{3+4+5+5+5}{5 \times 5} = 0.88
$$
所以，当取一个极端情况的时候，比如所有正样本都在所有负样本前面的时候，如下图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/2023/0321/7.jpg" style="zoom:50%;" />
$$
AUC = \frac{5+5+5+5+5}{5 \times 5} = 1
$$
对应的ROC曲线，已经成这样了：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/2023/0321/8.jpg" style="zoom:50%;" />
