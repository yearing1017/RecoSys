### 1. 逻辑回归

- Logistic回归是分类方法，它利用的是Sigmoid函数阈值在[0,1]这个特性。**Logistic回归进行分类的主要思想是：根据现有数据对分类边界线建立回归公式，以此进行分类。**

- 假设现在有一些数据点，我们利用一条直线对这些点进行拟合(该线称为最佳拟合直线)，这个拟合过程就称作为回归，如下图所示：

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/4-1.jpg)

- 先看一下Sigmoid函数 ，我们也可以称它为Logistic函数。它的公式如下：
  $$
  \begin{aligned} h_{\theta}(x)=& g\left(\theta^{T} x\right) \\ \mathrm{z}=\left[\begin{array}{cccc}{\theta_{0}} & {\theta_{1}} & {\dots} & {\theta_{n}}\end{array}\right]\left[\begin{array}{c}{x_{0}} \\ {x_{0}} \\ {x_{1}} \\ {\vdots} \\ {x_{n}}\end{array}\right]=\theta^{T} x \\ & g(z)=\frac{1}{1+e^{-z}} \end{aligned}
  $$
  整合成一个公式，就变成了如下公式：
  $$
  h_{\theta}(x)=g\left(\theta^{T} x\right)=\frac{1}{1+e^{-\theta^{T} x}}
  $$

- 函数图像如下：

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/4-2.jpg)

- z是一个矩阵，θ是参数列向量(要求解的)，x是样本列向量(给定的数据集)。$θ^T$表示θ的转置。g(z)函数实现了任意实数到[0,1]的映射，这样我们的数据集([x0,x1,...,xn])，不管是大于1或者小于0，都可以映射到[0,1]区间进行分类。$h_{\theta}(x)$给出了输出为1的概率。比如当$h_{\theta}(x)$=0.7，那么说明有70%的概率输出为1。输出为0的概率是输出为1的补集，也就是30%。

- 如果我们有合适的参数列向量$θ([θ_0,θ_1,...θ_n]^T)$，以及样本列向量$(x[x_0,x_1,...,x_n])$，那么我们对样本x分类就可以通过上述公式计算出一个概率，如果这个概率大于0.5，我们就可以说样本是正样本，否则样本是负样本。

- **如何得到合适的参数向量θ?**

  - 根据sigmoid函数的特性，我们可以做出如下的假设：
    $$
    \begin{aligned} P(y=1 | x ; \theta) &=h_{\theta}(x) \\ P(y=0 | x ; \theta) &=1-h_{\theta}(x) \end{aligned}
    $$

  - 上述公式即为在已知样本x和参数θ的情况下，样本x属性正样本(y=1)和负样本(y=0)的条件概率。

  - 理想状态下，根据上述公式，求出各个点的概率均为1，也就是完全分类都正确。但是考虑到实际情况，样本点的概率越接近于1，其分类效果越好。比如一个样本属于正样本的概率为0.51，那么我们就可以说明这个样本属于正样本。另一个样本属于正样本的概率为0.99，那么我们也可以说明这个样本属于正样本。但是显然，第二个样本概率更高，更具说服力。我们可以把上述两个概率公式合二为一：
    $$
    \operatorname{cost}\left(h_{\theta}(\mathrm{x}), \mathrm{y}\right)=h_{\theta}(x)^{y}\left(1-h_{\theta}(x)\right)^{(1-y)}
    $$

  - 合并出来的Cost，我们称之为代价函数(Cost Function)。当y等于1时，(1-y)项(第二项)为0；当y等于0时，y项(第一项)为0。为了简化问题，我们对整个表达式求对数，(将指数问题对数化是处理数学问题常见的方法)：
    $$
    \operatorname{cost}\left(h_{\theta}(\mathrm{x}), \mathrm{y}\right)=\mathrm{y} \log h_{\theta}(x)+(1-y) \log \left(1-h_{\theta}(x)\right)
    $$

  - 这个代价函数，是对于一个样本而言的。给定一个样本，我们就可以通过这个代价函数求出，样本所属类别的概率，而这个概率越大越好，所以也就是求解这个代价函数的最大值。既然概率出来了，那么最大似然估计也该出场了。假定样本与样本之间相互独立，那么整个样本集生成的概率即为所有样本生成概率的乘积，再将公式对数化，便可得到如下公式：
    $$
    \mathrm{J}(\theta)=\sum_{i=1}^{m}\left[y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
    $$

  - 其中，m为样本的总数，$y^{(i)}$表示第i个样本的类别，$x^{(i)}$表示第i个样本，需要注意的是θ是多维向量，$x^{(i)}$也是多维向量。

- **综上所述，满足J(θ)的最大的θ值即是我们需要求解的模型。**
  - 注：求解使J(θ)最大的θ值呢？因为是求最大值，所以我们需要使用梯度上升算法。如果面对的问题是求解使J(θ)最小的θ值，那么我们就需要使用梯度下降算法。面对我们这个问题，如果使J(θ) := -J(θ)，那么问题就从求极大值转换成求极小值了，使用的算法就从梯度上升算法变成了梯度下降算法。详见之前的一篇学习笔记：[逻辑回归]([https://yearing1017.site/2019/04/27/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-2-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/](https://yearing1017.site/2019/04/27/神经网络与深度学习-2-逻辑回归/))

### 2. 梯度上升算法

- 先看个简单的求极大值的例子：
  $$
  f(x)=-x^{2}+4 x
  $$

  - 求极值，先求函数的导数：
    $$
    f^{\prime}(x)=-2 x+4
    $$
    令导数为0，可求出x=2即取得函数f(x)的极大值。极大值等于f(2)=4

- 但是真实环境中的函数不会像上面这么简单，就算求出了函数的导数，也很难精确计算出函数的极值。此时我们就可以用迭代的方法来做。就像爬坡一样，一点一点逼近极值。这种寻找最佳拟合参数的方法，就是最优化算法。爬坡这个动作用数学公式表达即为：
  $$
  x_{i+1}=x_{i}+\alpha \frac{\partial f\left(x_{i}\right)}{x_{i}}
  $$

  - 其中，α为步长，也就是学习速率，控制更新的幅度，效果如下图所示：

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/4-3.jpg)

- 比如从(0,0)开始，迭代路径就是1->2->3->4->...->n，直到求出的x为函数极大值的近似值，停止迭代。我们可以编写Python3代码，来实现这一过程：

  ```python
  # -*- coding:UTF-8 -*-
  """
  函数说明:梯度上升算法测试函数
  求函数f(x) = -x^2 + 4x的极大值
  """
  def Gradient_Ascent_test():
      def f_prime(x_old):                                    #f(x)的导数
          return -2 * x_old + 4
      x_old = -1                                            #初始值，给一个小于x_new的值
      x_new = 0                                            #梯度上升算法初始值，即从(0,0)开始
      alpha = 0.01                                        #步长，也就是学习速率，控制更新的幅度
      presision = 0.00000001                                #精度，也就是更新阈值
      while abs(x_new - x_old) > presision:									#没达到最值的接近处，就一直迭代
          x_old = x_new
          x_new = x_old + alpha * f_prime(x_old)            #上面提到的公式
      print(x_new)                                        #打印最终求解的极值近似值
  
  if __name__ == '__main__':
      Gradient_Ascent_test()
  ```

  结果如下：

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/4-4.jpg)

- 结果很显然，已经非常接近我们的真实极值2了。这一过程，就是梯度上升算法。那么同理，J(θ)这个函数的极值，也可以这么求解。公式可以这么写：
  $$
  \theta_{j} :=\theta_{j}+\alpha \frac{\partial J(\theta)}{\theta_{j}}
  $$
  **那么，现在只要求出J(θ)的偏导，就可以利用梯度上升算法，求解J(θ)的极大值**。

- 开始求解J(θ)对θ的偏导，求解如下：
  $$
  \frac{\partial}{\theta_{j}} J(\theta)=\frac{\partial J(\theta)}{\partial g\left(\theta^{T} x\right)} * \frac{\partial g\left(\theta^{T} x\right)}{\partial \theta^{T} x} * \frac{\partial \theta^{T} x}{\partial \theta_{j}}
  $$
  其中：
  $$
  \frac{\partial J(\theta)}{\partial g\left(\theta^{T} x\right)}=\mathrm{y} * \frac{1}{g\left(\theta^{T} x\right)}+(y-1) * \frac{1}{1-g\left(\theta^{T} x\right)}
  $$
  再有：
  $$
  \begin{aligned} g^{\prime}(z)=& \frac{d}{d z} \frac{1}{1+e^{-z}}=\frac{1}{\left(1+e^{-z}\right)^{2}}\left(e^{-z}\right) \\ &=\frac{1}{\left(1+e^{-z}\right)}\left(1-\frac{1}{\left(1+e^{-z}\right)}\right)=g(z)(1-g(z)) \end{aligned}
  $$
  可得：
  $$
  \frac{\partial g\left(\theta^{T} x\right)}{\partial \theta^{T} x}=g\left(\theta^{T} x\right)\left(1-g\left(\theta^{T} x\right)\right)
  $$
  接下来，就剩下第三部分：
  $$
  \frac{\partial \theta^{T} x}{\theta_{j}}=\frac{\partial J\left(\theta_{1} x_{1}+\theta_{2} x_{2}+\cdots \theta_{n} x_{n}\right)}{\partial \theta_{j}}=x_{j}
  $$
  综上：
  $$
  \frac{\partial}{\theta_{j}} J(\theta)=\left(y-h_{\theta}(x)\right) x_{j}
  $$
  得梯度上升迭代公式：
  $$
  \theta_{j} :=\theta_{j}+\alpha \sum_{i=1}^{m}\left(y^{(i)}-h_{\theta}\left(x^{(i)}\right)\right) x_{j}^{(i)}
  $$

### 3. 数据实战部分（基于Python3）

- 数据集下载：[传送门](https://github.com/yearing1017/Machine-Learning/blob/master/Logistic/testSet.txt)

- 看一下数据集：

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/4-5.jpg)

- 这个数据有两维特征，因此可以将数据在一个二维平面上展示出来。我们可以将第一列数据(X1)看作x轴上的值，第二列数据(X2)看作y轴上的值。而最后一列数据即为分类标签。根据标签的不同，对这些点进行分类。

- 使用matplotlib来绘制数据集的分布情况：

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  """
  函数说明:绘制数据分布
  Returns:
      dataMat - 数据列表
      labelMat - 标签列表
  """
  def loadDataSet():
  	dataMat = []			   #数据列表
  	labelMat = []			   #标签列表
  	fr = open('testSet.txt')   #将数据集与py文件放在同一目录下，即可打开
  	for line in fr.readlines():			#逐行读取
  		lineArr = line.strip().split()  #strip函数去除每行首尾的空格，split函数按空格划分每行并放进列表。
  										#例如数据集的第一行返回：['-0.017612','14.053064','0']
  		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) #添加数据
  		##[1.0, -0.017612, 14.053064], [1.0, -1.395634, 4.662541], [1.0, -0.752157, 6.53862]
  		labelMat.append(int(lineArr[2]))#添加标签
  	fr.close() #关闭文件
  	return dataMat,labelMat
  	#print(dataMat)
  
  """
  函数说明:绘制数据集
  """
  def plotDataSet():
  	dataMat,labelMat = loadDataSet()
  	dataArr = np.array(dataMat)#dataMat转换为numpy的array数组，打印前两组如下：
  	#[ 1.0000000e+00 -1.7612000e-02  1.4053064e+01]
   	#[ 1.0000000e+00 -1.3956340e+00  4.6625410e+00]
  	n = np.shape(dataMat)[0]#m = dataArr.shape[0] 一样的效果，都可获知数据个数
  	xcord1 = []; ycord1 = [] #正样本
  	xcord2 = []; ycord2 = [] #负样本
  	for i in range(n):				#循环遍历每行数据的便签
  		if int(labelMat[i])==1:		#如果标签是1,则划为正数据
  			xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
  			#将每行数据的第二个数据放入x坐标，第三个数据放入y坐标
  		else:
  			xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
  	fig = plt.figure() #定义一个窗口
  	ax = fig.add_subplot(111) #添加subplot，111表示1x1网格，第一子图
  	ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5) #绘制正样本
  	#scatter用来绘制散点图，s代表标量，默认20，marker=’s‘代表点为正方形点
  	ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)             #绘制负样本
  	plt.title('DataSet')                                                #绘制title
  	plt.xlabel('x'); plt.ylabel('y')                                    #绘制label
  	plt.show()             
  
  if __name__ == '__main__':
  	plotDataSet()
  	#loadDataSet()
  ```

- 运行结果：

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/4-6.png)
  - 从上图可以看出数据的分布情况。假设Sigmoid函数的输入记为z，那么$z=w_0X_0 + w_1X_1 + w_2X_2$，即可将数据分割开。其中，$X_0$为全是1的向量，$X_1$为数据集的第一列数据，$X_2$为数据集的第二列数据。令z=0，则$0 = w_0X0+w_1X1+w_2X2$，其中$X_0$ =1，所以，$,X2 = （-w_0-w_1X1）/w_2$，这个方程即所需求得方程，未知的参数为$w_0$，$w_1$，$w_2$，也就是我们需要求的回归系数(最优参数)。

- 梯度上升迭代公式：

$$
\theta_{j} :=\theta_{j}+\alpha \sum_{i=1}^{m}\left(y^{(i)}-h_{\theta}\left(x^{(i)}\right)\right) x_{j}^{(i)}
$$

- 转换为矢量：

$$
\theta :=\theta+\alpha X^{T}(\overrightarrow{\mathrm{y}}-g(X \theta))
$$

- 代码如下：

```python
import matplotlib.pyplot as plt
import numpy as np
"""
函数说明:绘制数据分布
Returns:
    dataMat - 数据列表
    labelMat - 标签列表
"""
def loadDataSet():
	dataMat = []			   #数据列表
	labelMat = []			   #标签列表
	fr = open('testSet.txt')   #将数据集与py文件放在同一目录下，即可打开
	for line in fr.readlines():			#逐行读取
		lineArr = line.strip().split()  #strip函数去除每行首尾的空格，split函数按空格划分每行并放进列表。
										#例如数据集的第一行返回：['-0.017612','14.053064','0']
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) #添加数据
		##[1.0, -0.017612, 14.053064], [1.0, -1.395634, 4.662541], [1.0, -0.752157, 6.53862]
		labelMat.append(int(lineArr[2]))#添加标签
	fr.close() #关闭文件
	return dataMat,labelMat
	#print(dataMat)

"""
函数说明:绘制数据集
"""
def plotDataSet():
	dataMat,labelMat = loadDataSet()
	dataArr = np.array(dataMat)#dataMat转换为numpy的array数组，打印前两组如下：
	#[ 1.0000000e+00 -1.7612000e-02  1.4053064e+01]
 	#[ 1.0000000e+00 -1.3956340e+00  4.6625410e+00]
	n = np.shape(dataMat)[0]#m = dataArr.shape[0] 一样的效果，都可获知数据个数
	xcord1 = []; ycord1 = [] #正样本
	xcord2 = []; ycord2 = [] #负样本
	for i in range(n):				#循环遍历每行数据的便签
		if int(labelMat[i])==1:		#如果标签是1,则划为正数据
			xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
			#将每行数据的第二个数据放入x坐标，第三个数据放入y坐标
		else:
			xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
	fig = plt.figure() #定义一个窗口
	ax = fig.add_subplot(111) #添加subplot，111表示1x1网格，第一子图
	ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5) #绘制正样本
	#scatter用来绘制散点图，s代表标量，默认20，marker=’s‘代表点为正方形点
	ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)             #绘制负样本
	plt.title('DataSet')                                                #绘制title
	plt.xlabel('x'); plt.ylabel('y')                                    #绘制label
	plt.show()             

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


"""
函数说明:梯度上升算法
Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
"""
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)        		   #转换成numpy的mat,即将其转换为矩阵
    labelMat = np.mat(classLabels).transpose()   #转换成numpy的mat,并进行矩阵的转置
    m, n = np.shape(dataMatrix)                  #返回矩阵dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                              #最大迭代次数
    weights = np.ones((n,1))                     #创建一个n行1列的向量weights，元素全是1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)        #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()                				 #将矩阵转换为数组，返回权重数组


if __name__ == '__main__':
	#plotDataSet()
	dataMat, labelMat = loadDataSet()           
	print(gradAscent(dataMat, labelMat))
```

- 运行结果如下，求出三个参数：

```
[[ 4.12414349]
 [ 0.48007329]
 [-0.6168482 ]]
[Finished in 0.8s]
```

- 绘制决策边界线

```python
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()                                 #加载数据集
    dataArr = np.array(dataMat)                                       #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                          #数据个数
    xcord1 = []; ycord1 = []                                          #正样本
    xcord2 = []; ycord2 = []                                          #负样本
    for i in range(n):                                                #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                  #绘制label
    plt.show()  
```

- 绘制如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/4-7.png)

### 4. 总结

- 不太理解的地方：
  - 为什么令Sigmoid函数=0
  - 矢量转换怎么转换的