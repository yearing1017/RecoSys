### 1. 前言

- 在《机器学习实战》这本书前面已经学习了KNN、决策树、朴素贝叶斯、逻辑回归、SVM5种分类器，各有优缺点，我们若将几种分类器组合起来，这种组合结果则被称为**集成方法(ensemble method)或者元算法(meta-algorithm)**。
- 使用集成方法时会有多种形式：可以是不同算法的集成，也可以是同一种算法在不同设置下的集成，还可以是数据集不同部分分配给不同分类器之后的集成。

### 2. 集成方法

- 集成方法通过组合多个学习器来完成学习任务。集成方法主要包括Bagging、Boosting两种方法，都是将已有的分类或回归方法通过一定方式组合起来，形成一个性能更将强大的分类器。

#### 2.1 Bagging

- 自举汇聚法（bootstrap aggregating），也称为bagging方法。Bagging对训练数据采用自举采样（boostrap sampling），即有放回地采样数据，主要思想：
  - **从原始样本集中抽取训练集。本轮采取Bootstraping的方法抽取n个训练样本**（有些样本可能会被多次抽到，有些则有可能会一次抽不到）。共进行k轮抽取，得到k个训练集（k个之间相互独立）。
  - **每次使用一个训练集得到一个模型，k个训练集得到k个模型**。（这里没有具体的分类或者回归方法，我们可以根据具体问题采用不同的分类或回归方法，如决策树、感知器）。
  - **对分类问题：将上步得到的k个模型采用投票的方式得到分类结果；对回归问题：计算上述模型的均值作为最后的结果。（所有模型的重要性相同）**

#### 2.2 Boosting

- Boosting和Bagging很相似。该方法的思路是采用**重赋权法（re-weighting）**迭代的训练基分类器，主要思想：
  - 每一轮的训练数据样本赋予一个权重，并且每一轮样本的权值分布依赖于上一轮的分类结果。
  - 基分类器之间采用序列式的线性加权方式进行组合。

#### 2.3 两者的区别

- 样本选择上：
  - Bagging：训练集是在原始集中有放回选取的，从原始集中选出的各轮训练集之间是独立的。
  - Boosting：每一轮的训练集不变，只是训练集中每个样例在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整。
- 样例权重：
  - Bagging：采用均匀取样，每个样例的权重相等。
  - Boosting：根据错误率不断调整样例的权值，错误率越大则其权重越大。
- 预测函数：
  - Bagging：所有预测函数的权重相等。
  - Boosting：每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重。
- 并行计算：
  - Bagging：各个预测函数都可并行生成。
  - Boosting：各个预测函数只能顺序生成，因为后一轮模型参数需要前一轮模型的结果。

#### 2.4 总结

- 这两种方法都是把若干个分类器整合为一个分类器的方法，只是整合的方式不一样，最终得到不一样的效果，将不同的分类算法套入到此类算法框架中一定程度上会提高了原单一分类器的分类效果，但是也增大了计算量。
- 下面是将决策树与这些算法框架进行结合所得到的新的算法：
  - Bagging + 决策树 = 随机森林
  - AdaBoost + 决策树 = 提升树
  - Gradient Boosting + 决策树 = GBDT

**集成方法众多，本文主要记录学习Boosting方法中的一种最流行的版本，即AdaBoost。**

### 3. AdaBoost

- AdaBoost算法是基于Boosting思想的机器学习算法，AdaBoost是adaptive boosting（自适应boosting）的缩写，其运行过程如下：

  - **计算样本权重**：

    - **训练数据中的每个样本，赋予其权重，即样本权重**，用向量D表示，这些权重都初始化成相等值。假设有n个样本的训练集：${(x_1,y_1),(x_2,y_2)...(x_n,y_n)}$
    - 设定每个权重都是相等的，即1/n。

  - **计算错误率**：

    - 利用第一个弱学习算法h1对其进行学习，学习完成后进行错误率的统计：
    - $\varepsilon = \frac{未正确分类的样本数目}{所有的样本数目}$

  - **计算弱学习算法权重**

    - 弱学习算法也有一个权重，用向量$\alpha$表示，利用错误率计算权重$\alpha$:

    $$
    \alpha=\frac{1}{2} \ln \left(\frac{1-\varepsilon}{\varepsilon}\right)
    $$

  - **更新样本权重**

    - 在第一次学习完成之后，计算出alpha值之后，对权重进行更新，**使得正确分类的样本权重降低，而错分的样本权重升高。接下来的学习中可对其重点学习**，D的计算如下：

    $$
    D_{t+1}(i)=\frac{D_{t}(i)}{Z_{t}} \times\left\{\frac{e^{-\alpha_t}}{e^{\alpha_t}} \quad  \frac{\text { if } h_{t}\left(x_{i}\right)  = y_{i}}{\text { if } h_{t}\left(x_{i}\right) \neq y_{i}}\right.
    $$

    - 其中，ht(xi) = yi表示对第i个样本训练正确，不等于则表示分类错误。Zt是一个归一化因子：

    $$
    Z_t = sum(D)
    $$

    - 将上述两个公式合并化简：

    $$
    D_{t+1}(i)=\frac{D_{t}(i) \exp \left(-\alpha_{t} y_{i} h_{t}\left(x_{i}\right)\right)}{\operatorname{sum}(D)}
    $$

### 4. AdaBoost算法

- 算法示意图：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/10-1.jpg" style="zoom: 33%;" />

- 上图所示，左边是数据集，其中直方图的**不同宽度表示每个样例上的不同权重**。在经过一个分类器之后，加权的预测结果会通过三角形中的alpha值进行加权。每个三角形中输出的加权结果在圆形中求和，得到最终的输出结果。
- 算法的运行过程：
  - **训练数据中的每个样本，并赋予一个权重，这些权重构成了向量D。**
  - **一开始，这些权重都初始化成相等值。首先在训练数据上训练出一个弱分类器并计算该分类器的错误率，然后在同一数据集上再次训练弱分类器。**
  - **在分类器的第二次训练中，将会重新调整每个样本的权重，其中第一次分对的样本的权重会降低，第一次分错的样本的权重会提高。**
  - **在计算出D之后，AdaBoost又进入下一轮迭代。算法会不断重复训练和调整权重的过程，直到训练错误率为0或者弱分类器的数目达到用户的指定值为止。**

### 5. 基于单层决策树构建弱分类器

- 在建立AdaBoost算法之前，我们必须先建立弱分类器，并保存样本的权重。弱分类器使用单层决策树，也称为决策树桩，它是一种简单的决策树，通过给定的阈值，进行分类。

#### 5.1 可视化训练数据

- 创建单层决策树，线创建训练数据，代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt
"""
创建单层决策树的数据集
returns：
	dataMat:数据矩阵
	classLabels：数据标签
"""
def loadSimpData():
	dataMat = np.matrix(
		[[1. , 2.1],
		[1.5, 1. ],
		[1.3, 1. ],
		[1. , 1. ],
		[2. , 1. ]])
	classLabels = [1.0,1.0,-1.0,-1.0,1.0]
	return dataMat,classLabels 
"""
数据可视化
Parameters:
	dataMat:数据矩阵
	labelMat：数据标签
"""
def showDataSet(dataMat,labelMat):
	data_plus = []             #正样本
	data_minus = []            #负样本
	for i in range(len(dataMat)):
		if labelMat[i]>0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	#将两个列表转为numpy矩阵
	data_plus_np = np.array(data_plus)
	data_minus_np = np.array(data_minus)
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
	plt.show()
 
if __name__ == '__main__':
	dataArr,classLabels = loadSimpData()
	showDataSet(dataArr,classLabels)

```

- 解释两个后有的numpy矩阵，输出如下：

```python
[[[1.  2.1]]

 [[1.5 1. ]]

 [[2.  1. ]]]   # data_plus_np
 
 
[[[1.3 1. ]]

 [[1.  1. ]]]  # data_minus_np
 
[[1.  1.5 2. ]] # np.transpose(data_plus_np)[0]
```

- 构造数据集可视化如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/10-2.jpg)

- 可以看到，如果想要试着从某个坐标轴上选择一个值（即选择一条与坐标轴平行的直线）来将所有的蓝色圆点和橘色圆点分开，这显然是不可能的。这就是单层决策树难以处理的一个著名问题。通过使用多颗单层决策树，我们可以构建出一个能够对该数据集完全正确分类的分类器。

#### 5.2 构建单层决策树

- 我们首先设置一个分类阈值，这里的分类阈值指的是移动如下图的切分线与坐标轴的交点的大小：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/10-3.jpg)

- 横线上边的是一个类别，横线下边是一个类别。显然，此时有一个蓝点分类错误，计算此时的分类误差，误差为1/5 = 0.2。这个横线与坐标轴的y轴的交点，就是我们设置的阈值，通过不断改变阈值的大小，找到使单层决策树的分类误差最小的阈值。同理，竖线也是如此，找到最佳分类的阈值，就找到了最佳单层决策树，编写代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt
"""
创建单层决策树的数据集
returns：
	dataMat:数据矩阵
	classLabels：数据标签
"""
def loadSimpData():
	dataMat = np.matrix(
		[[1. , 2.1],
		[1.5, 1.6],
		[1.3, 1. ],
		[1. , 1. ],
		[2. , 1. ]])
	classLabels = [1.0,1.0,-1.0,-1.0,1.0]
	return dataMat,classLabels 
"""
数据可视化
Parameters:
	dataMat:数据矩阵
	labelMat：数据标签
"""
def showDataSet(dataMat,labelMat):
	data_plus = []             #正样本
	data_minus = []            #负样本
	for i in range(len(dataMat)):
		if labelMat[i]>0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	#将两个列表转为numpy矩阵
	data_plus_np = np.array(data_plus)
	data_minus_np = np.array(data_minus)
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])  #正样本散点图
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])#负样本散点图
	plt.show()
 

"""
 单层决策树分类函数
 Parameters：
	dataMatrix
	dimen:第dimen列，也就是第几个特征
	threshVal：阈值
	threshIneq：标志
returns：
	retArray：分类结果
"""
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray = np.ones((np.shape(dataMatrix)[0],1))       #初始化retArray为1
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0    #将小于阈值和大于阈值的都赋值为-1
	return retArray

"""
找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
"""
def buildStump(dataArr,classLabels,D):
	dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
	m,n = np.shape(dataMatrix)
	numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
	minError = float('inf')                                    #最小误差初始化为正无穷大
	for i in range(n):                                         #遍历所有特征
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()        
		#找到特征中最小的值和最大值
		stepSize = (rangeMax - rangeMin) / numSteps              #计算步长
		for j in range(-1, int(numSteps) + 1):                                     
			for inequal in ['lt', 'gt']:   #大于和小于的情况，均遍历。lt:less than，gt:greater than
				threshVal = (rangeMin + float(j) * stepSize)                     #计算阈值
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#计算分类结果
				errArr = np.mat(np.ones((m,1)))                                 #初始化误差矩阵
				errArr[predictedVals == labelMat] = 0                           #分类正确的,赋值为0
				weightedError = D.T * errArr                                    #计算误差
				print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
				if weightedError < minError:                                 #找到误差最小的分类方式
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump,minError,bestClasEst
 
if __name__ == '__main__':
	dataArr,classLabels = loadSimpData()
	dataMatrix = np.array(dataArr)
	print(dataMatrix)
	D = np.mat(np.ones((5, 1)) / 5)
	bestStump,minError,bestClasEst = buildStump(dataArr,classLabels,D)
	print('bestStump:\n', bestStump)
	print('minError:\n', minError)
	print('bestClasEst:\n', bestClasEst)


```

- 运行结果：

```python
[[1.  2.1]
 [1.5 1.6]
 [1.3 1. ]
 [1.  1. ]
 [2.  1. ]]
split: dim 0, thresh 0.90, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 0.90, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.00, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.00, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.10, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.10, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.20, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.20, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.30, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.30, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.40, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.40, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.50, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.50, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.60, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.60, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.70, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.70, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.80, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.80, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.90, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.90, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 2.00, thresh ineqal: lt, the weighted error is 0.600
split: dim 0, thresh 2.00, thresh ineqal: gt, the weighted error is 0.400
split: dim 1, thresh 0.89, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 0.89, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.00, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.00, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.11, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.11, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.22, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.22, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.33, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.33, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.44, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.44, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.55, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.55, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.66, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.66, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.77, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.77, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.88, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.88, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.99, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.99, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 2.10, thresh ineqal: lt, the weighted error is 0.600
split: dim 1, thresh 2.10, thresh ineqal: gt, the weighted error is 0.400
bestStump:
 {'dim': 0, 'thresh': 1.3, 'ineq': 'lt'}
minError:
 [[0.2]]
bestClasEst:
 [[-1.]
 [ 1.]
 [-1.]
 [-1.]
 [ 1.]]
[Finished in 0.4s]
```

- 代码不难理解，就是通过遍历，改变不同的阈值，计算最终的分类误差，找到分类误差最小的分类方式，即为我们要找的最佳单层决策树。这里lt表示less than，表示分类方式，对于小于阈值的样本点赋值为-1，gt表示greater than，也是表示分类方式，对于大于阈值的样本点赋值为-1。经过遍历，我们找到，训练好的最佳单层决策树的最小分类误差为0.2，就是对于该数据集，无论用什么样的单层决策树，分类误差最小就是0.2。这就是我们训练好的弱分类器。
- 接下来，使用AdaBoost算法提升分类器性能，将分类误差缩短到0：

#### 5.3 使用AdaBoost算法提示分类器性能

- 代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt
"""
创建单层决策树的数据集
returns：
	dataMat:数据矩阵
	classLabels：数据标签
"""
def loadSimpData():
	dataMat = np.matrix(
		[[1. , 2.1],
		[1.5, 1.6],
		[1.3, 1. ],
		[1. , 1. ],
		[2. , 1. ]])
	classLabels = [1.0,1.0,-1.0,-1.0,1.0]
	return dataMat,classLabels 
"""
数据可视化
Parameters:
	dataMat:数据矩阵
	labelMat：数据标签
"""
def showDataSet(dataMat,labelMat):
	data_plus = []             #正样本
	data_minus = []            #负样本
	for i in range(len(dataMat)):
		if labelMat[i]>0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	#将两个列表转为numpy矩阵
	data_plus_np = np.array(data_plus)
	data_minus_np = np.array(data_minus)
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
	plt.show()
 

"""
 单层决策树分类函数
 Parameters：
	dataMatrix
	dimen:第dimen列，也就是第几个特征
	threshVal：阈值
	threshIneq：标志
returns：
	retArray：分类结果
"""
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray = np.ones((np.shape(dataMatrix)[0],1))       #初始化retArray为1
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0    #将小于阈值和大于阈值的都赋值为-1
	return retArray

"""
找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
"""
def buildStump(dataArr,classLabels,D):
	dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
	m,n = np.shape(dataMatrix)
	numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
	minError = float('inf')                                      #最小误差初始化为正无穷大
	for i in range(n):                                           #遍历所有特征
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()        
		#找到特征中最小的值和最大值
		stepSize = (rangeMax - rangeMin) / numSteps                #计算步长
		for j in range(-1, int(numSteps) + 1):                                     
			for inequal in ['lt', 'gt']:                             #大于和小于的情况，均遍历。lt:less than，gt:greater than
				threshVal = (rangeMin + float(j) * stepSize)                     #计算阈值
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#计算分类结果
				errArr = np.mat(np.ones((m,1)))                                 #初始化误差矩阵
				errArr[predictedVals == labelMat] = 0                           #分类正确的,赋值为0
				weightedError = D.T * errArr                                    #计算误差
				print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
				if weightedError < minError:                                  #找到误差最小的分类方式
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump,minError,bestClasEst
 

"""
AdaBoost算法---提升分类器性能
"""
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
	weakClassArr = []
	m = np.shape(dataArr)[0]      #先取dataArr的形状（5，2），然后取得[0] = 5
	D = np.mat(np.ones((m,1))/m)  #初始化权重，（5，1）的0，2矩阵
	aggClassEst = np.mat(np.zeros((m,1)))
	for i in range(numIt):
		#构建单层决策树
		bestStump, error, classEst = buildStump(dataArr, classLabels, D) 
		print("D:",D.T)
		alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
		#计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
		bestStump['alpha'] = alpha   #存储弱学习算法权重
		weakClassArr.append(bestStump)   #存储单层决策树
		print("classEst:",classEst.T)
		expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
		#计算e的指数项:将原有的标签矩阵与预测的矩阵元素对应相乘，若分类正确，则得1，分类错误得-1
		D = np.multiply(D, np.exp(expon))
		D = D/D.sum()      #上下两步根据公式更新样本权重
		#计算AdaBoost误差，当误差为0的时候，退出循环
		aggClassEst += alpha * classEst                 
		#分类器的权重，每次都累加，分类器权重的符号与原数据标签符号一致才是正确
		print("aggClassEst: ", aggClassEst.T)
		# np.sign：使大于0的元素变为1，等于0的变为0，小于0的变为-1
		aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))     #计算误差
		errorRate = aggErrors.sum() / m
		print("total error: ", errorRate)
		if errorRate == 0.0: break                 #误差为0，退出循环
	return weakClassArr, aggClassEst

if __name__ == '__main__':
	dataArr,classLabels = loadSimpData()
	D = np.mat(np.ones((5, 1)) / 5)
	weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
	print(weakClassArr)
	print(aggClassEst)


```

- 运行结果数据：

```python
split: dim 0, thresh 0.90, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 0.90, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.00, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.00, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.10, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.10, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.20, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.20, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.30, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.30, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.40, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.40, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.50, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.50, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.60, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.60, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.70, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.70, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.80, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.80, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.90, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.90, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 2.00, thresh ineqal: lt, the weighted error is 0.600
split: dim 0, thresh 2.00, thresh ineqal: gt, the weighted error is 0.400
split: dim 1, thresh 0.89, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 0.89, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.00, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.00, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.11, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.11, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.22, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.22, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.33, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.33, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.44, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.44, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.55, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.55, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.66, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.66, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.77, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.77, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.88, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.88, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.99, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.99, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 2.10, thresh ineqal: lt, the weighted error is 0.600
split: dim 1, thresh 2.10, thresh ineqal: gt, the weighted error is 0.400
D: [[0.2 0.2 0.2 0.2 0.2]]
classEst: [[-1.  1. -1. -1.  1.]]
aggClassEst:  [[-0.69314718  0.69314718 -0.69314718 -0.69314718  0.69314718]]
total error:  0.2
split: dim 0, thresh 0.90, thresh ineqal: lt, the weighted error is 0.250
split: dim 0, thresh 0.90, thresh ineqal: gt, the weighted error is 0.750
split: dim 0, thresh 1.00, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.00, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.10, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.10, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.20, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.20, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.30, thresh ineqal: lt, the weighted error is 0.500
split: dim 0, thresh 1.30, thresh ineqal: gt, the weighted error is 0.500
split: dim 0, thresh 1.40, thresh ineqal: lt, the weighted error is 0.500
split: dim 0, thresh 1.40, thresh ineqal: gt, the weighted error is 0.500
split: dim 0, thresh 1.50, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.50, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.60, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.60, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.70, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.70, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.80, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.80, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.90, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.90, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 2.00, thresh ineqal: lt, the weighted error is 0.750
split: dim 0, thresh 2.00, thresh ineqal: gt, the weighted error is 0.250
split: dim 1, thresh 0.89, thresh ineqal: lt, the weighted error is 0.250
split: dim 1, thresh 0.89, thresh ineqal: gt, the weighted error is 0.750
split: dim 1, thresh 1.00, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.00, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.11, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.11, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.22, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.22, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.33, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.33, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.44, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.44, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.55, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.55, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.66, thresh ineqal: lt, the weighted error is 0.250
split: dim 1, thresh 1.66, thresh ineqal: gt, the weighted error is 0.750
split: dim 1, thresh 1.77, thresh ineqal: lt, the weighted error is 0.250
split: dim 1, thresh 1.77, thresh ineqal: gt, the weighted error is 0.750
split: dim 1, thresh 1.88, thresh ineqal: lt, the weighted error is 0.250
split: dim 1, thresh 1.88, thresh ineqal: gt, the weighted error is 0.750
split: dim 1, thresh 1.99, thresh ineqal: lt, the weighted error is 0.250
split: dim 1, thresh 1.99, thresh ineqal: gt, the weighted error is 0.750
split: dim 1, thresh 2.10, thresh ineqal: lt, the weighted error is 0.750
split: dim 1, thresh 2.10, thresh ineqal: gt, the weighted error is 0.250
D: [[0.5   0.125 0.125 0.125 0.125]]
classEst: [[ 1.  1. -1. -1. -1.]]
aggClassEst:  [[ 0.27980789  1.66610226 -1.66610226 -1.66610226 -0.27980789]]
total error:  0.2
split: dim 0, thresh 0.90, thresh ineqal: lt, the weighted error is 0.143
split: dim 0, thresh 0.90, thresh ineqal: gt, the weighted error is 0.857
split: dim 0, thresh 1.00, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.00, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.10, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.10, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.20, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.20, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.30, thresh ineqal: lt, the weighted error is 0.286
split: dim 0, thresh 1.30, thresh ineqal: gt, the weighted error is 0.714
split: dim 0, thresh 1.40, thresh ineqal: lt, the weighted error is 0.286
split: dim 0, thresh 1.40, thresh ineqal: gt, the weighted error is 0.714
split: dim 0, thresh 1.50, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.50, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.60, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.60, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.70, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.70, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.80, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.80, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.90, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.90, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 2.00, thresh ineqal: lt, the weighted error is 0.857
split: dim 0, thresh 2.00, thresh ineqal: gt, the weighted error is 0.143
split: dim 1, thresh 0.89, thresh ineqal: lt, the weighted error is 0.143
split: dim 1, thresh 0.89, thresh ineqal: gt, the weighted error is 0.857
split: dim 1, thresh 1.00, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.00, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.11, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.11, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.22, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.22, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.33, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.33, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.44, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.44, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.55, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.55, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.66, thresh ineqal: lt, the weighted error is 0.571
split: dim 1, thresh 1.66, thresh ineqal: gt, the weighted error is 0.429
split: dim 1, thresh 1.77, thresh ineqal: lt, the weighted error is 0.571
split: dim 1, thresh 1.77, thresh ineqal: gt, the weighted error is 0.429
split: dim 1, thresh 1.88, thresh ineqal: lt, the weighted error is 0.571
split: dim 1, thresh 1.88, thresh ineqal: gt, the weighted error is 0.429
split: dim 1, thresh 1.99, thresh ineqal: lt, the weighted error is 0.571
split: dim 1, thresh 1.99, thresh ineqal: gt, the weighted error is 0.429
split: dim 1, thresh 2.10, thresh ineqal: lt, the weighted error is 0.857
split: dim 1, thresh 2.10, thresh ineqal: gt, the weighted error is 0.143
D: [[0.28571429 0.07142857 0.07142857 0.07142857 0.5       ]]
classEst: [[1. 1. 1. 1. 1.]]
aggClassEst:  [[ 1.17568763  2.56198199 -0.77022252 -0.77022252  0.61607184]]
total error:  0.0
[{'dim': 0, 'thresh': 1.3, 'ineq': 'lt', 'alpha': 0.6931471805599453}, {'dim': 1, 'thresh': 1.0, 'ineq': 'lt', 'alpha': 0.9729550745276565}, {'dim': 0, 'thresh': 0.9, 'ineq': 'lt', 'alpha': 0.8958797346140273}]
[[ 1.17568763]
 [ 2.56198199]
 [-0.77022252]
 [-0.77022252]
 [ 0.61607184]]
[Finished in 0.4s]
```

- 代码流程分析：
  - 先进入AdaBoost算法流程，在该步骤迭代调用构建单层决策树算法（弱分类器）
  - 得到该弱分类器的信息保存至weakClassArr
  - 根据之前的alpha公式及更新D权重公式进行更新
  - 计算错误率，这个步骤采用比较分类器权重矩阵aggClassEst的符号与真实标签的符号判断正误
  - 若错误率不为0.在numIt之内进行循环更新迭代
  - 若错误率为0，则break结束，得到分类器组合信息weakClassArr和aggClassEst分类器权重
- 在第一轮迭代中，D中的所有值都相等。于是，只有第一个数据点被错分了。因此在第二轮迭代中，D向量给第一个数据点0.5的权重。这就可以通过变量aggClassEst的符号来了解总的类别。第二次迭代之后，我们就会发现第一个数据点已经正确分类了，但此时最后一个数据点却是错分了。D向量中的最后一个元素变为0.5，而D向量中的其他值都变得非常小。最后，第三次迭代之后aggClassEst所有值的符号和真是类别标签都完全吻合，那么训练错误率为0，程序终止运行。
- 最后训练结果包含了三个弱分类器，其中包含了分类所需要的所有信息。一共迭代了3次，所以训练了3个弱分类器构成一个使用AdaBoost算法优化过的分类器，分类器的错误率为0。

#### 5.4 测试

- 得到多个分类器信息及alpha值之后，可如下进行测试：

```python
"""
AdaBoost分类函数
Parameters:
	datToClass - 待分类样例
	classifierArr - 训练好的分类器
Returns:
	分类结果
"""
def adaClassify(datToClass,classifierArr):
	dataMatrix = np.mat(datToClass)
	m = np.shape(dataMatrix)[0]
	aggClassEst = np.mat(np.zeros((m,1)))
	for i in range(len(classifierArr)):      #遍历所有分类器，进行分类
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])            
		aggClassEst += classifierArr[i]['alpha'] * classEst
		print(aggClassEst)
	return np.sign(aggClassEst)              #分类结果
if __name__ == '__main__':
	dataArr,classLabels = loadSimpData()
	weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
	print(adaClassify([[0,0],[5,5]], weakClassArr))
```

- 运行结果数据：

```python
split: dim 0, thresh 0.90, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 0.90, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.00, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.00, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.10, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.10, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.20, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.20, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.30, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.30, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.40, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.40, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.50, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.50, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.60, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.60, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.70, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.70, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.80, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.80, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.90, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.90, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 2.00, thresh ineqal: lt, the weighted error is 0.600
split: dim 0, thresh 2.00, thresh ineqal: gt, the weighted error is 0.400
split: dim 1, thresh 0.89, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 0.89, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.00, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.00, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.11, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.11, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.22, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.22, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.33, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.33, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.44, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.44, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.55, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.55, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.66, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.66, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.77, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.77, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.88, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.88, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.99, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.99, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 2.10, thresh ineqal: lt, the weighted error is 0.600
split: dim 1, thresh 2.10, thresh ineqal: gt, the weighted error is 0.400
D: [[0.2 0.2 0.2 0.2 0.2]]
classEst: [[-1.  1. -1. -1.  1.]]
aggClassEst:  [[-0.69314718  0.69314718 -0.69314718 -0.69314718  0.69314718]]
total error:  0.2
split: dim 0, thresh 0.90, thresh ineqal: lt, the weighted error is 0.250
split: dim 0, thresh 0.90, thresh ineqal: gt, the weighted error is 0.750
split: dim 0, thresh 1.00, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.00, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.10, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.10, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.20, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.20, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.30, thresh ineqal: lt, the weighted error is 0.500
split: dim 0, thresh 1.30, thresh ineqal: gt, the weighted error is 0.500
split: dim 0, thresh 1.40, thresh ineqal: lt, the weighted error is 0.500
split: dim 0, thresh 1.40, thresh ineqal: gt, the weighted error is 0.500
split: dim 0, thresh 1.50, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.50, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.60, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.60, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.70, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.70, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.80, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.80, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 1.90, thresh ineqal: lt, the weighted error is 0.625
split: dim 0, thresh 1.90, thresh ineqal: gt, the weighted error is 0.375
split: dim 0, thresh 2.00, thresh ineqal: lt, the weighted error is 0.750
split: dim 0, thresh 2.00, thresh ineqal: gt, the weighted error is 0.250
split: dim 1, thresh 0.89, thresh ineqal: lt, the weighted error is 0.250
split: dim 1, thresh 0.89, thresh ineqal: gt, the weighted error is 0.750
split: dim 1, thresh 1.00, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.00, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.11, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.11, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.22, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.22, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.33, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.33, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.44, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.44, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.55, thresh ineqal: lt, the weighted error is 0.125
split: dim 1, thresh 1.55, thresh ineqal: gt, the weighted error is 0.875
split: dim 1, thresh 1.66, thresh ineqal: lt, the weighted error is 0.250
split: dim 1, thresh 1.66, thresh ineqal: gt, the weighted error is 0.750
split: dim 1, thresh 1.77, thresh ineqal: lt, the weighted error is 0.250
split: dim 1, thresh 1.77, thresh ineqal: gt, the weighted error is 0.750
split: dim 1, thresh 1.88, thresh ineqal: lt, the weighted error is 0.250
split: dim 1, thresh 1.88, thresh ineqal: gt, the weighted error is 0.750
split: dim 1, thresh 1.99, thresh ineqal: lt, the weighted error is 0.250
split: dim 1, thresh 1.99, thresh ineqal: gt, the weighted error is 0.750
split: dim 1, thresh 2.10, thresh ineqal: lt, the weighted error is 0.750
split: dim 1, thresh 2.10, thresh ineqal: gt, the weighted error is 0.250
D: [[0.5   0.125 0.125 0.125 0.125]]
classEst: [[ 1.  1. -1. -1. -1.]]
aggClassEst:  [[ 0.27980789  1.66610226 -1.66610226 -1.66610226 -0.27980789]]
total error:  0.2
split: dim 0, thresh 0.90, thresh ineqal: lt, the weighted error is 0.143
split: dim 0, thresh 0.90, thresh ineqal: gt, the weighted error is 0.857
split: dim 0, thresh 1.00, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.00, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.10, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.10, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.20, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.20, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.30, thresh ineqal: lt, the weighted error is 0.286
split: dim 0, thresh 1.30, thresh ineqal: gt, the weighted error is 0.714
split: dim 0, thresh 1.40, thresh ineqal: lt, the weighted error is 0.286
split: dim 0, thresh 1.40, thresh ineqal: gt, the weighted error is 0.714
split: dim 0, thresh 1.50, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.50, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.60, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.60, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.70, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.70, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.80, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.80, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 1.90, thresh ineqal: lt, the weighted error is 0.357
split: dim 0, thresh 1.90, thresh ineqal: gt, the weighted error is 0.643
split: dim 0, thresh 2.00, thresh ineqal: lt, the weighted error is 0.857
split: dim 0, thresh 2.00, thresh ineqal: gt, the weighted error is 0.143
split: dim 1, thresh 0.89, thresh ineqal: lt, the weighted error is 0.143
split: dim 1, thresh 0.89, thresh ineqal: gt, the weighted error is 0.857
split: dim 1, thresh 1.00, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.00, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.11, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.11, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.22, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.22, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.33, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.33, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.44, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.44, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.55, thresh ineqal: lt, the weighted error is 0.500
split: dim 1, thresh 1.55, thresh ineqal: gt, the weighted error is 0.500
split: dim 1, thresh 1.66, thresh ineqal: lt, the weighted error is 0.571
split: dim 1, thresh 1.66, thresh ineqal: gt, the weighted error is 0.429
split: dim 1, thresh 1.77, thresh ineqal: lt, the weighted error is 0.571
split: dim 1, thresh 1.77, thresh ineqal: gt, the weighted error is 0.429
split: dim 1, thresh 1.88, thresh ineqal: lt, the weighted error is 0.571
split: dim 1, thresh 1.88, thresh ineqal: gt, the weighted error is 0.429
split: dim 1, thresh 1.99, thresh ineqal: lt, the weighted error is 0.571
split: dim 1, thresh 1.99, thresh ineqal: gt, the weighted error is 0.429
split: dim 1, thresh 2.10, thresh ineqal: lt, the weighted error is 0.857
split: dim 1, thresh 2.10, thresh ineqal: gt, the weighted error is 0.143
D: [[0.28571429 0.07142857 0.07142857 0.07142857 0.5       ]]
classEst: [[1. 1. 1. 1. 1.]]
aggClassEst:  [[ 1.17568763  2.56198199 -0.77022252 -0.77022252  0.61607184]]
total error:  0.0
[[-0.69314718]
 [ 0.69314718]]
[[-1.66610226]
 [ 1.66610226]]
[[-2.56198199]
 [ 2.56198199]]
[[-1.]
 [ 1.]]
[Finished in 0.7s]
```

- 代码很简单，在之前代码的基础上，添加adaClassify()函数，该函数遍历所有训练得到的弱分类器，利用单层决策树，输出的类别估计值乘以该单层决策树的分类器权重alpha，然后累加到aggClassEst上，最后通过sign函数最终的结果。可以看到，分类没有问题，(5,5)属于正类，(0,0)属于负类。