### 1. 前言

- 本次将学习使用AdaBoost算法来实战训练一些较大的数据集，学习Sklearn中的AdaBoost算法。

- 在学习Logistic回归时，使用Logistic回归方法训练马疝病数据集，预测病马死亡率。当时的训练结果如下图所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/11-1.jpg)

- 这个是使用Sklearn的LogisticRegression()训练的分类器，可以看到，正确率约为73.134%，也就是说错误率约为26.866%。可以看到错误率还是蛮高的，现在我们使用AdaBoost算法，训练出一个更强的分类器，这里的数据集有所变化，之前的标签是0和1，现在将标签改为+1和-1，其他数据不变。
- 更改后的[数据下载地址](https://github.com/yearing1017/Machine-Learning/tree/master/AdaBoost)

### 2. 手写AdaBoost算法训练

- 用Python写的AbaBoost算法进行训练，添加loadDataSet函数用于加载数据集。编写代码如下：

```python
import numpy as np

def loadDataSet(fileName):
	numFeat = len((open(fileName).readline().split('\t')))
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat - 1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat, labelMat

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
	minError = float('inf')                                             #最小误差初始化为正无穷大
	for i in range(n):                                                  #遍历所有特征
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
		#找到特征中最小的值和最大值
		stepSize = (rangeMax - rangeMin) / numSteps                                #计算步长
		for j in range(-1, int(numSteps) + 1):                                     
			for inequal in ['lt', 'gt']:  #大于和小于的情况，均遍历。lt:less than，gt:greater than
				threshVal = (rangeMin + float(j) * stepSize)                     #计算阈值
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#计算分类结果
				errArr = np.mat(np.ones((m,1)))                                 #初始化误差矩阵
				errArr[predictedVals == labelMat] = 0                           #分类正确的,赋值为0
				weightedError = D.T * errArr                                    #计算误差
				#print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
				if weightedError < minError:                                    #找到误差最小的分类方式
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
		#print("D:",D.T)
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
		#print("aggClassEst: ", aggClassEst.T)
		# np.sign：使大于0的元素变为1，等于0的变为0，小于0的变为-1
		aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))     #计算误差
		errorRate = aggErrors.sum() / m
		print("total error: ", errorRate)
		if errorRate == 0.0: break                 #误差为0，退出循环
	return weakClassArr, aggClassEst


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
	for i in range(len(classifierArr)):               #遍历所有分类器，进行分类
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])            
		aggClassEst += classifierArr[i]['alpha'] * classEst
		#print(aggClassEst)
	return np.sign(aggClassEst) #分类结果




if __name__ == '__main__':
	dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
	weakClassArr,aggClassEst = adaBoostTrainDS(dataArr,LabelArr)
	testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
	for item in weakClassArr:
		print(item)          #输出所有的弱分类器
	predictions = adaClassify(dataArr,weakClassArr)
	#计算错误率
	errArr = np.mat(np.ones((len(dataArr), 1)))
	print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))
	predictions = adaClassify(testArr, weakClassArr)
	errArr = np.mat(np.ones((len(testArr), 1)))
	print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
```

- 运行结果如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/11-2.jpg)

- 这里输出了AdaBoost算法训练好的分类器的组合，我们只迭代了40次，也就是训练了40个弱分类器。
- 最终，训练集的错误率为19.732%，测试集的错误率为19.403%，可以看到相对于Sklearn的罗辑回归方法，错误率降低了很多。这个仅仅是我们训练40个弱分类器的结果，如果训练更多弱分类器，效果会更好。
- 但是当弱分类器数量过多的时候，你会发现训练集错误率降低很多，但是测试集错误率提升了很多，这种现象就是**过拟合(overfitting)。**分类器对训练集的拟合效果好，但是缺失了普适性，只对训练集的分类效果好，这是我们不希望看到的。

### 3. 使用Sklearn的AdaBoost

- [英文文档地址](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)

- sklearn.ensemble模块提供了很多集成方法，AdaBoost、Bagging、随机森林等。本文使用的是AdaBoostClassifier。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/11-3.jpg)

- 先看下AdaBoostClassifier这个函数，一共有5个参数：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/11-4.jpg)

- **参数说明如下：**
  - **base_estimator：可选参数**，默认为DecisionTreeClassifier。理论上可以选择任何一个分类或者回归学习器，不过需要支持样本权重。我们常用的一般是CART决策树或者神经网络MLP。默认是决策树，即AdaBoostClassifier默认使用CART分类树DecisionTreeClassifier，而AdaBoostRegressor默认使用CART回归树DecisionTreeRegressor。另外有一个要注意的点是，如果我们选择的AdaBoostClassifier算法是SAMME.R，则我们的弱分类学习器还需要支持概率预测，也就是在scikit-learn中弱分类学习器对应的预测方法除了predict还需要有predict_proba。
  - **algorithm：**可选参数，默认为SAMME.R。scikit-learn实现了两种Adaboost分类算法，SAMME和SAMME.R。两者的主要区别是弱学习器权重的度量，SAMME使用对样本集分类效果作为弱学习器权重，而SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重。由于SAMME.R使用了概率度量的连续值，迭代一般比SAMME快，因此AdaBoostClassifier的默认算法algorithm的值也是SAMME.R。我们一般使用默认的SAMME.R就够了，但是要注意的是使用了SAMME.R， 则弱分类学习器参数base_estimator必须限制使用支持概率预测的分类器。SAMME算法则没有这个限制。
  - **n_estimators：**整数型，可选参数，默认为50。弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是50。在实际调参的过程中，我们常常将n_estimators和下面介绍的参数learning_rate一起考虑。
  - **learning_rate：**浮点型，可选参数，默认为1.0。每个弱学习器的权重缩减系数，取值范围为0到1，对于同样的训练集拟合效果，较小的v意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的v开始调参，默认是1。
  - **random_state：**整数型，可选参数，默认为None。如果RandomState的实例，random_state是随机数生成器; 如果None，则随机数生成器是由np.random使用的RandomState实例。
- 编写代码如下：

```python
# -*-coding:utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat
if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('horseColicTraining2.txt')
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), algorithm = "SAMME", n_estimators = 10)
    bdt.fit(dataArr, classLabels)
    predictions = bdt.predict(dataArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != classLabels].sum() / len(dataArr) * 100))
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))
```

- bdt：分类器组合
- bdt.fit(X,Y):在数据集(X,Y)上训练模型
- bdt.predict(X)预测数据集X的结果
- 运行结果如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/11-5.jpg)

- 我们使用DecisionTreeClassifier作为使用的弱分类器，使用AdaBoost算法训练分类器。可以看到训练集的错误率为16.054%，测试集的错误率为：17.910%。更改n_estimators参数，你会发现跟我们自己写的代码，更改迭代次数的效果是一样的。n_enstimators参数过大，会导致过拟合。