### 1. 前言

- 线性回归创建的模型需要拟合所有样本点（局部加权线性回归除外），**但是当数据拥有众多特征并且特征之间关系十分复杂时，构建全局模型就很困难了。**
- **本章学习CART（分类回归树）的树构建算法。该算法既可用于分类又可用于回归。接着学习树的剪枝技术（防止树的过拟合）。**

### 2. CART用于回归

- 之前学习了决策树的原理和代码实现，使用使用决策树进行分类。
- 决策树不断将数据切分成小数据集，直到所有目标标量完全相同，或者数据不能再切分为止。
- 决策树是一种贪心算法，它要在给定时间内做出最佳选择，但不关心能否达到全局最优。

#### 2.1 ID3算法的弊端

- 决策树的树构建算法是ID3。**ID3的做法是每次选取当前最佳的特征来分割数据，并按照该特征的所有可能取值来切分。**也就是说，如果一个特征有4种取值，那么数据将被切分成4份。
- 一旦按某特征切分后，该特征在之后的算法执行过程中将不再起作用，有观点认为这种切分方式过于迅速。
- 除了切分过于迅速外，**ID3算法还存在另一个问题，它不能直接处理连续型特征。只有事先将连续型特征离散化，才能在ID3算法中使用。**但这种转换过程会破坏连续型变量的内在特性。

#### 2.2 CART算法

- 与ID3算法相反，CART算法正好适用于连续型特征。
- **CART算法使用二元切分法来处理连续型变量。**而使用二元切分法则易于对树构建过程进行调整以处理连续型特征。**具体的处理方法是：如果特征值大于给定值就走左子树，否则就走右子树。**

- CART算法分为两步：**决策树生成、决策树剪枝**。
- **决策树生成**：递归地构建二叉决策树的过程，基于训练数据集生成决策树，生成的决策树要尽量大；自上而下从根开始建立节点，在每个节点处要选择一个最好的属性来分裂，使得子节点中的训练集尽量的纯。不同的算法使用不同的指标来定义"最好"。
- 在使用ID3算法构建决策树中，我们先根据信息熵的计算找到最佳特征切分数据集构建决策树。CART算法的决策树生成也是如此，实现过程如下：
  - **使用CART算法选择特征**
  - **根据特征切分数据集合**
  - **构建树**

#### 2.3 简单的根据特征切分数据集合

- **numpy中的nonzero函数用法：**返回数组中不为0的下标。
- 简单用法：在一维数组中，直接输出下标，如下代码：

```python
a=np.array([True,False,True,False])
b=np.nonzero(a)
print(b)
# 此处用的数组为true和false类型，与真实数字用法相同
# 输出：(array([0, 2], dtype=int64)) 即下标为0和2的元素值非0
```

- 二维数组中：输出行和列的下标，进行组合得出位置。

```python
a=np.array([[True,False,True],
						[True,False,False]])
b=np.nonzero(a)
print(b)
# 输出：(array([0, 0, 1], dtype=int64), array([0, 2, 0], dtype=int64))
# 前面是行，后面是列，组合得：下标为（0,0）,(0,2),(1,0)的元素值不为0
```

- 了解了该方法的用例，我们来看一个简单的根据特征切分数据：

```python
import numpy as np
"""
函数说明:根据特征切分数据集合
    Parameters:
        dataSet - 数据集合
        feature - 带切分的特征
        value - 该特征的值
    Returns:
        mat0 - 切分的数据集合0
        mat1 - 切分的数据集合1
"""
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    print(dataSet[:,feature]) #输出feature代表的列，前面的:代表取所有的行
    print(dataSet[:,feature]>value)
    print(np.nonzero(dataSet[:,feature]>value))
    print(np.nonzero(dataSet[:,feature]<=value))
    return mat0, mat1
 
if __name__ == '__main__':
    testMat = np.mat(np.eye(4))
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print('原始集合:\n', testMat)
    print('mat0:\n', mat0)
```

- 运行结果：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-1.jpg)

- 从上图我们可以看到nonzero方法的结果，是个二维的，所以只取[0]第一维，再取得特定的行的内容。
- 我们先创建一个单位矩阵，然后根据切分规则，对数据矩阵进行切分。可以看到binSplitDataSet函数根据特定规则，对数据矩阵进行切分。
- 现在我们已经可以根据特征和特征值对数据进行切分了，mat0存放的是大于指定特征值的矩阵，mat1存放的是小于指定特征值的矩阵。接下来，我们就看看如何使用CART算法选择最佳分类特征。

#### 2.4 CART算法

- 假设X与Y分别为输入和输出变量，并且Y是连续变量，给定训练数据集：D表示整个数据集合，n为特征数。

$$
D=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{n}, y_{n}\right)\right\}
$$

- **一个回归树对应着输入空间（即特征空间）的一个划分以及在划分的单元上的输出值。**
- 假设已将输入空间划分为M个单元R1,R2,...Rm，并且在每个单元Rm上有一个固定的输出值Cm，于是回归树模型可表示为：

$$
f(x)=\sum_{m=1}^{M} c_{m} I\left(x \in R_{m}\right)
$$

- 计算模型输出值与实际值的误差：

$$
\sum_{x_{i} \in R_{m}}\left(y_{i}-f\left(x_{i}\right)\right)^{2}
$$

- 我们希望每个单元上的Cm，可以使得平方误差最小化。易知，当**Cm为相应单元的所有实际值的均值时，可以到最优：**

$$
\hat{c}_{m}=\operatorname{ave}\left(y_{i} | x_{i} \in R_{m}\right)
$$

- **那么如何生成这些单元划分？**

- 假设，我们选择变量$x_j$ 为切分变量，它的取值 s 为切分点，那么就会得到两个区域：

$$
\mathrm{R}_{1}(j, s)=\left\{x | x^{(j)} \leq s\right\}, \mathrm{R}_{2}(j, s)=\left\{x | x^{(j)}>s\right\}
$$

- **当j和s固定时，我们要找到两个区域的代表值c1，c2使各自区间上的平方差最小：**

$$
\min _{j, s}\left[\min _{c_{1}} \sum_{\left.x_{i} \in R_{1} j, s\right)}\left(y_{i}-c_{1}\right)^{2}+\min _{c_{2}} \sum_{\left.x_{i} \in R_{2} j, s\right)}\left(y_{i}-c_{2}\right)^{2}\right]
$$

- 前面已经知道c1，c2为区间上的平均值时，可以得到最优。
- **那么对固定的 j 只需要找到最优的s。**
- **通过遍历所有的变量，我们可以找到最优的j，**这样我们就可以得到最优对（j，s），并得到两个区间。

- 这样的回归树通常称为最小二乘回归树。

- **上述算法流程如下图所示：**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-2.png)

- 除此之外，我们再定义两个参数，tolS和tolN，分别用于控制误差变化限制和切分特征最少样本数。这两个参数的意义是什么呢？就是防止过拟合，提前设置终止条件，实际上是在进行一种所谓的预剪枝操作。

- 数据集下载：[数据集](https://github.com/yearing1017/Machine-Learning/blob/master/Regression%20Trees/ex00.txt)，大致如下所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-3.jpg)

- 如上图所示，数据是2维的。先看下数据的分布情况，编写代码如下：

```python
#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
"""
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - 数据矩阵
"""
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')						# strip()去除line首尾字符，默认为空格
        fltLine = list(map(float, curLine))           # 转化为float类型
        dataMat.append(fltLine)												# dataMat类似[[1,2],[2,3],[3,4]]
    return dataMat
"""
    函数说明:绘制数据集
    Parameters:
        filename - 文件名
    Returns:
        无
"""
def plotDataSet(filename):
    dataMat = loadDataSet(filename)                                 # 加载数据集
    n = len(dataMat)                                                # 数据个数
    xcord = []; ycord = []                                          # 样本点
    for i in range(n):                                                    
        xcord.append(dataMat[i][0]); ycord.append(dataMat[i][1])    # 对每组数据取x,y坐标
    fig = plt.figure()
    ax = fig.add_subplot(111)                                       # 添加subplot
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)         # 绘制样本点
    plt.title('DataSet')                                            # 绘制title
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    filename = 'ex00.txt'
    plotDataSet(filename)
```

- 运行结果如下图所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-4.jpg)

- 可以看到，这是一个很简单的数据集，我们先利用这个数据集测试CART算法:

```python
import numpy as np

"""
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - 数据矩阵
"""
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')			  # strip()去除line首尾字符，默认为空格
        fltLine = list(map(float, curLine))       # 转化为float类型
        dataMat.append(fltLine)						        # dataMat类似[[1,2],[2,3],[3,4]]
    return dataMat


"""
函数说明:根据特征切分数据集合
    Parameters:
        dataSet - 数据集合
        feature - 带切分的特征
        value - 该特征的值
    Returns:
        mat0 - 切分的数据集合0
        mat1 - 切分的数据集合1
"""
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1

"""
    函数说明:生成叶结点
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的均值
"""
def regLeaf(dataSet):
	return np.mean(dataSet[:,-1]) # 取最后一列的平均值

"""
    函数说明:误差估计函数
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的总方差
"""
def regErr(dataSet):
	# np.var()是均方差，均方差就是将方差进行了平均化,乘于数据的行数为总方差
	return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]


"""
    函数说明:找到数据的最佳二元切分方式函数
    Parameters:
        dataSet  - 数据集合
        leafType - 生成叶结点
        regErr   - 误差估计函数
        ops      - 用户定义的参数构成的元组
    Returns:
        bestIndex - 最佳切分特征
        bestValue - 最佳特征值
"""
def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
	import types
	# tolS允许的误差下降值，tolN切分的最少样本数
	tolS = ops[0]; tolN = ops[1]
	# 使用set判断是否所有值都相等
	# dataSet[:,-1].T和tolist前后形式一样，都是[[1,2,3,4]]类似
	# 但是tolist之后，就可以使用[0]取出[1,2,3,4]
	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
		return None, leafType(dataSet)  # 返回空的最佳切分特征
	m,n = np.shape(dataSet)
	# 默认最后一个特征为最佳切分特征，计算误差估计
	S = errType(dataSet)
	# 最佳误差、最佳特征切分的索引值、最佳特征值
	bestS = float('inf'); bestIndex = 0; bestValue = 0
	# 遍历所有特征列
	for featIndex in range(n - 1):
		# 遍历所有特征值,A代表转成numpy数组,set用来去重
		for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
			# 根据特征和特征值切分数据集
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			# 如果切分的数据量少于tolN，则退出本次循环
			if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
			# 计算误差估计
			newS = errType(mat0) + errType(mat1)
			# 如果误差估计更小，则更新索引值和特征值
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	# 循环完之后得到的误差，如果与默认的S减少不大，则退出
	if (S - bestS)< tolS:
		return None, leafType(dataSet)
	# 根据最佳的切分特征和特征值切分数据集合
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	# 如果切分出的数据集很小则退出
	if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
		return None, leafType(dataSet)
	# 返回最佳切分特征和特征值
	return bestIndex, bestValue



if __name__ == '__main__':
	myDat = loadDataSet('ex00.txt') # myDat类似[[1,2],[2,3],[3,4]]
	myMat = np.mat(myDat)						
  """
  myMat类似[[1 2]
  				  [2 3]
            [3 4]]
  """
  
	feat, val = chooseBestSplit(myMat, regLeaf, regErr, (1, 4))
	print(feat)
	print(val)
```

- 运行结果为：0， 0.48813
- 可以看到，切分的最佳特征为第1列特征，最佳切分特征值为0.48813，这个特征值怎么选出来的？就是根据误差估计的大小，我们选择的这个特征值可以使误差最小化。
- 切分的特征和特征值我们已经选择好了，接下来就是利用选出的这两个变量创建回归树了。
- 创建方法很简单，**我们根据切分的特征和特征值切分出两个数据集，然后将两个数据集分别用于左子树的构建和右子树的构建，直到无法找到切分的特征为止。**因此，我们可以使用递归实现这个过程，在上面的代码基础上添加如下方法，编写代码如下：

```python
"""
    函数说明:树构建函数
    Parameters:
        dataSet - 数据集合
        leafType - 建立叶结点的函数
        errType - 误差计算函数
        ops - 包含树构建所有其他参数的元组
    Returns:
        retTree - 构建的回归树
"""
def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):

	# 选择最佳切分特征和特征值
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	# 如果没有特征,则返回特征值
	if feat == None: return val
	# 回归树
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	# 分成左数据集和右数据集
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	# 创建左子树和右子树，递归进入继续对左右子集切分
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree 



if __name__ == '__main__':
	myDat = loadDataSet('ex00.txt')
	myMat = np.mat(myDat)
	print(createTree(myMat))
```

- 运行结果如下，可以看出这棵树只有两个叶结点。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-5.jpg)

- 换一个复杂一点的数据集，分段常数数据集。下载地址：[数据](https://github.com/yearing1017/Machine-Learning/blob/master/Regression%20Trees/ex0.txt)，数据大致如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-6.jpg)

- 将第2列作为x轴数据，第3列作为y轴数据。对数据进行可视化，编写代码如下：

```python
#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - 数据矩阵
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))                    #转化为float类型
        dataMat.append(fltLine)
    return dataMat

def plotDataSet(filename):
    """
    函数说明:绘制数据集
    Parameters:
        filename - 文件名
    Returns:
        无
    """
    dataMat = loadDataSet(filename)                                     # 加载数据集
    n = len(dataMat)                                                    # 数据个数
    xcord = []; ycord = []                                              # 样本点
    for i in range(n):                                                    
        xcord.append(dataMat[i][1]); ycord.append(dataMat[i][2])        # 样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)                                           # 添加subplot
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)             # 绘制样本点
    plt.title('DataSet')                                                # 绘制title
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    filename = 'ex0.txt'
    plotDataSet(filename)
```

- 运行结果如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-7.jpg)

- 可以看到，这个数据集是分段的。我们针对此数据集创建回归树。代码同上，运行结果如下图所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-8.jpg)

- 可以看到，该数的结构中包含5个叶结点。现在为止，已经完成回归树的构建，但是**需要某种措施来检查构建过程是否得当。这个技术就是剪枝（tree pruning）技术。**

### 3. 剪枝

- 通过降低树的复杂度来避免过拟合的过程称为剪枝（pruning）。上面我们也已经知道，设置tolS和tolN就是一种预剪枝操作。
- **另一种形式的剪枝需要使用测试集和训练集，称作后剪枝（postpruning）**。

#### 3.1 预剪枝

- 预剪枝具有一定的局限性，我们通过一个新的数据集来观察：
- 数据集下载：[数据集](https://github.com/yearing1017/Machine-Learning/blob/master/Regression%20Trees/ex2.txt)，绘制数据如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-9.jpg)

- 可以看到，对于这个数据集与我们使用的第一个数据集很相似，但是区别在于y的数量级差100倍，数据分布相似，因此构建出的树应该也是只有两个叶结点。但是我们使用默认tolS和tolN参数创建树，你会发现运行结果如下所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-10.jpg)

- 可以看到，构建出的树有很多叶结点。产生这个现象的原因在于，停止条件tolS对误差的数量级十分敏感。如果在选项中花费时间并对上述误差容忍度取平均值，或许也能得到仅有两个叶结点组成的树：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-11.jpg)

- 可以看到，将参数tolS修改为10000后，构建的树就是只有两个叶结点。然而，显然这个值，需要我们经过不断测试得来，显然通过不断修改停止条件来得到合理结果并不是很好的办法。事实上，我们常常甚至不确定到底需要寻找什么样的结果。因为对于一个很多维度的数据集，你也不知道构建的树需要多少个叶结点。

#### 3.2 后剪枝

- **使用后剪枝方法需要将数据集分成测试集和训练集。**
- 首先指定参数，使得构建出的树足够大、足够复杂，便于剪枝。**接下来从上而下找到叶结点，用测试集来判断这些叶结点合并是否能降低测试集误差。如果是的话就合并。**

- 我们使用ex2.txt文件作为训练集，而使用的新数据集ex2test.txt文件作为测试集
- 测试集下载地址：[数据集](https://github.com/yearing1017/Machine-Learning/blob/master/Regression%20Trees/ex2test.txt)

- 现在我们使用ex2.txt训练回归树，然后利用ex2test.txt对回归树进行剪枝。
- 我们需要创建三个函数isTree()、getMean()、prune()。
- **isTree()用于测试输入变量是否是一棵树，返回布尔类型的结果。换句话说，该函数用于判断当前处理的结点是否是叶结点。**
- 第二个函数**getMean()是一个递归函数，它从上往下遍历树直到叶结点为止。如果找到两个叶结点则计算它们的平均值。该函数对树进行塌陷处理（即返回树平均值）。**
- 第三个函数**prune()则为后剪枝函数。**
- 编写代码如下：

```python
import numpy as np

"""
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - 数据矩阵
"""
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')			  # strip()去除line首尾字符，默认为空格
        fltLine = list(map(float, curLine))           # 转化为float类型
        dataMat.append(fltLine)						  # dataMat类似[[1,2],[2,3],[3,4]]
    return dataMat


"""
函数说明:根据特征切分数据集合
    Parameters:
        dataSet - 数据集合
        feature - 带切分的特征
        value - 该特征的值
    Returns:
        mat0 - 切分的数据集合0
        mat1 - 切分的数据集合1
"""
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1

"""
    函数说明:生成叶结点
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的均值
"""
def regLeaf(dataSet):
	return np.mean(dataSet[:,-1]) # 取最后一列的平均值

"""
    函数说明:误差估计函数
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的总方差
"""
def regErr(dataSet):
	# np.var()是均方差，均方差就是将方差进行了平均化,乘于数据的行数为总方差
	return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]


"""
    函数说明:找到数据的最佳二元切分方式函数
    Parameters:
        dataSet  - 数据集合
        leafType - 生成叶结点
        regErr   - 误差估计函数
        ops      - 用户定义的参数构成的元组
    Returns:
        bestIndex - 最佳切分特征
        bestValue - 最佳特征值
"""
def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
	import types
	# tolS允许的误差下降值，tolN切分的最少样本数
	tolS = ops[0]; tolN = ops[1]
	# 使用set判断是否所有值都相等
	# dataSet[:,-1].T和tolist前后形式一样，都是[[1,2,3,4]]类似
	# 但是tolist之后，就可以使用[0]取出[1,2,3,4]
	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
		return None, leafType(dataSet)  # 返回空的最佳切分特征
	m,n = np.shape(dataSet)
	# 默认最后一个特征为最佳切分特征，计算误差估计
	S = errType(dataSet)
	# 最佳误差、最佳特征切分的索引值、最佳特征值
	bestS = float('inf'); bestIndex = 0; bestValue = 0
	# 遍历所有特征列,最后一列为标签
	for featIndex in range(n - 1):
		# 遍历所有特征值,A代表转成numpy数组,set用来去重
		for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
			# 根据特征和特征值切分数据集
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			# 如果切分的数据量少于tolN，则退出本次循环
			if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
			# 计算误差估计
			newS = errType(mat0) + errType(mat1)
			# 如果误差估计更小，则更新索引值和特征值
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	# 循环完之后得到的误差，如果与默认的S减少不大，则退出
	if (S - bestS)< tolS:
		return None, leafType(dataSet)
	# 根据最佳的切分特征和特征值切分数据集合
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	# 如果切分出的数据集很小则退出
	if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
		return None, leafType(dataSet)
	# 返回最佳切分特征和特征值
	return bestIndex, bestValue
"""
    函数说明:树构建函数
    Parameters:
        dataSet - 数据集合
        leafType - 建立叶结点的函数
        errType - 误差计算函数
        ops - 包含树构建所有其他参数的元组
    Returns:
        retTree - 构建的回归树
"""
def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):

	# 选择最佳切分特征和特征值
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	# 如果没有特征,则返回特征值
	if feat == None: return val
	# 回归树
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	# 分成左数据集和右数据集
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	# 创建左子树和右子树，递归进入继续对左右子集切分
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree 

# 判断测试输入变量是否为一棵树
def isTree(obj):
	import types
	return (type(obj).__name__ == 'dict')

# 对树进行塌陷处理，即求均值
def getMean(tree):
	if isTree(tree['right']): tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):  tree['left'] = getMean(tree['left'])
	return (tree['left']+tree['right'])/2.0

# 后剪枝
def prune(tree, testData):
	# 如果测试集为空，则对树进行塌陷处理
	if np.shape(testData)[0] == 0: return getMean(tree)
	# 如果有左子树或者右子树，则切分数据
	if (isTree(tree['right']) or isTree(tree['left'])):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
	# 左子树 剪枝
	if isTree(tree['left']):tree['left'] = prune(tree['left'], lSet)

	#右子树 剪枝
	if isTree(tree['right']):tree['right'] = prune(tree['right'], rSet)
	# 递归剪枝中：
	# 如果当前结点的左右结点为叶结点，则使用测试集计算合并前后误差
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
		#计算没有合并的误差
		errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) + np.sum(np.power(rSet[:,-1] - tree['right'],2))
		#计算合并的均值
		treeMean = (tree['left'] + tree['right']) / 2.0
		#计算合并的误差
		errorMerge = np.sum(np.power(testData[:,-1] - treeMean, 2))
		#如果合并的误差小于没有合并的误差,则合并
		if errorMerge < errorNoMerge:
			return treeMean
		else: return tree
	else: return tree





if __name__ == '__main__':
	print('剪枝前:')
	train_filename = 'ex2.txt'
	train_Data = loadDataSet(train_filename)
	train_Mat = np.mat(train_Data)
	tree = createTree(train_Mat)
	print(tree)
	print('\n剪枝后:')
	test_filename = 'ex2test.txt'
	test_Data = loadDataSet(test_filename)
	test_Mat = np.mat(test_Data)
	print(prune(tree, test_Mat))
```

- 运行结果：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/15-12.jpg)

- 可以看到，树的大量结点已经被剪枝掉了，但没有像预期的那样剪枝成两部分，这说明后剪枝可能不如预剪枝有效。一般地，为了寻求最佳模型可以同时使用两种剪枝技术。