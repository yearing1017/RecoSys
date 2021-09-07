上节课主要介绍了逻辑回归，以输出概率的形式来处理二分类问题。介绍了逻辑回归的Cost function表达式，并使用梯度下降算法来计算最小化Cost function时对应的参数w和b。通过计算图的方式来讲述了神经网络的正向传播和反向传播两个过程。下面学习Python和向量化的相关知识。

### 1、**Vectorization**（向量化）

深度学习算法中，数据量很大，在程序中应该尽量减少使用loop循环语句，而可以使用向量运算来提高程序运行速度。

向量化就是利用矩阵运算的思想，大大提高运算速度。例如下面所示在Python中使用向量化要比使用循环计算速度快得多。

```python
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b) #dot(a,b)计算矩阵乘积，若a、b为一维数组，计算内积
toc = time.time()

print(c)
print("Vectorized version:" + str(1000*(toc-tic)) + "ms")

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print("for loop:" + str(1000*(toc-tic)) + "ms")
```

输出结果类似于：

```python
250286.989866
Vectorized version:1.5027523040771484ms
250286.989866
For loop:474.29513931274414ms
```

从程序运行结果上来看，该例子使用for循环运行时间是使用向量运算运行时间的约300倍。因此，深度学习算法中，使用向量化矩阵运算的效率要高得多。

为了加快深度学习神经网络运算速度，可以使用比CPU运算能力更强大的GPU。事实上，GPU和CPU都有并行指令（parallelization instructions），称为Single Instruction Multiple Data（SIMD）。SIMD是单指令多数据流，能够复制多个操作数，并把它们打包在大型寄存器的一组指令集。SIMD能够大大提高程序运行速度，例如python的numpy库中的内建函数（built-in function）就是使用了SIMD指令。相比而言，GPU的SIMD要比CPU更强大一些。

### 2、**More Vectorization Examples**

上一部分讲了应该尽量避免使用for循环而使用向量化矩阵运算。在python的numpy库中，我们通常使用np.dot()函数来进行矩阵运算。

将向量化的思想使用在逻辑回归算法上，尽可能减少for循环，而只使用矩阵运算。值得注意的是，算法最顶层的迭代训练的for循环是不能替换的。而每次迭代过程对J，dw，b的计算是可以直接使用矩阵运算。

### 3、**Vectorizing Logistic Regression**

在《神经网络与深度学习》课程笔记（2）中我们介绍过，整个训练样本构成的输入矩阵X的维度是（$n_x$，m），权重矩阵w的维度是（$n_x$，1），b是一个常数值，而整个训练样本构成的输出矩阵Y的维度为（1，m）。利用向量化的思想，所有m个样本的线性输出Z可以用矩阵表示：
$$
Z=w^TX+b
$$
在python的numpy库中可以表示为：

```python
Z = np.dot(w.T,X) + b
A = sigmoid(Z)

```

其中，w.T表示w的转置。

### 4、**Vectorizing Logistic Regression’s Gradient Output**

再来看逻辑回归中的梯度下降算法如何转化为向量化的矩阵形式。对于所有m个样本，dZ的维度是（1，m），可表示为：
$$
dZ=A-Y
$$
db可表示为：
$$
db =\frac{1}{m}\sum_{i=1}^m(dz^{(i)})
$$
对应的代码为：

```python
db = 1/m*np.sum(dZ)

```

dw可表示为：
$$
dw = \frac{1}{m}X.dZ^T
$$
对应的程序为：

```python
dw = 1/m*np.dot(X,dZ.T)
```

这样，我们把整个逻辑回归中的for循环尽可能用矩阵运算代替，对于单次迭代，梯度下降算法流程如下所示：

```python
Z = np.dot(w.T,X) + b
A = sigmoid(Z)
dZ = A-Y
dw = 1/m*np.dot(X,dZ.T)
db = 1/m*np.sum(dZ)

w = w - alpha*dw
b = b - alpha*db
```

其中，alpha是学习因子，决定w和b的更新速度。上述代码只是对单次训练更新而言的，外层还需要一个for循环，表示迭代次数。

### 5、**Broadcasting in Python**

下面介绍使用python的另一种技巧：广播（Broadcasting）。python中的广播机制可由下面四条表示：

- **让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分都通过在前面加1补齐**
- **输出数组的shape是输入数组shape的各个轴上的最大值**
- **如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，否则出错**
- **当输入数组的某个轴的长度为1时，沿着此轴运算时都用此轴上的第一组值**

简而言之，就是python中可以对不同维度的矩阵进行四则混合运算，但至少保证有一个维度是相同的。下面给出几个广播的例子：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%EF%BC%883%EF%BC%89/4-1.jpg)

在python程序中为了保证矩阵运算正确，可以使用reshape()函数来对矩阵设定所需的维度。这是一个很好且有用的习惯。

### 6、**A note on python/numpy vectors**

接下来我们将总结一些python的小技巧，避免不必要的code bug。

python中，如果我们用下列语句来定义一个向量：

```python
a = np.random.randn(5)
```

这条语句生成的a的维度是（5，）。它既不是行向量也不是列向量，我们把a叫做rank 1 array。这种定义会带来一些问题。例如我们对a进行转置，还是会得到a本身。所以，如果我们要定义（5，1）的列向量或者（1，5）的行向量，最好使用下来标准语句，避免使用rank 1 array。

```python
a = np.random.randn(5,1)
b = np.random.randn(1,5)
```

除此之外，我们还可以使用assert语句对向量或数组的维度进行判断，例如：

```python
assert(a.shape == (5,1))
```

assert会对内嵌语句进行判断，即判断a的维度是不是（5，1）的。如果不是，则程序在此处停止。使用assert语句也是一种很好的习惯，能够帮助我们及时检查、发现语句是否正确。

另外，还可以使用reshape函数对数组设定所需的维度：

```python
a.reshape((5,1))
```

### 7、**Summary**

本节课主要介绍了神经网络基础——python和向量化。在深度学习程序中，使用向量化和矩阵运算的方法能够大大提高运行速度，节省时间。以逻辑回归为例，我们将其算法流程包括梯度下降转换为向量化的形式。同时也介绍了python的相关编程方法和技巧。

