#### Log - Likelihood Ratio

#### 计算方式

在计算两个事件(例如推荐系统中的点击行为)的 LLR 值来做相似度衡量的时候，可以通过两个事件的计数来计算事件之间的相似度。

以推荐系统中物品相似度的计算为例来学习 LLR 的计算方式：

- 有两个物品i和j，对应的浏览事件（也可等效于喜欢）形式化如下：
  - $k_{11}$ 表示同时浏览了物品 i 和物品 j 的用户
  - $k_{12}$ 表示浏览了物品 i，但没有浏览物品 j 的用户
  - $k_{21}$ 表示浏览了物品 j，但没有浏览物品 i 的用户
  - $k_{22}$ 表示没有浏览物品 i， 并且没有浏览物品 j 的用户

上述转为矩阵形式：

|                 | 浏览 item i | 没有浏览 item i |
| :-------------: | :---------: | :-------------: |
|   浏览 item j   |  $k_{11}$   |    $k_{21}$     |
| 没有浏览 item j |  $k_{12}$   |    $k_{22}$     |

Item i 和 item j 的 LLR 的 计算公式：
$$
S=2 \times\left(H_{m}-H_{c}-H_{r}\right)
$$
$H_m$表示上述矩阵的矩阵熵，总体的计算公式如下：
$$
H_{m}=-\left(\frac{k_{11}}{N} \log \left(\frac{k_{11}}{N}\right)+\frac{k_{12}}{N} \log \left(\frac{k_{12}}{N}\right)+\frac{k_{21}}{N} \log \left(\frac{k_{21}}{N}\right)+\frac{k_{22}}{N} \log \left(\frac{k_{22}}{N}\right)\right)
$$
$H_r$表示每行相加计算得到的信息熵，相关的具体计算公式如下：
$$
H_{r}=-\left(\frac{k_{11}+k_{12}}{N} \log \left(\frac{k_{11}+k_{12}}{N}\right)+\frac{k_{21}+k_{22}}{N} \log \left(\frac{k_{21}+k_{22}}{N}\right)\right)
$$
$H_c$表示矩阵的每一列相加计算得到的信息熵，相关的具体计算公式如下：
$$
H_{c}=-\left(\frac{k_{11}+k_{21}}{N} \log \left(\frac{k_{11}+k_{21}}{N}\right)+\frac{k_{21}+k_{22}}{N} \log \left(\frac{k_{21}+k_{22}}{N}\right)\right)
$$
其中：$N = k_{11} + k_{12} + k_{21} + k_{22}$

#### 代码实现

```scala
object LogLikelihood {
  def xLogX(x: Long): Double = if (x == 0) 0.0 else x * Math.log(x)

  def entropy(elements: Long*): Double = {
    var sum: Long = 0
    var result: Double = 0.0
    for (element <- elements) {
      result += xLogX(element)
      sum += element
    }
    xLogX(sum) - result
  }

  def logLike(item1Count: Long, item2Count: Long, common: Long, all: Long): Double = {
    val k11 = common // 同时喜欢item1和item2的人数
    val k12 = item1Count - common // 喜欢item2不喜欢item1的人数
    val k21 = item2Count - common // 喜欢item1不喜欢item2的人数
    val k22 = all - item1Count - item2Count + common // 不喜欢item1也不喜欢item2的人数
    val rowEntropy = entropy(k11 + k12, k21 + k22)
    val columnEntropy = entropy(k11 + k21, k12 + k22)
    val matrixEntropy = entropy(k11, k12, k21, k22)
    val sim = Math.max(0.0, 2 * (rowEntropy + columnEntropy - matrixEntropy))

    sim
  }
}
```
