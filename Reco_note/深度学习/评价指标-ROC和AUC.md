#### 1. TPR和FPR基础概念

- 混淆矩阵中有着Positive、Negative、True、False的概念，其意义如下：
  - **称预测类别为1的为Positive（阳性），预测类别为0的为Negative（阴性）。**
  - **预测正确的为True（真），预测错误的为False（伪）。**
- 对上述概念进行组合，就产生了如下的混淆矩阵：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/CNN/19-17.png)

- **然后**，由此引出True Positive Rate（真阳率）、False Positive（伪阳率）两个概念：

$$
\begin{array}{l}\text {TPRate}=\frac{T P}{T P+F N} \\ \text {FPRate}=\frac{F P}{F P+T N}\end{array}
$$

- 仔细看这两个公式，发现其实TPRate就是TP除以TP所在的列，FPRate就是FP除以FP所在的列，二者意义如下：
  - **TPRate的意义是所有真实类别为1的样本中，预测类别为1的比例。**
  - **FPRate的意义是所有真实类别为0的样本中，预测类别为1的比例。**

#### 2. ROC曲线和AUC

- **ROC曲线的横轴为FPR，纵轴为TPR；而AUC为ROC曲线下的面积；示例图如下**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/CNN/19-18.jpeg)

- ROC曲线有4个关键点：
  - **点(0,0)：FPR=TPR=0；即TP=FP=0，表示分类器预测的都为负样本**
  - **点(1,1)：FPR=TPR=1；即TN=FN=0，表示分类器预测的都为正样本**
  - **点(0,1)：FPR=0，TPR=1；即FN=FP=0，表示最优分类器，所有样本正确分类**
  - **点(1,0)：FPR=1，TPR=0；即TN=TP=0，表示最差分类器，所有样本错误分类**
- ROC曲线越接近左上角，表示该分类器性能越好

- 举例如下：
- 首先对于硬分类器（例如SVM，NB），预测类别为离散标签，对于8个样本的预测情况如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/CNN/19-19.png)

- 得到的混淆矩阵如下：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/CNN/19-20.png" style="zoom:50%;" />

- 进而算得TPRate=3/4，FPRate=2/4，得到ROC曲线：最终得到AUC为0.625

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/CNN/19-21.png" style="zoom:50%;" />

- 对于LR等预测类别为概率的分类器，依然用上述例子，假设预测结果如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/CNN/19-22.png)

- 这时，需要设置阈值来得到混淆矩阵，不同的阈值会影响得到的TPRate，FPRate，如果阈值取0.5，小于0.5的为0，否则为1，那么我们就得到了与之前一样的混淆矩阵。其他的阈值就不再啰嗦了。依次使用所有预测值作为阈值，得到一系列TPRate，FPRate，描点，求面积，即可得到AUC。
- AUC的优势：**AUC的计算方法同时考虑了分类器对于正例和负例的分类能力，在样本不平衡的情况下，依然能够对分类器作出合理的评价。**
- 例如在反欺诈场景，设欺诈类样本为正例，正例占比很少（假设0.1%），如果使用准确率评估，把所有的样本预测为负例，便可以获得**99.9%的准确率**。但是如果使用AUC，把所有样本预测为负例，TPRate和FPRate同时为0（没有Positive），与(0,0) (1,1)连接，得出**AUC仅为0.5**，成功规避了样本不均匀带来的问题。

