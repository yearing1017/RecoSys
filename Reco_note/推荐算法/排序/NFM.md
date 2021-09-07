下图就是 NFM 的模型架构图，Bi-Interaction Pooling 层。那这个夹在 Embedding 层和 MLP 之间的层到底做了什么呢？

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/26-3.png" style="zoom:50%;" />

- Bi-Interaction Pooling Layer 翻译成中文就是“**两两特征交叉池化层**”。假设 **Vx 是所有特征域的 Embedding 集合**，那么特征交叉池化层的具体操作如下所示。

$$
f_{\mathrm{PI}}\left(V_{x}\right)=\sum_{i=1}^{n} \sum_{j=i+1}^{n} x_{i} \boldsymbol{v}_{i} \odot \boldsymbol{x}_{j} \boldsymbol{v}_{j}
$$

- 其中 ⊙ 运算代表两个向量的元素积（Element-wise Product）操作，即**两个长度相同的向量对应维相乘得到元素积向量**。其中，第 k 维的操作如下所示。

$$
\left(V_{i} \odot V_{j}\right)_{K}=v_{i k} v_{j k}
$$

- **在进行两两特征 Embedding 向量的元素积操作后，再求取所有交叉特征向量之和，我们就得到了池化层的输出向量**。接着，我们再把该向量输入上层的多层全连接神经网络，就能得出最后的预测得分。
- 总的来说，**NFM 并没有使用内积操作来进行特征 Embedding 向量的交叉，而是使用元素积的操作。在得到交叉特征向量之后，也没有使用 concatenate 操作把它们连接起来，而是采用了求和的池化操作，把它们叠加起来**。

> 内积：**a**·**b**=|**a**||**b**|·cosθ  两个向量对应位置相乘再求和
>
> 外积：**a**×**b**=**c**，其中|**c**|=|**a**||**b**|·sinθ，**c**的方向遵守右手定则
>
> 哈达玛积：元素积  两个长度相同的向量对应维相乘得到元素积向量

