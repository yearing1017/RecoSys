#### 欲解决的问题

阿里的Embedding方法EGES（Enhanced Graph Embedding with Side Information），其**基本思想是在DeepWalk生成的graph embedding基础上引入补充信息**

如果单纯使用用户行为生成的物品相关图，固然可以生成物品的embedding，但是如果遇到新加入的物品，或者没有过多互动信息的长尾物品，推荐系统将出现严重的冷启动问题。**为了使“冷启动”的商品获得“合理”的初始Embedding，阿里团队通过引入了更多补充信息来丰富Embedding信息的来源，从而使没有历史行为记录的商品获得Embedding**

生成Graph embedding的第一步是生成物品关系图，通过用户行为序列可以生成物品相关图，利用相同属性、相同类别等信息，也可以通过这些相似性建立物品之间的边，从而生成基于内容的knowledge graph。而基**于knowledge graph生成的物品向量可以被称为补充信息（side information）embedding向量**，当然，根据补充信息类别的不同，可以有多个side information embedding向量

**side-information也代指类似item的一级分类、二级分类等信息**

#### Graph Embedding with Side Information（GES）

该方案**增加 item 的额外信息（例如category, brand, price等）丰富 item 表征力度**。根据 **EGES**的算法框架可知：

（1）**item 和 side information（例如category, brand, price等） 的 Embedding 是通过 word2vec 算法一起训练得到的。**如果分开训练，得到的item_embedding和category_embedding, brand_embedding, price_embedding不在一个向量空间中，做运算无意义。

> 即：通过 DeepWalk 方案得到 item 的游走序列，同时得到对应的category（brand, price）序列。然后将所有序列数据放到word2vec模型中进行训练。

（2）**针对每个 item，将得到：item_embedding，category_embedding，brand_embedding，price_embedding 等 embedding 信息。将这些 embedding 信息求均值来表示该 item**。即：
$$
H_{v}=\frac{1}{n+1} \sum_{s=0}^{n} W_{v}^{s}
$$

- **BGE 与 GES  使用w2v训练的不同**  

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/27-7.jpg" style="zoom:50%;" />

#### EGES如何将多个Embedding融合

**如何融合一个物品的多个embedding向量，使之形成物品最后的embedding呢？**

**最简单的方法是GES：在深度神经网络中加入average pooling层将不同embedding平均起来，阿里在此基础上进行了加强，对每个embedding加上了权重**，如图7所示，对每类特征对应的Embedding向量，分别赋予了权重a0，a1…an。

图中的Hidden Representation层就是对不同Embedding进行加权平均操作的层，得到加权平均后的Embedding向量后，再直接输入softmax层，这样**通过梯度反向传播，就可以求的每个embedding的权重**ai(i=0…n)

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/27-5.png" style="zoom:33%;" />

- 注：这里解释权重的生成：**让模型学出来的**
- 上图的模型，其实是**加强版的word2vec模型；在隐层前多学了一个权重的参数**

在实际的模型中，阿里采用了 $e^{a_j}$ 而不是$a_j$作为相应embedding的权重，一是避免权重为0，二是因为 $ e^{a_j}$在梯度下降过程中有良好的数学性质。 **最终学到的embedding向量融合如下：**
$$
H_{v}=\frac{\sum_{j=0}^{n} e^{a_{v}^{j}} W_{v}^{j}}{\sum_{j=0}^{n} e^{a_{v}^{j}}}
$$
阿里的EGES并没有过于复杂的理论创新，但给出一个工程性的结合多种Embedding的方法，**降低了某类Embedding缺失造成的冷启动问题，是实用性极强的Embedding方法**

#### 模型

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/27-6.png" style="zoom:33%;" />

- 输入：
  - 上图的Sparse Features代表 item 和 side information 的ID信息；
  - Dense Embeddings 表示 item 和 side information 的 embedding 信息；
- 输出：
  - item 和 side information 的 Embedding向量 
  - 学到的权重矩阵