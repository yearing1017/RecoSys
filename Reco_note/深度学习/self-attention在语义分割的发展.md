### 1. 前言

- Attention 机制在 NLP （自然语言处理）领域取得主导地位之后，近两年在 CV 领域也引来大多的研究。率先将之引入的是 Kaiming He 组的 Non-local；
- 2019年，在语义分割领域就有多篇高影响力文章，如 DANet，CCNet等
- 本篇文章从NLP中的self-attention出发，引出在**计算机视觉中的self-attention模块的工作原理，进而总结近两年语义分割中self-attention模块的发展过程**。

### 2. NLP的attention机制

- 以一个生活中常见的事来引出self-attention：**一个漫威迷想去图书馆进一步了解漫威时**

- 图书馆（source）里有很多**书（value）**，为了方便查找，我们给书做了**编号（key）**。
- 当我们想要**了解漫威（query）**的时候，我们就可以看看那些动漫、电影、甚至二战（美国队长）相关的书籍。为了提高效率，并不是所有的书都会仔细看，**针对漫威来说，动漫，电影相关的会看的仔细一些（权重高），但是二战的就只需要简单扫一下即可（权重低）**。当我们全部看完后就对漫威有一个全面的了解了。

- 以上便包含了self-attention中的全部内容。事实上，**self-attention可以分解为3步**：
  - **第一步：query 和 key 进行相似度计算，得到权值**
  - **第二步：将权值进行归一化，得到直接可用的权重**
  - **第三步：将权重和 value 进行加权求和，从而得出我们更应该关注那些value。**

### 3. 计算机视觉中的注意力机制

- 计算机视觉中的注意力机制的基本思想就是想让系统学会注意力——**能够忽略无关信息而关注重点信息。**
- 近几年来，深度学习与视觉注意力机制结合的研究工作，大多数是**集中于使用掩码(mask)来形成注意力机制。掩码的原理在于通过另一层新的权重，将图片数据中关键的特征标识出来**，通过学习训练，让深度神经网络学到每一张新图片中需要关注的区域，也就形成了注意力。

- 注意力机制一种是**软注意力(soft attention)**，另一种则是**强注意力(hard attention)**。

- **软注意力的关键点**在于：
  - **更关注区域或者通道**，而且软注意力是确定性的注意力，学习完成后直接可以通过网络生成；
  - 最关键的是**软注意力是可微**的，这是一个非常重要的地方。**可以微分的注意力就可以通过神经网络算出梯度并且前向传播和后向反馈来学习得到注意力的权重。**
- **强注意力与软注意力不同点**在于：
  - **强注意力是更加关注点，也就是图像中的每个点都有可能延伸出注意力**，同时强注意力是一个随机的预测过程，更强调动态变化；
  - 强注意力是一个**不可微的注意力**，训练过程往往通过**增强学习(reinforcement learning)**来完成的。

- 在计算机视觉中，很多领域的相关工作(例如，分类、检测、分割、生成模型、视频处理等)都在使用Soft Attention，这些方法**共同的部分都是利用相关特征学习权重分布，再用学出来的权重施加在特征之上进一步提取相关知识**。
- 不过施加权重的方式略有差别，可以总结如下：
  - 加权可以作用在原图上；
  - 加权可以作用在**空间尺度**上，给不同空间区域加权；
  - 加权可以作用在**Channel尺度**上，给不同通道特征加权；
  - 加权可以作用在**不同时刻历史特征**上，结合循环结构添加权重，例如机器翻译，或者视频相关的工作。
- 本文写的**self-attention机制就属于软注意力的一种。**

### 4. 计算机视觉中self-attention

#### 4.1 Self-attention机制

- Self-Attention是从NLP中借鉴过来的思想，因此仍然保留了Query, Key和Value等名称；
- 下图是self-attention的基本结构，feature maps是由基本的深度卷积网络得到的特征图，如ResNet、Xception等，这些**基本的深度卷积网络被称为backbone**，通常**将最后ResNet的两个下采样层去除使获得的特征图是原输入图像的1/8大小。**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/8-1.png)

- Self-attention结构**自上而下分为三个分支，分别是query、key和value**。计算时通常分为三步

  - 第一步是将query和每个key进行**相似度计算得到权重**，常用的相似度函数有点积，拼接等；

  - 第二步一般是使用一个softmax函数**对这些权重进行归一化**；

  - 第三步将权重和相应的键值value进行**加权求和**得到最后的attention。

- self-attention的代码如下：

```python
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
 
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) 
        # B X C X (N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
 
        out = self.gamma*out + x
        return out,attention
```

- 假设feature maps的大小$B \times C \times W \times H$
- 在初始化函数中，定义了**三个1×1卷积，分别是query_conv ， key_conv 和 value_conv**：
  - 在query_conv卷积中，输入为$B×C×W×H$，输出为$B×C/8×W×H$；
  - 在key_conv卷积中，输入为$B×C×W×H$，输出为$B×C/8×W×H$；
  - 在value_conv卷积中，输入为$B×C×W×H$，输出为$B×C×W×H$。 

- 在forward函数中，定义了self-attention的具体步骤:

- **步骤1：**

  ```python
  proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
  ```

- proj_query**本质上就是卷积，只不过加入了reshape的操作**。首先是对输入的feature map进行query_conv卷积，输出为$B×C/8×W×H$；

- **view函数是改变了输出的维度**，就单张feature map而言，就是将W×H大小拉直，变为1×(W×H)大小；就batchsize大小而言，输出就是$B×C/8×(W×H)$；

- **permute函数则对第二维和第三维进行倒置，**输出为$B×(W×H)×C/8$。

- **proj_query中的第i行表示第i个像素位置上所有通道的值，如下图所示：**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/8-2.png" style="zoom:50%;" />

- proj_key与proj_query相似，**只是没有最后一步倒置**，输出为$B×C/8×(W×H)$。
- **proj_key中的第j列表示第j个像素位置上所有通道的值。如下图**

```python
proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)
```

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/8-3.png" style="zoom:50%;" />

- **步骤二：**

```python
energy =  torch.bmm(proj_query,proj_key)
```

- 将batch_size中的**每一对proj_query和proj_key分别进行矩阵相乘**，输出为$B×(W×H)×(W×H)$。

- Energy中的第(i,j)元素是将proj_query中的第i行与proj_key中的第j行点乘得到。
- 这个步骤的意义是**energy中第(i,j)位置的元素是指输入特征图第j个元素对第i个元素的影响，从而实现全局上下文任意两个元素的依赖关系。**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/8-4.png" style="zoom:50%;" />

- **步骤三：**

```python
attention = self.softmax(energy)
```

- 这一步是将energe进行softmax归一化，是**对行的归一化**。
- 归一化后每行的之和为1，对于(i,j)位置即可理解为第j位置对i位置的权重，相同的j对i位置的权重之和为1，此时得到attention_map。

```python
proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)
```

- proj_value和proj_query与proj_key一样，只是输入为$B×C×W×H$，输出同样为$B×C×(W×H)$

- 从self-attention结构图中可以知道**proj_value是与attention_map进行矩阵相乘**，即下面两行代码。

```python
out = torch.bmm(proj_value,attention.permute(0,2,1) )
out = out.view(m_batchsize,C,width,height)
```

- 在对proj_value与attention_map点乘之前，**先对attention进行转置**。这是**由于attention中每一行的权重之和为1，是原特征图第j个位置对第i个位置的权重，将其转置之后，每一列之和为1**；

- proj_value的每一行与attention中的每一列点乘，将权重施加于proj_value上，输出为$B×C×(W×H)$。

- 如下图：可理解为，**proj_value图第i行为第i个像素位置所有通道的值，attention转置之后，第j列表示第j个像素位置的值对第i像素位置的值的权重，两者相乘，相当于施加了权重。**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/8-5.png" style="zoom:50%;" />

- 对输出加权：

```python
out = self.gamma*out + x
```

- 这一步是**对attention之后的out进行加权**，x是原始的特征图，将其叠加在原始特征图上。
- Gamma是经过学习得到的，初始gamma为0，输出即原始特征图，随着学习的深入，在原始特征图上增加了加权的attention，得到特征图中**任意两个位置的全局依赖关系。**

#### 4.2 Non-local->DANet->CCNet

- 近年来，语义分割的模型几乎都是基于FCN框架，**利用Dilated Conv、Up sample、skip connection等各种组件来变化**。
- 但是这种模型有很大的问题：**无法对不同的像素之间的关系进行显式的建模**，像素之间的唯一联系就是`感受野`的重叠。**从本质上来说，这种语义分割框架是一种Dense prediction, 是分立的对每个像素进行预测。**

- 由此，**对语义分割中的不同像素/上下文之间的关系建模**就非常重要，近期的研究中，很多论文着眼于`context`和`relationship`，基于self-attention机制去捕获上下文关系的文章是影响力较强的一个分支。

- 【1】**Non-local：Non-LocalNeural Network，CVPR，2018.**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/8-6.png" style="zoom:50%;" />

- 由图示可得知：**non-local的原理与self-attention运行原理一样**。这里不再详述。
- 想要说明的是这里 θ，φ，g对效果影响不大。在做过的语义分割实验中，甚至可以考虑将 θ，φ，g转换省略，直接用 x 本身计算，而把 1x1 卷积放在模块之前，这样的效果也不逊色。当然，不同的任务应该对应于不同的最优选项。

- **non-local 的缺陷所在**：
  - 只涉及到了位置注意力模块，而没有涉及常用的通道注意力机制
  - 如果特征图较大，那么两个$(batch,h \times w,512)$矩阵乘是非常耗内存和计算量

- 对于这两个缺点，DANet和CCNet分别给出了他们的解决方案

- 【2】**DANet：DualAttention Network for Scene Segmentation，CVPR2019**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/8-7.png)

- DANet结构如上图，而在下图空间与通道的self-attention机制具体结构中可以看到，**在spatial维度与non-local是完全一样的思路，利用self-attention结构建立全局上下文关系。**
- 不仅如此，这篇文章还**把self-attention的思想拓展到了channel维度，由于不同的channel代表不同的滤波器响应，在高层特征中可以代表不同的语义概念，作者希望利用channel-wise attention建立不同语义类别之间的依赖关系。**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/8-8.png" style="zoom:50%;" />

- 通道维度上的操作。同样是Q，K，V三个特征图，只不过Q，K reshape之后与之前non-local的相乘顺序互换了一下。所以，图中最终乘出来的X的大小为C×C。

- 【3】**CCNet：Criss-CrossAttention for Semantic Segmentation，ICCV2019**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/8-9.png" style="zoom:50%;" />

- 上图（a）是Non-local，它计算当前点与所有像素点的相似性。
- 上图（b）即为CCNet的改进。可以看到，与non-local计算当前点与所有像素点的相似性不同，**CCNet只计算当前 $x_i$ 周围十字型区域像素 $x_j$ 与它的相关性。**当然，我们**想要获取的是所有像素与 $x_i$ 的相关性，**于是作者将这个过程进行堆叠，并且通过实验发现，**只需堆叠两次即可覆盖所有点，并超越non-local的效果**。

- 有关详细的论文笔记：[链接](http://yearing1017.cn/2020/03/26/CCNet-paper/)

