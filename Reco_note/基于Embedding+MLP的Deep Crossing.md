### 前言

- 本文记录了《深度学习推荐系统实战》**第17讲Embedding+MLP模型**的要点
- 课程地址：[深度学习推荐系统实战](https://time.geekbang.org/column/intro/349)

### 1. Embedding+MLP 模型的结构

- 图 1 展示的就是微软在 2016 年提出的深度学习模型 Deep Crossing，微软把它用于广告推荐这个业务场景上。它是一个经典的 Embedding+MLP 模型结构，我们可以看到，**Deep Crossing 从下到上可以分为 5 层，分别是 Feature 层、Embedding 层、Stacking 层、MLP 层和 Scoring 层**。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/19-1.jpg" style="zoom:50%;" />

- **Feature 层**：Feature 层也叫做输入特征层，它处于 Deep Crossing 的最底部，作为整个模型的输入。仔细看图 1 的话，**Feature#1 向上连接到了 Embedding 层，而 Feature#2 就直接连接到了更上方的 Stacking 层**。这是怎么回事呢？原因就在于 **Feature#1 代表的是类别型特征经过 One-hot 编码后生成的特征向量，而 Feature#2 代表的是数值型特征**。**One-hot 特征太稀疏了，不适合直接输入到后续的神经网络中进行训练，所以我们需要通过连接到 Embedding 层的方式，把这个稀疏的 One-hot 向量转换成比较稠密的 Embedding 向量。**

- **Embedding 层**：Embedding 层就是**为了把稀疏的 One-hot 向量转换成稠密的 Embedding 向量**而设置的，需要注意的是，**Embedding 层并不是全部连接起来的，而是每一个特征对应一个 Embedding 层，不同 Embedding 层之间互不干涉**。那 Embedding 层的内部结构到底是什么样子的呢？**Embedding 层的结构就是 Word2vec 模型中从输入神经元到隐层神经元的部分**（如图 2 红框内的部分）。参照下面的示意图，我们可以看到，这部分就是**一个从输入层到隐层之间的全连接网络**。
- 一般来说，**Embedding 向量的维度应远小于原始的稀疏特征向量**，按照经验，**几十到上百维就能够满足需求，这样它才能够实现从稀疏特征向量到稠密特征向量的转换**。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/19-2.jpg" style="zoom:50%;" />

- **Stacking 层**：Stacking 层中文名是堆叠层，我们也经常叫它连接（Concatenate）层。它的作用比较简单，就是**把不同的 Embedding 特征和数值型特征拼接在一起，形成新的包含全部特征的特征向量**。

- MLP 层就是的多层神经网络层，在图 1 中指的是 Multiple Residual Units 层，中文叫多层残差网络。微软在实现 Deep Crossing 时针对特定的问题选择了残差神经元，但事实上，神经元的选择有非常多种，比如以 Sigmoid 函数为激活函数的神经元，以及使用 tanh、ReLU 等其他激活函数的神经元。我们具体选择哪种是一个调参的问题，一般来说，**ReLU 最经常使用在隐层神经元上，Sigmoid 则多使用在输出神经元**，实践中也可以选择性地尝试其他神经元，根据效果作出最后的决定。
- 不管选择哪种神经元，**MLP 层的特点是全连接，就是不同层的神经元两两之间都有连接**。就像图 3 中的两层神经网络一样，**它们两两连接，只是连接的权重会在梯度反向传播的学习过程中发生改变**。
- **MLP 层的作用是让特征向量不同维度之间做充分的交叉，让模型能够抓取到更多的非线性特征和组合特征**的信息，这就使深度学习模型在表达能力上较传统机器学习模型大为增强。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/19-3.jpg" style="zoom:33%;" />

- **Scoring 层**，它也被称为输出层。虽然深度学习模型的结构可以非常复杂，但最终我们要预测的目标就是一个分类的概率。**如果是点击率预估，就是一个二分类问题，那我们就可以采用逻辑回归作为输出层神经元，而如果是类似图像分类这样的多分类问题，我们往往在输出层采用 softmax 这样的多分类模型**。

- Embedding+MLP 的五层结构用一句话总结就是，**对于类别特征，先利用 Embedding 层进行特征稠密化，再利用 Stacking 层连接其他特征，输入 MLP 的多层结构，最后用 Scoring 层预估结果**。

### 2. Embedding+MLP 模型的实战

#### 2.1 特征选择和模型设计

- 秉着“**类别型特征 Embedding 化，数值型特征直接输入 MLP**”的原则，我们选择 movieId、userId、movieGenre、userGenre 作为 Embedding 化的特征，选择物品和用户的统计型特征作为直接输入 MLP 的数值型特征，具体的特征选择你可以看看下面的表格：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/19-4.jpg" style="zoom:50%;" />

#### 2.2 Embedding+MLP 模型的 TensorFlow 实现

- 选择好特征后，就是 MLP 部分的模型设计了。我们选择了一个三层的 MLP 结构，其中前两层是 128 维的全连接层。我们这里**采用好评 / 差评标签作为样本标签，因此要解决的是一个类 CTR 预估的二分类问题，对于二分类问题，我们最后一层采用单个 sigmoid 神经元作为输出层**就可以了。

##### 2.2.1 导入 TensorFlow 包

```python

import tensorflow as tf


TRAIN_DATA_URL = "file:///Users/zhewang/Workspace/SparrowRecSys/src/main/resources/webroot/sampledata/modelSamples.csv"
samples_file_path = tf.keras.utils.get_file("modelSamples.csv", TRAIN_DATA_URL)
```

##### 2.2.2 载入训练数据

- 利用 TensorFlow 自带的 CSV 数据集的接口载入训练数据。注意这里有两个比较重要的参数，一个是 **label_name，它指定了 CSV 数据集中的标签列**。另一个是 batch_size，它指定了训练过程中，一次输入几条训练数据进行梯度下降训练。载入训练数据之后，我们把它们分割成了测试集和训练集。

```python
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value="?",
        num_epochs=1,
        ignore_errors=True)
    return dataset

# sample dataset size 110830/12(batch_size) = 9235
raw_samples_data = get_dataset(samples_file_path)

test_dataset = raw_samples_data.take(1000)
train_dataset = raw_samples_data.skip(1000)
```

##### 2.2.3 载入类别型特征

- 用到的类别型特征主要有这三类，分别是 genre、userId 和 movieId。
- **在载入 genre 类特征时，我们采用了 tf.feature_column.categorical_column_with_vocabulary_list 方法把字符串型的特征转换成了 One-hot 特征**。在这个转换过程中我们需要用到一个词表，你可以看到我在开头就**定义好了包含所有 genre 类别的词表 genre_vocab**。
- 在转换 userId 和 movieId 特征时，我们又使用了 tf.feature_column.categorical_column_with_identity 方法把 ID 转换成 One-hot 特征，这个方法不用词表，它会**直接把 ID 值对应的那个维度置为 1**。比如，我们输入这个方法的 movieId 是 340，总的 movie 数量是 1001，使用这个方法，就会把这个 1001 维的 One-hot movieId 向量的第 340 维置为 1，剩余的维度都为 0。
- **为了把稀疏的 One-hot 特征转换成稠密的 Embedding 向量，我们还需要在 One-hot 特征外包裹一层 Embedding 层**，你可以看到 tf.feature_column.embedding_column(movie_col, 10) 方法完成了这样的操作，它在**把 movie one-hot 向量映射到了一个 10 维的 Embedding 层上**。

```python

genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 								'Western', 'Documentary','Sci-Fi', 'Drama', 'Thriller','Crime', 'Fantasy', 								'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

GENRE_FEATURES = {
    'userGenre1': genre_vocab,
    'userGenre2': genre_vocab,
    'userGenre3': genre_vocab,
    'userGenre4': genre_vocab,
    'userGenre5': genre_vocab,
    'movieGenre1': genre_vocab,
    'movieGenre2': genre_vocab,
    'movieGenre3': genre_vocab
}

categorical_columns = []
for feature, vocab in GENRE_FEATURES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    emb_col = tf.feature_column.embedding_column(cat_col, 10)
    categorical_columns.append(emb_col)


movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
categorical_columns.append(movie_emb_col)


user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
categorical_columns.append(user_emb_c
```

##### 2.2.4 数值型特征的处理

- 直接把特征值输入到 MLP 内，然后把特征逐个声明为 tf.feature_column.numeric_column 就可以了，不需要经过其他的特殊处理。

```python
numerical_columns = [tf.feature_column.numeric_column('releaseYear'),
                   tf.feature_column.numeric_column('movieRatingCount'),
                     tf.feature_column.numeric_column('movieAvgRating'),
                     tf.feature_column.numeric_column('movieRatingStddev'),
                     tf.feature_column.numeric_column('userRatingCount'),
                     tf.feature_column.numeric_column('userAvgRating'),
                     tf.feature_column.numeric_column('userRatingStddev')]
```

##### 2.2.5 定义模型结构

- 直接**利用 DenseFeatures 把类别型 Embedding 特征和数值型特征连接在一起形成稠密特征向量，然后依次经过两层 128 维的全连接层，最后通过 sigmoid 输出神经元产生最终预估值**。

```python
preprocessing_layer = tf.keras.layers.DenseFeatures(numerical_columns + categorical_columns)

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
```

##### 2.2.6 定义模型训练相关的参数

- 设置模型的损失函数，梯度反向传播的优化方法，以及模型评估所用的指标。关于损失函数，我们使用的是二分类问题最常用的二分类交叉熵，优化方法使用的是深度学习中很流行的 adam，最后是评估指标，使用了准确度 accuracy 作为模型评估的指标。

```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
```

##### 2.2.7 模型的训练和评估

- 调用 fit 函数，然后使用 evaluate 函数在测试集上进行评估。不过，这里我们要注意一个参数 epochs，它代表了模型训练的轮数，一轮代表着使用所有训练数据训练一遍，epochs=10 代表着训练 10 遍。

```python
model.fit(train_dataset, epochs=10)

test_loss, test_accuracy = model.evaluate(test_dataset)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy)
      
'''
Epoch 1/10
8236/8236 [==============================] - 20s 2ms/step - loss: 2.7379 - accuracy: 0.5815
Epoch 2/10
8236/8236 [==============================] - 21s 3ms/step - loss: 0.6397 - accuracy: 0.6659
Epoch 3/10
8236/8236 [==============================] - 21s 3ms/step - loss: 0.5550 - accuracy: 0.7179
Epoch 4/10
8236/8236 [==============================] - 21s 2ms/step - loss: 0.5209 - accuracy: 0.7431
Epoch 5/10
8236/8236 [==============================] - 21s 2ms/step - loss: 0.5010 - accuracy: 0.7564
Epoch 6/10
8236/8236 [==============================] - 20s 2ms/step - loss: 0.4866 - accuracy: 0.7641
Epoch 7/10
8236/8236 [==============================] - 20s 2ms/step - loss: 0.4770 - accuracy: 0.7702
Epoch 8/10
8236/8236 [==============================] - 21s 2ms/step - loss: 0.4688 - accuracy: 0.7745
Epoch 9/10
8236/8236 [==============================] - 20s 2ms/step - loss: 0.4633 - accuracy: 0.7779
Epoch 10/10
8236/8236 [==============================] - 20s 2ms/step - loss: 0.4580 - accuracy: 0.7800
1000/1000 [==============================] - 1s 1ms/step - loss: 0.5037 - accuracy: 0.7473

Test Loss 0.5036991238594055, Test Accuracy 0.747250020503997
'''
```

