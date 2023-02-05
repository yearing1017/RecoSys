#### 前言

- 本文记录了RDD常用算子及相关例子，保持更新中

#### Transformations 和 Action 算子

根据数据处理方式的不同将**Transformations算子整体上分为 Value 类型、双 Value 类型和 Key-Value类型**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/TA.jpg)

#### **Value类型**

- **map**
  - Applies a transformation function on each item of the RDD and returns the result as a new RDD.
  - 将处理的数据逐条进行映射转换，这里的转换可以是类型的转换，也可以是值的转换。

```scala
val dataRDD: RDD[Int] = sparkContext.makeRDD(List(1,2,3,4))
val dataRDD1: RDD[Int] = dataRDD.map(num => {num * 2})
val dataRDD2: RDD[String] = dataRDD1.map(num => {"" + num})
val a = sc.parallelize(List("dog", "salmon", "salmon", "rat", "elephant"), 3)
val b = a.map(_.length)
val c = a.zip(b)
c.collect

res0: Array[(String, Int)] = Array((dog,3), (salmon,6), (salmon,6), (rat,3), (elephant,8))
```



- **mapPartitions**
  - This is a specialized map that is called only once for each partition. The entire content of the respective partitions is available as a sequential stream of values via the input argument (*Iterarator[T]*). The custom function must return yet another *Iterator[U]*. The combined result iterators are automatically converted into a new RDD. Please note, that the tuples (3,4) and (6,7) are missing from the following result due to the partitioning we chose.
  - 将待处理的数据以分区为单位发送到计算节点进行处理，这里的处理是指可以进行任意的处 理，哪怕是过滤数据。

```scala
val dataRDD1: RDD[Int] = dataRDD.mapPartitions(
   datas => {
   	datas.filter(_==2)
   }
)
```

- **map和mapPartitions的区别**
  - 数据处理角度：Map 算子是分区内一个数据一个数据的执行，类似于串行操作。而 mapPartitions 算子 是以分区为单位进行批处理操作。
  - 功能角度：Map 算子主要目的将数据源中的数据进行转换和改变。但是不会减少或增多数据。 MapPartitions 算子需要传递一个迭代器，返回一个迭代器，没有要求的元素的个数保持不变， 所以可以增加或减少数据
  - Map 算子因为类似于串行操作，所以性能比较低，而是 mapPartitions 算子类似于批处 理，所以性能较高。但是 mapPartitions 算子会长时间占用内存，那么这样会导致内存可能 不够用，出现内存溢出的错误。所以在内存有限的情况下，不推荐使用。使用 map 操作。
- **mapPartitionsWithIndex**
  - 将待处理的数据以分区为单位发送到计算节点进行处理，这里的处理是指可以进行任意的处 理，哪怕是过滤数据，在处理时同时可以获取当前分区索引。

```scala
val dataRDD1 = dataRDD.mapPartitionsWithIndex(
   (index, datas) => {
    	datas.map(index, _)
   }
)
```

- **flatMap**
  - 将处理的数据进行扁平化后再进行映射处理，所以算子也称之为扁平映射
  - Similar to *map*, but allows emitting more than one item in the map function.

```scala
val a = sc.parallelize(1 to 10, 5)
a.flatMap(1 to _).collect

//res47: Array[Int] = Array(1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

sc.parallelize(List(1, 2, 3), 2).flatMap(x => List(x, x, x)).collect
//res85: Array[Int] = Array(1, 1, 1, 2, 2, 2, 3, 3, 3)
```

- **groupBy**
  - 将数据根据指定的规则进行分组, 分区默认不变，但是数据会被打乱重新组合，我们将这样的操作称之为 shuffle。极限情况下，数据可能被分在同一个分区中。一个组的数据在一个分区中，但是并不是说一个分区中只有一个组

```scala
val a = sc.parallelize(1 to 9, 3)
a.groupBy(x => { if (x % 2 == 0) "even" else "odd" }).collect

//res42: Array[(String, Seq[Int])] = Array((even,ArrayBuffer(2, 4, 6, 8)), (odd,ArrayBuffer(1, 3, 5, 7, 9)))

val a = sc.parallelize(1 to 9, 3)
def myfunc(a: Int) : Int ={a % 2}
a.groupBy(myfunc).collect

//res3: Array[(Int, Seq[Int])] = Array((0,ArrayBuffer(2, 4, 6, 8)), (1,ArrayBuffer(1, 3, 5, 7, 9)))
```

- **filter**
  - 将数据根据指定的规则进行筛选过滤，符合规则的数据保留，不符合规则的数据丢弃。
  - 当数据进行筛选过滤后，分区不变，但是分区内的数据可能不均衡，生产环境下，可能会出现数据倾斜。

```scala
val a = sc.parallelize(1 to 10, 3)
val b = a.filter(_ % 2 == 0)
b.collect
//res3: Array[Int] = Array(2, 4, 6, 8, 10)
```

- **sample**
  - 根据指定的规则从数据集中抽取数据

```scala
val dataRDD = sparkContext.makeRDD(List(1,2,3,4),1)

// 抽取数据不放回（伯努利算法）
// 伯努利算法：又叫 0、1 分布。例如扔硬币，要么正面，要么反面。
// 具体实现：根据种子和随机算法算出一个数和第二个参数设置几率比较，小于第二个参数要，大于不要
// 第一个参数：抽取的数据是否放回，false：不放回
// 第二个参数：抽取的几率，范围在[0,1]之间,0：全不取；1：全取；
// 第三个参数：随机数种子

val dataRDD1 = dataRDD.sample(false, 0.5)

// 抽取数据放回（泊松算法）
// 第一个参数：抽取的数据是否放回，true：放回；false：不放回
// 第二个参数：重复数据的几率，范围大于等于 0.表示每一个元素被期望抽取到的次数
// 第三个参数：随机数种子

val dataRDD2 = dataRDD.sample(true, 2)
```

- **distinct**
  - def distinct(numPartitions: Int)(implicit ord: Ordering[T] = null): RDD[T]
  - 将数据集中重复的数据去重

```scala
val c = sc.parallelize(List("Gnu", "Cat", "Rat", "Dog", "Gnu", "Rat"), 2)
c.distinct.collect
//res6: Array[String] = Array(Dog, Gnu, Cat, Rat)
val a = sc.parallelize(List(1,2,3,4,5,6,7,8,9,10))
a.distinct(2).partitions.length
//res16: Int = 2
a.distinct(3).partitions.length
//res17: Int = 3
```

- **coalesce**
  - Coalesces the associated data into a given number of partitions. 
  - 根据数据量缩减分区，用于大数据集过滤后，提高小数据集的执行效率,当 spark 程序中，存在过多的小任务的时候，可以通过 coalesce 方法，收缩合并分区，减少分区的个数，减小任务调度成本

```scala
val y = sc.parallelize(1 to 10, 10)
val z = y.coalesce(2, false)
z.partitions.length
res9: Int = 2
```


- **repartition**
  - repartition(numPartitions) is simply an abbreviation for coalesce(numPartitions, shuffle = true).
  - This function changes the number of partitions to the number specified by the numPartitions parameter 
  - 该操作内部其实执行的是 coalesce 操作，参数 shuffle 的默认值为 true。无论是将分区数多的 RDD 转换为分区数少的 RDD，还是将分区数少的 RDD 转换为分区数多的 RDD，repartition 操作都可以完成，因为无论如何都会经 shuffle 过程。

```scala
val rdd = sc.parallelize(List(1, 2, 10, 4, 5, 2, 1, 1, 1), 3)
rdd.partitions.length
//res2: Int = 3
val rdd2 = rdd.repartition(5)
rdd2.partitions.length
//res6: Int = 5
```

- **sortBy**
  - This function sorts the input RDD's data and stores it in a new RDD. The first parameter requires you to specify a function which maps the input data into the key that you want to sortBy. The second parameter (optional) specifies whether you want the data to be sorted in ascending or descending order.
  - 该操作用于排序数据。在排序之前，可以将数据通过 f 函数进行处理，之后按照 f 函数处理 的结果进行排序，默认为升序排列。排序后新产生的 RDD 的分区数与原 RDD 的分区数一 致。中间存在 shuffle 的过程

```scala
val y = sc.parallelize(Array(5, 7, 1, 3, 2, 1))
y.sortBy(c => c, true).collect
//res101: Array[Int] = Array(1, 1, 2, 3, 5, 7)

y.sortBy(c => c, false).collect
//res102: Array[Int] = Array(7, 5, 3, 2, 1, 1)

val z = sc.parallelize(Array(("H", 10), ("A", 26), ("Z", 1), ("L", 5)))
z.sortBy(c => c._1, true).collect
//res109: Array[(String, Int)] = Array((A,26), (H,10), (L,5), (Z,1))

z.sortBy(c => c._2, true).collect
//res108: Array[(String, Int)] = Array((Z,1), (L,5), (H,10), (A,26))
```

- **keyBy**
  - Constructs two-component tuples (key-value pairs) by applying a function on each data item. The result of the function becomes the key and the original data item becomes the value of the newly created tuples.

```scala
val a = sc.parallelize(List("dog", "salmon", "salmon", "rat", "elephant"), 3)
val b = a.keyBy(_.length)
b.collect

//res26: Array[(Int, String)] = Array((3,dog), (6,salmon), (6,salmon), (3,rat), (8,elephant))
```

- **zipWithIndex**
  - Zips the elements of the RDD with its element indexes. The indexes start from 0. If the RDD is spread across multiple partitions then a spark Job is started to perform this operation.

```scala
val z = sc.parallelize(Array("A", "B", "C", "D"))
val r = z.zipWithIndex
//res110: Array[(String, Long)] = Array((A,0), (B,1), (C,2), (D,3))

val z = sc.parallelize(100 to 120, 5)
val r = z.zipWithIndex
r.collect
//res11: Array[(Int, Long)] = Array((100,0), (101,1), (102,2), (103,3), (104,4), (105,5), (106,6), (107,7), (108,8), (109,9), (110,10), (111,11), (112,12), (113,13), (114,14), (115,15), (116,16), (117,17), (118,18), (119,19), (120,20))
```

- **top**
  - Utilizes the implicit ordering of $T$ to determine the top $k$ values and returns them as an array.

```scala
val c = sc.parallelize(Array(6, 9, 4, 7, 5, 8), 2)
c.top(2)
//res28: Array[Int] = Array(9, 8)
val topFreqUser = userRatings.map(l => (l._1, 1)).reduceByKey(_ + _).top(100)(Ordering[Int].on(_._2))
```

- **takeOrdered**
    - Orders the data items of the RDD using their inherent implicit ordering function and returns the first n items as an array.

```scala
val b = sc.parallelize(List("dog", "cat", "ape", "salmon", "gnu"), 2)
b.takeOrdered(2)
res19: Array[String] = Array(ape, cat)

// 按第二位的大小从小到大取top100
val topFreqUser = userRatings.map(l => (l._1, 1)).reduceByKey(_ + _).takeOrdered(100)(Ordering[Int].on(_._2))
```

- **toDebugString**
  - Returns a string that contains debug information about the RDD and its dependencies.

```scala
val a = sc.parallelize(1 to 9, 3)
val b = sc.parallelize(1 to 3, 3)
val c = a.subtract(b)
c.toDebugString
/* res6: String =
MappedRDD[15] at subtract at <console>:16 (3 partitions)
 SubtractedRDD[14] at subtract at <console>:16 (3 partitions)
	MappedRDD[12] at subtract at <console>:16 (3 partitions)
   ParallelCollectionRDD[10] at parallelize at <console>:12 (3 partitions)
  MappedRDD[13] at subtract at <console>:16 (3 partitions)
   ParallelCollectionRDD[11] at parallelize at <console>:12 (3 partitions)
```



- **dependencies**
  - Returns the RDD on which this RDD depends.

```scala
val b = sc.parallelize(List(1,2,3,4,5,6,7,8,2,4,2,1,1,1,1,1))
//b: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[32] at parallelize at <console>:12

b.dependencies.length
Int = 0
b.map(a => a).dependencies.length
//res40: Int = 1

// 笛卡尔积
b.cartesian(a).dependencies.length
//res41: Int = 2

b.cartesian(a).dependencies
/* res42: Seq[org.apache.spark.Dependency[_]] = List(org.apache.spark.rdd.CartesianRDD$$anon$1@576ddaaa, org.apache.spark.rdd.CartesianRDD$$anon$2@6d2efbbd)
```

**双Value类型**

- **intersection**
  - 对源 RDD 和参数 RDD 求交集后返回一个新的 RDD

```scala
val x = sc.parallelize(1 to 20)
val y = sc.parallelize(10 to 30)
val z = x.intersection(y)
z.collect
//res74: Array[Int] = Array(16, 12, 20, 13, 17, 14, 18, 10, 19, 15, 11)
```

- **union  ++**
  - 对源 RDD 和参数 RDD 求并集后返回一个新的 RDD
  - 这个并集会有重复的  需要distinct去重

```scala
val a = sc.parallelize(1 to 3, 1)
val b = sc.parallelize(5 to 7, 1)
(a ++ b).collect
//res0: Array[Int] = Array(1, 2, 3, 5, 6, 7)
```

- **subtract**
  - 以一个 RDD 元素为主，去除两个 RDD 中重复元素，将其他元素保留下来。求差集

```scala
val a = sc.parallelize(1 to 9, 3)
val b = sc.parallelize(1 to 3, 3)
val c = a.subtract(b)
c.collect
//res3: Array[Int] = Array(6, 9, 4, 7, 5, 8)
```

- **subtractByKey**
  - Very similar to *subtract*, but instead of supplying a function, the key-component of each pair will be automatically used as criterion for removing items from the first RDD.
  - 根据key对两个RDD去重

```scala
val a = sc.parallelize(List("dog", "tiger", "lion", "cat", "spider", "eagle"), 2)
val b = a.keyBy(_.length)
val c = sc.parallelize(List("ant", "falcon", "squid"), 2)
val d = c.keyBy(_.length)
b.subtractByKey(d).collect
//res15: Array[(Int, String)] = Array((4,lion))
```

- **zip**
  - 将两个 RDD 中的元素，以键值对的形式进行合并。
  - 其中，键值对中的 Key 为第 1 个 RDD 中的元素，Value 为第 2 个 RDD 中的相同位置的元素。

```scala
val a = sc.parallelize(1 to 100, 3)
val b = sc.parallelize(101 to 200, 3)
a.zip(b).collect
/*
res1: Array[(Int, Int)] = Array((1,101), (2,102), (3,103), (4,104), (5,105), (6,106), (7,107), (8,108), (9,109), (10,110), (11,111), (12,112), (13,113), (14,114), (15,115), (16,116), (17,117), (18,118), (19,119), (20,120), (21,121), (22,122), (23,123), (24,124), (25,125), (26,126), (27,127), (28,128), (29,129), (30,130), (31,131), (32,132), (33,133), (34,134), (35,135), (36,136), (37,137), (38,138), (39,139), (40,140), (41,141), (42,142), (43,143), (44,144), (45,145), (46,146), (47,147), (48,148), (49,149), (50,150), (51,151), (52,152), (53,153), (54,154), (55,155), (56,156), (57,157), (58,158), (59,159), (60,160), (61,161), (62,162), (63,163), (64,164), (65,165), (66,166), (67,167), (68,168), (69,169), (70,170), (71,171), (72,172), (73,173), (74,174), (75,175), (76,176), (77,177), (78,...
*/
val a = sc.parallelize(1 to 100, 3)
val b = sc.parallelize(101 to 200, 3)
val c = sc.parallelize(201 to 300, 3)
a.zip(b).zip(c).map((x) => (x._1._1, x._1._2, x._2 )).collect
/*
res12: Array[(Int, Int, Int)] = Array((1,101,201), (2,102,202), (3,103,203), (4,104,204), (5,105,205), (6,106,206), (7,107,207), (8,108,208), (9,109,209), (10,110,210), (11,111,211), (12,112,212), (13,113,213), (14,114,214), (15,115,215), (16,116,216), (17,117,217), (18,118,218), (19,119,219), (20,120,220), (21,121,221), (22,122,222), (23,123,223), (24,124,224), (25,125,225), (26,126,226), (27,127,227), (28,128,228), (29,129,229), (30,130,230), (31,131,231), (32,132,232), (33,133,233), (34,134,234), (35,135,235), (36,136,236), (37,137,237), (38,138,238), (39,139,239), (40,140,240), (41,141,241), (42,142,242), (43,143,243), (44,144,244), (45,145,245), (46,146,246), (47,147,247), (48,148,248), (49,149,249), (50,150,250), (51,151,251), (52,152,252), (53,153,253), (54,154,254), (55,155,255)...
*/
```

- **collectAsMap**
  - Similar to *collect*, but works on key-value RDDs and converts them into Scala maps to preserve their key-value structure.

```scala
val a = sc.parallelize(List(1, 2, 1, 3), 1)
val b = a.zip(a)
b.collectAsMap
//res1: scala.collection.Map[Int,Int] = Map(2 -> 2, 1 -> 1, 3 -> 3)
```

- mapValues
  - Takes the values of a RDD that consists of two-component tuples, and applies the provided function to transform each value. Then, it forms new two-component tuples using the key and the transformed value and stores them in a new RDD.

```scala
val a = sc.parallelize(List("dog", "tiger", "lion", "cat", "panther", "eagle"), 2)
val b = a.map(x => (x.length, x))
b.mapValues("x" + _ + "x").collect
//res5: Array[(Int, String)] = Array((3,xdogx), (5,xtigerx), (4,xlionx), (3,xcatx), (7,xpantherx), (5,xeaglex))
```

- cartesian
  - Computes the cartesian product between two RDDs (i.e. Each item of the first RDD is joined with each item of the second RDD) and returns them as a new RDD. *(Warning: Be careful when using this function.! Memory consumption can quickly become an issue!)*

```scala
val x = sc.parallelize(List(1,2,3,4,5))
val y = sc.parallelize(List(6,7,8,9,10))
x.cartesian(y).collect
/*
res0: Array[(Int, Int)] = Array((1,6), (1,7), (1,8), (1,9), (1,10), (2,6), (2,7), (2,8), (2,9), (2,10), (3,6), (3,7), (3,8), (3,9), (3,10), (4,6), (5,6), (4,7), (5,7), (4,8), (5,8), (4,9), (4,10), (5,9), (5,10))
*/
```

#### **Key-Value类型**

- **reduceByKey**
  - 可以将数据按照相同的 Key 对 Value 进行聚合

```scala
val a = sc.parallelize(List("dog", "cat", "owl", "gnu", "ant"), 2)
val b = a.map(x => (x.length, x))
b.reduceByKey(_ + _).collect
//res86: Array[(Int, String)] = Array((3,dogcatowlgnuant))

val a = sc.parallelize(List("dog", "tiger", "lion", "cat", "panther", "eagle"), 2)
val b = a.map(x => (x.length, x))
b.reduceByKey(_ + _).collect
//res87: Array[(Int, String)] = Array((4,lion), (3,dogcat), (7,panther), (5,tigereagle))
```

- **groupByKey**
  - 将数据源的数据根据 key 对 value 进行分组

```scala
val a = sc.parallelize(List("dog", "tiger", "lion", "cat", "spider", "eagle"), 2)
val b = a.groupByKey(_.length)
b.groupByKey.collect

//res11: Array[(Int, Seq[String])] = Array((4,ArrayBuffer(lion)), (6,ArrayBuffer(spider)), (3,ArrayBuffer(dog, cat)), (5,ArrayBuffer(tiger, eagle)))
```

- **reduceByKey和groupByKey的区别**
  - 从 shuffle 的角度：reduceByKey 和 groupByKey 都存在 shuffle 的操作，但是 reduceByKey 可以在 shuffle 前对分区内相同 key 的数据进行预聚合（combine）功能，这样会减少落盘的 数据量，而 groupByKey 只是进行分组，不存在数据量减少的问题，reduceByKey 性能比较 高。
  - 从功能的角度：reduceByKey 其实包含分组和聚合的功能。GroupByKey 只能分组，不能聚 合，所以在分组聚合的场合下，推荐使用 reduceByKey，如果仅仅是分组而不需要聚合。那 么还是只能使用 groupByKey
- **leftOuterJoin**
  - 类似于 SQL 语句的左外连接;  返回数据集左边的全部数据和数据集左边与右边有交集的数据

```scala
val a = sc.parallelize(List("dog", "salmon", "salmon", "rat", "elephant"), 3)
val b = a.keyBy(_.length)
val c = sc.parallelize(List("dog","cat","gnu","salmon","rabbit","turkey","wolf","bear","bee"), 3)
val d = c.keyBy(_.length)
b.leftOuterJoin(d).collect
/*
res1: Array[(Int, (String, Option[String]))] = Array((6,(salmon,Some(salmon))), (6,(salmon,Some(rabbit))), (6,(salmon,Some(turkey))), (6,(salmon,Some(salmon))), (6,(salmon,Some(rabbit))), (6,(salmon,Some(turkey))), (3,(dog,Some(dog))), (3,(dog,Some(cat))), (3,(dog,Some(gnu))), (3,(dog,Some(bee))), (3,(rat,Some(dog))), (3,(rat,Some(cat))), (3,(rat,Some(gnu))), (3,(rat,Some(bee))), (8,(elephant,None)))
*/
```

#### **Action算子**

- **reduce**
  - 聚集 RDD 中的所有元素，先聚合分区内数据，再聚合分区间数据

```scala
val a = sc.parallelize(1 to 100, 3)
a.reduce(_ + _)
//res41: Int = 5050
```

- **collect  toArray**
  - 在驱动程序中，以数组 Array 的形式返回数据集的所有元素

```scala
val c = sc.parallelize(List("Gnu", "Cat", "Rat", "Dog", "Gnu", "Rat"), 2)
c.collect
//res29: Array[String] = Array(Gnu, Cat, Rat, Dog, Gnu, Rat)
```

- **count**
  - 返回 RDD 中元素的个数

```scala
val c = sc.parallelize(List("Gnu", "Cat", "Rat", "Dog"), 2)
c.count
//res2: Long = 4
```

- **first**
  - 返回 RDD 中的第一个元素

```scala
val c = sc.parallelize(List("Gnu", "Cat", "Rat", "Dog"), 2)
c.first
//res1: String = Gnu
```

- **take**
  - 返回一个由 RDD 的前 n 个元素组成的数组

```scala
val b = sc.parallelize(List("dog", "cat", "ape", "salmon", "gnu"), 2)
b.take(2)
//res18: Array[String] = Array(dog, cat)
```

- **takeOrdered**
  - 返回该 RDD 排序后的前 n 个元素组成的数组

```scala
val b = sc.parallelize(List("dog", "cat", "ape", "salmon", "gnu"), 2)
b.takeOrdered(2)
//res19: Array[String] = Array(ape, cat)
```

- **countByKey**
  - 统计每种 key 的个数

```scala
val c = sc.parallelize(List((3, "Gnu"), (3, "Yak"), (5, "Mouse"), (3, "Dog")), 2)
c.countByKey
//res3: scala.collection.Map[Int,Long] = Map(3 -> 3, 5 -> 1)
```

- **save**
  - 将数据保存到不同格式的文件中

```scala
// 保存成 Text 文件
rdd.saveAsTextFile("output")
// 序列化成对象保存到文件
rdd.saveAsObjectFile("output1")
// 保存成 Sequencefile 文件
rdd.map((_,1)).saveAsSequenceFile("output2")
```

