#### 前言

Spark 计算框架为了能够进行高并发和高吞吐的数据处理，封装了三大数据结构，用于处理不同的应用场景。三大数据结构分别如下，本文为RDD的学习笔记

- **RDD** : 弹性分布式数据集

- **累加器**：分布式共享只写变量

- **广播变量**：分布式共享只读变量

#### 什么是RDD

RDD（Resilient Distributed Dataset）叫做**弹性分布式数据集**，是 Spark 中最基本的数据处理模型。代码中是**一个抽象类，它代表一个弹性的、不可变、可分区、里面的元素可并行计算的集合**。

- **弹性**
  - 存储的弹性：内存与磁盘的自动切换；
  - 容错的弹性：数据丢失可以自动恢复；
  - 计算的弹性：计算出错重试机制；
  - 分片的弹性：可根据需要重新分片。

- 分布式：**数据存储在大数据集群不同节点上**
- 数据集：**RDD封装了计算逻辑，并不保存数据**


- 数据抽象：**RDD是一个抽象类，需要子类具体实现**


- 不可变：**RDD封装了计算逻辑，是不可改变的，想要改变，只能产生新的，在新的RDD里面封装计算逻辑**
- **可分区、并行计算**

#### 核心属性

- **分区列表**
  - RDD 数据结构中存在分区列表，用于执行任务时并行计算，是实现分布式计算的重要属性
- **分区计算函数**
  - Spark 在计算时，是使用分区函数对每一个分区进行计算
- **RDD之间的依赖关系**
  - RDD 是计算模型的封装，当需要将多个计算模型进行组合时，就需要将多个 RDD 建立依赖关系

- 分区器（可选）
  - 当数据为 Key- Value 类型数据时，可以通过设定分区器自定义数据的分区

- 首选位置（可选）
  - 计算数据时，可以根据计算节点的状态选择不同的节点位置进行计算

#### 执行原理

从计算的角度来讲，数据处理过程中需要计算资源（内存 & CPU）和计算模型（逻辑）。 执行时，需要将计算资源和计算模型进行协调和整合；

**Spark 框架在执行时，先申请资源，然后将应用程序的数据处理逻辑分解成一个一个的 计算任务。然后将任务发到已经分配资源的计算节点上, 按照指定的计算模型进行数据计 算。最后得到计算结果。**

RDD 是 Spark 框架中用于数据处理的核心模型，在 Yarn 环境中，RDD 的工作原理:

- **启动Yarn集群环境**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/3-1.jpg" style="zoom:33%;" />

- **Spark通过申请资源创建调度节点和计算节点**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/3-2.jpg" style="zoom:33%;" />

- **Spark框架根据需求将计算逻辑根据分区划分成不同的任务**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/3-3.jpg" style="zoom:33%;" />

- **调度节点将任务根据计算节点状态发送到对应的计算节点进行计算**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/3-4.jpg" style="zoom:33%;" />

从以上流程可以看出，RDD 在整个流程中主要用于将逻辑进行封装，并生成 Task 发送给 Executor 节点执行计算

#### 基础编程

- 本部分包含RDD的创建、Transformations算子、Action算子；具体看博文[RDD基础编程及常用算子]()

#### RDD序列化

本部分内容包含于博文：[Spark中的闭包与广播变量](http://yearing1017.cn/2021/05/20/Spark中的闭包与广播变量/)

#### RDD依赖关系

##### RDD血缘关系

RDD 只支持粗粒度转换，即在大量记录上执行的单个操作。将创建 RDD 的一系列 Lineage （血统）记录下来，以便恢复丢失的分区。RDD 的 Lineage 会记录 RDD 的元数据信息和转 换行为，当该 RDD 的部分分区数据丢失时，它可以根据这些信息来重新运算和恢复丢失的数据分区。

```scala
// toDebugString : Returns a string that contains debug information about the RDD and its dependencies.
val fileRDD: RDD[String] = sc.textFile("input/1.txt")
println(fileRDD.toDebugString)
println("----------------------")

val wordRDD: RDD[String] = fileRDD.flatMap(_.split(" "))
println(wordRDD.toDebugString)
println("----------------------")

val mapRDD: RDD[(String, Int)] = wordRDD.map((_,1))
println(mapRDD.toDebugString)
println("----------------------")

val resultRDD: RDD[(String, Int)] = mapRDD.reduceByKey(_+_)
println(resultRDD.toDebugString)
resultRDD.collect()
```

##### RDD依赖关系

依赖关系，其实就是两个相邻 RDD 之间的关系

```scala
val b = sc.parallelize(List(1,2,3,4,5,6,7,8,2,4,2,1,1,1,1,1))
b: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[32] at parallelize at <console>:12
b.dependencies.length
Int = 0
// dependencies Returns the RDD on which this RDD depends.
b.map(a => a).dependencies.length
res40: Int = 1
// 笛卡尔积
b.cartesian(a).dependencies.length
res41: Int = 2

b.cartesian(a).dependencies
res42: Seq[org.apache.spark.Dependency[_]] = List(org.apache.spark.rdd.CartesianRDD$$anon$1@576ddaaa, org.apache.spark.rdd.CartesianRDD$$anon$2@6d2efbbd)
```

##### RDD窄依赖

窄依赖表示**每一个父(上游)RDD 的 Partition 最多被子（下游）RDD 的一个 Partition 使用**， 窄依赖我们形象的比喻为独生子女

##### RDD宽依赖

宽依赖表示**同一个父（上游）RDD 的 Partition 被多个子（下游）RDD 的 Partition 依赖，会引起 Shuffle**，总结：宽依赖我们形象的比喻为多生

##### RDD阶段划分

DAG（Directed Acyclic Graph）有向无环图是由点和线组成的拓扑图形，该图形具有方向，不会闭环。例如，DAG 记录了 RDD 的转换过程和任务的阶段

- Spark是**根据shuffle类算子来进行stage的划分**。如果我们的代码中执行了某个shuffle类算子（比如reduceByKey、join等），那么就会在该算子处，划分出一个stage界限来。
- 可以大致理解为，**shuffle算子执行之前的代码会被划分为一个stage，shuffle算子执行以及之后的代码会被划分为下一个stage**。因此一个stage刚开始执行的时候，它的每个Task可能都会从上一个stage的Task所在的节点，去通过网络传输拉取需要自己处理的所有key，然后对拉取到的所有相同的key使用我们自己编写的算子函数执行聚合操作（比如reduceByKey()算子接收的函数）。这个过程就是shuffle。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/3-5.jpg" style="zoom:33%;" />

##### RDD任务划分

**RDD 任务切分中间分为：Application、Job、Stage 和 Task**

- Application：初始化一个 SparkContext 即生成一个 Application；
- Job：一个 Action 算子就会生成一个 Job；
- Stage：Stage 等于宽依赖(ShuffleDependency)的个数加 1；
- Task：一个 Stage 阶段中，最后一个 RDD 的分区个数就是 Task 的个数。

注意：**Application->Job->Stage->Task 每一层都是 1 对 n 的关系**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/3-6.jpg" style="zoom:33%;" />

#### RDD持久化

##### RDDCache缓存

**RDD 通过 Cache 或者 Persist 方法将前面的计算结果缓存**，默认情况下会把数据以缓存在 JVM 的堆内存中。但是**并不是这两个方法被调用时立即缓存，而是触发后面的 action 算 子时，该 RDD 将会被缓存在计算节点的内存中，并供后面重用。**

```scala
// cache 操作会增加血缘关系，不改变原有的血缘关系
println(wordToOneRdd.toDebugString)
// 数据缓存
wordToOneRdd.cache()
// 可以更改存储级别
mapRdd.persist(StorageLevel.MEMORY_AND_DISK_2)

//存储级别
object StorageLevel {
val NONE = new StorageLevel(false, false, false, false)
val DISK_ONLY = new StorageLevel(true, false, false, false)
val DISK_ONLY_2 = new StorageLevel(true, false, false, false, 2)
val MEMORY_ONLY = new StorageLevel(false, true, false, true)
val MEMORY_ONLY_2 = new StorageLevel(false, true, false, true, 2)
val MEMORY_ONLY_SER = new StorageLevel(false, true, false, false)
val MEMORY_ONLY_SER_2 = new StorageLevel(false, true, false, false, 2)
val MEMORY_AND_DISK = new StorageLevel(true, true, false, true)
val MEMORY_AND_DISK_2 = new StorageLevel(true, true, false, true, 2)
val MEMORY_AND_DISK_SER = new StorageLevel(true, true, false, false)
val MEMORY_AND_DISK_SER_2 = new StorageLevel(true, true, false, false, 2)
val OFF_HEAP = new StorageLevel(true, true, true, false, 1)
```

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/3-7.jpg" style="zoom:50%;" />

缓存有可能丢失，或者存储于内存的数据由于内存不足而被删除，**RDD 的缓存容错机制保证了即使缓存丢失也能保证计算的正确执行。通过基于 RDD 的一系列转换，丢失的数据会被重算**，由于 RDD 的各个 Partition 是相对独立的，因此只需要计算丢失的部分即可， 并不需要重算全部 Partition。

Spark 会自动对一些 Shuffle 操作的中间数据做持久化操作(比如：reduceByKey)。这样做的目的是为了当一个节点 Shuffle 失败了避免重新计算整个输入。但是，在实际使用的时候，如果想重用数据，仍然建议调用 persist 或 cache。

##### RDD CheckPoint 检查点

所谓的**检查点其实就是通过将 RDD 中间结果写入磁盘**，由于血缘依赖过长会造成容错成本过高，这样就不如在中间阶段做检查点容错，如果检查点之后有节点出现问题，可以从检查点开始重做血缘，减少了开销。

**对 RDD 进行 checkpoint 操作并不会马上被执行，必须执行 Action 操作才能触发**。

```scala
// 设置检查点路径
sc.setCheckpointDir("./checkpoint1")

// 创建一个 RDD，读取指定位置文件:hello atguigu atguigu
val lineRdd: RDD[String] = sc.textFile("input/1.txt")

// 业务逻辑
val wordRdd: RDD[String] = lineRdd.flatMap(line => line.split(" "))
val wordToOneRdd: RDD[(String, Long)] = wordRdd.map {
	word => {
		(word, System.currentTimeMillis())
}}

// 增加缓存,避免再重新跑一个 job 做 checkpoint
wordToOneRdd.cache()

// 数据检查点：针对 wordToOneRdd 做检查点计算
wordToOneRdd.checkpoint()

// 触发执行逻辑
wordToOneRdd.collect().foreach(println)
```

##### 缓存和检查点的区别

- Cache 缓存只是将数据保存起来，不切断血缘依赖。Checkpoint 检查点切断血缘依赖。

- Cache 缓存的数据通常存储在磁盘、内存等地方，可靠性低。Checkpoint 的数据通常存 储在 HDFS 等容错、高可用的文件系统，可靠性高。

- 建议对 checkpoint()的 RDD 使用 Cache 缓存，这样 checkpoint 的 job 只需从 Cache 缓存 中读取数据即可，否则需要再从头计算一次 RDD。

#### RDD分区器

Spark 目前支持 Hash 分区和 Range 分区，和用户自定义分区。Hash 分区为当前的默认分区。分区器直接决定了 RDD 中分区的个数、RDD 中每条数据经过 Shuffle 后进入哪个分区，进而决定了 Reduce 的个数。

- 只有 Key-Value 类型的 RDD 才有分区器，非 Key-Value 类型的 RDD 分区的值是 None
- 每个 RDD 的分区 ID 范围：0 ~ (numPartitions - 1)，决定这个值是属于那个分区的

**Hash 分区**：对于给定的 key，计算其 hashCode,并除以分区个数取余

**Range 分区**：将一定范围内的数据映射到一个分区中，尽量保证每个分区数据均匀，而 且分区间有序

#### RDD文件读取与保存

Spark 的数据读取及数据保存可以从两个维度来作区分：**文件格式以及文件系统**

- 文件格式分为：text 文件、csv 文件、sequence 文件以及 Object 文件
- 文件系统分为：本地文件系统、HDFS、HBASE 以及数据库

**text文件**

```scala
// 读取输入文件
val inputRDD: RDD[String] = sc.textFile("input/1.txt")

// 保存数据
inputRDD.saveAsTextFile("output")
```

**sequence 文件**

SequenceFile 文件是 用来存储二进制形式的 key-value 对而设计的一种平面文件(Flat File)

```scala
// 保存数据为 SequenceFile
dataRDD.saveAsSequenceFile("output")

// 读取 SequenceFile 文件
sc.sequenceFile[Int,Int]("output").collect().foreach(println)
```

**object 对象文件**

对象文件是将对象序列化后保存的文件，采用 Java 的序列化机制。可以通过 objectFile函数接收一个路径，读取对象文件，返回对应的 RDD，也可以通过调用 saveAsObjectFile()实现对对象文件的输出。因为是序列化所以要指定类型。

```scala
// 保存数据
dataRDD.saveAsObjectFile("output")

// 读取数据
sc.objectFile[Int]("output").collect().foreach(println)
```

