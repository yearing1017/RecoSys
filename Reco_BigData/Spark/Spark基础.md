#### 前言

- Spark基础概念、运行原理、流程

#### Spark是什么

- Spark 是一种**基于内存的快速、通用、可扩展的大数据分析计算引擎**

#### Spark和Hadoop

- **Hadoop**
  - 由 java 语言编写的，**在分布式服务器集群上存储海量数据并运行分布式分析应用的开源框架**
  - 作为 Hadoop 分布式文件系统，HDFS 处于 Hadoop 生态圈的最下层，存储着所有的数据 ， 支持着 Hadoop 的所有服 务
  - MapReduce 是一种编程模型，Hadoop 根据 Google 的 MapReduce 论文将其实现， 作为 Hadoop 的分布式计算模型，是 Hadoop 的核心。
  - **HBase** 是对 Google 的 Bigtable 的开源实现，但又和 Bigtable 存在许多不同之处。 **HBase 是一个基于 HDFS 的分布式数据库，擅长实时地随机读/写超大规模数据集**。 是 Hadoop非常重要的组件。Hadoop database 的简称，也就是基于Hadoop数据库，是一种NoSQL数据库，主要**适用于海量明细数据（十亿、百亿）的随机实时查询**，如日志明细、交易清单、轨迹行为等。
  - **Hive：Hadoop数据仓库**，严格来说，不是数据库，主要是**让开发人员能够通过SQL来计算和处理HDFS上的结构化数据，适用于离线的批量数据计算。**
    - **通过元数据来描述Hdfs上的结构化文本数据**，通俗点来说，就是定义一张表来描述HDFS上的结构化文本，包括各列数据名称，数据类型是什么等，方便我们处理数据，当前很多SQL ON Hadoop的计算引擎均用的是hive的元数据，如Spark SQL、Impala等
    - 通过SQL来处理和计算HDFS的数据，Hive会将SQL翻译为MapReduce来处理数据
- **Spark**
  - 一种由 **Scala 语言开发的快速、通用、可扩展的大数据分析引擎**
  - **Spark Core** 中提供了 Spark 最基础与最核心的功能
  - **Spark SQL** 是 Spark 用来操作结构化数据的组件。通过 Spark SQL，用户可以使用 SQL 或者 Apache Hive 版本的 SQL 方言（HQL）来查询数据。
  - **Spark Streaming 是 Spark针对实时数据进行流式计算的组件**，提供了丰富的处理数据流的 API

#### Spark Or Hadoop

- Hadoop MapReduce 由于其设计初衷并不是为了满足循环迭代式数据流处理，因此在多并行运行的数据可复用场景（如：机器学习、图挖掘算法、交互式数据挖掘算法）中存 在诸多计算效率等问题。所以 Spark 应运而生，**Spark 就是在传统的 MapReduce 计算框 架的基础上，利用其计算过程的优化，从而大大加快了数据分析、挖掘的运行和读写速度，并将计算单元缩小到更适合并行计算和重复使用的 RDD 计算模型。**
- 机器学习中 ALS、凸优化梯度下降等。这些都需要基于数据集或者数据集的衍生数据 反复查询反复操作。MR 这种模式不太合适，即使多 MR 串行处理，性能和时间也是一 个问题。数据的共享依赖于磁盘。另外一种是交互式数据挖掘， MR 显然不擅长。而 Spark 所基于的 scala 语言恰恰擅长函数的处理。
- Spark 是一个分布式数据快速分析项目。**它的核心技术是弹性分布式数据集（Resilient Distributed Datasets），提供了比 MapReduce 丰富的模型，可以快速在内存中对数据集进行多次迭代，来支持复杂的数据挖掘算法和图形计算算法。**
- **Spark 和 Hadoop 的根本差异是多个作业之间的数据通信问题 : Spark 多个作业之间数据通信是基于内存，而 Hadoop 是基于磁盘。**
- **Spark Task 的启动时间快**。Spark 采用 fork 线程的方式，而 Hadoop 采用创建新的进程的方式。
- Spark 只有在 shuffle 的时候将数据写入磁盘，而 Hadoop 中多个 MR 作业之间的数据交 互都要依赖于磁盘交互
- Spark 的缓存机制比 HDFS 的缓存机制高效

#### 概念

- **Driver**
  - 控制程序，负责为Application构建DAG图
  - 驱动；运行Application的main主函数 并且 创建SparkContext
  - **Spark 驱动器节点，用于执行 Spark 任务中的 main 方法，负责实际代码的执行工作**：
    - 将用户程序转化为作业（job）
    - 在 Executor 之间调度任务(task)
    - 跟踪 Executor 的执行情况
    - 通过 UI 展示查询运行情况
  
- **Executer**
  - 执行单元；为某Application运行在WorkNode上的一个进程，该进程负责运行Task，并且负责将数据存在内存或磁盘上，每个Application都有各自独立的Executors
  
  - **Spark Executor 是集群中工作节点（Worker）中的一个 JVM 进程，负责在 Spark 作业 中运行具体任务（Task），任务彼此之间相互独立**
  
  - Spark 应用启动时，Executor 节点被同时启动，并且始终伴随着整个 Spark 应用的生命周期而存在。**如果有 Executor 节点发生了 故障或崩溃，Spark 应用也可以继续执行，会将出错节点上的任务调度到其他 Executor 节点 上继续运行。**
  
  - **核心功能**
    - 负责运行组成 Spark 应用的任务，并将结果返回给驱动器进程
    - 它们通过自身的块管理器（Block Manager）为用户程序中要求缓存的 RDD 提供内存 式存储
    - RDD 是直接缓存在 Executor 进程内的，因此任务可以在运行时充分利用缓存 数据加速运算
    
  - 在提交应用中，可以提供参数指定计算节点的个数，以及对应的资 源。这里的资源一般指的是工作节点 Executor 的内存大小和使用的虚拟 CPU 核（Core）数量
  
  - |       名称        | 说明                                   |
    | :---------------: | :------------------------------------- |
    |  --num-executors  | 配置 Executor 的数量                   |
    | --executor-memory | 配置每个 Executor 的内存大小           |
    | --executor-cores  | 配置每个 Executor 的虚拟 CPU core 数量 |
  
- **Master & Worker**
  - Spark 集群的独立部署环境中，不需要依赖其他的资源调度框架，自身就实现了资源调度的功能，所以环境中还有其他两个核心组件：Master 和 Worker，**这里的 Master 是一个进 程，主要负责资源的调度和分配，并进行集群的监控等职责，类似于 Yarn 环境中的 ResourceManager, 而 Worker 呢，也是进程，一个 Worker 运行在集群中的一台服务器上，由 Master 分配资源对数据进行并行的处理和计算，类似于 Yarn 环境中 NodeManager。**
  
- **ApplicationMaster**
  - Hadoop 用户向 YARN 集群提交应用程序时，提交程序中应该包含 ApplicationMaster，用于向资源调度器申请执行任务的资源容器 Container，运行用户自己的程序任务 job，监控整个任务的执行，跟踪整个任务的状态，处理任务失败等异常情况。
  - 说的简单点就是，ResourceManager（资源）和 Driver（计算）之间的解耦合靠的就是 ApplicationMaster。
  
- **并行度**

  - 在分布式计算框架中一般都是多个任务同时执行，由于任务分布在不同的计算节点进行 计算，所以能够真正地实现多任务并行执行，记住，这里是并行，而不是并发。这里我们将 整个集群并行执行任务的数量称之为并行度。一个作业到底并行度是多少呢？这个取决 于框架的默认配置。应用程序也可以在运行过程中动态修改。

- Cluster Manager
  - 集群资源管理中心，负责分配计算资源
  - 在集群上获取资源的外部服务(例如：Local、Standalone、Mesos或Yarn等集群管理系统)；
  
- Operation
  - 操作；作用于RDD的各种操作分为**Transformation和Action**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/TA.jpg" style="zoom:33%;" />

- RDD
  - 弹性分布式数据集（Resilient Distributed Dataset），是分布式内存的一个抽象概念，提供了一种高度受限的共享内存模型
- DAG
  - 有向无环图（Directed Acyclic Graph），反映RDD之间的依赖关系
  - 这里所谓的有向无环图，并不是真正意义的图形，而是由 Spark 程序直接映射成的数据 流的高级抽象模型。简单理解就是将整个程序计算的执行过程用图形表示出来,这样更直观， 更便于理解，可以用于表示程序的拓扑结构。
- Application
  - 用户程序；基于spark的用户程序，包含了一个**Driver Program** 和集群中多个的 **Executor**
  - 用户编写的Spark应用程序，一个Application包含多个Job
- Job
  - 作业，一个Job包含多个RDD及作用于相应RDD上的各种操作
  - Job 以Action 方法为界，遇到一个Action方法则触发一个Job
- Stage
  - 阶段，是作业的基本调度单位，一个作业会分为多组任务，每组任务被称为“阶段”
  - Stage是Job的子集，以RDD宽依赖（即 Shuffle）为界，遇到Shuffle则做一次划分
- Task
  - 任务，运行在Executor上的工作单元，是Executor中的一个线程
  - Task是Stage的子集，以并行度（分区数）来衡量，分区数是多少，则有多少个Task

#### 流程图示

- Application、Job、Stage、Task的关系

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/1.jpg" style="zoom: 33%;" />

- Driver、Cluster Manager 、Work Node、 Executor、Task关系
- 如下图所示，它展示了一个 **Spark 执行时的基本结构**。图形中的 **Driver 表示 master， 负责管理整个集群中的作业任务调度。图形中的 Executor 则是 slave，负责实际执行任务**。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/2.jpg" style="zoom:33%;" />

- 流程调度图示

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/3.jpg" style="zoom:33%;" />



- 提交流程
  - 所谓的提交流程，其实就是开发人员根据需求写的应用程序通过 Spark 客户端提交 给 Spark 运行环境执行计算的流程。工作中，将 Spark 引用部署到 Yarn 环境中会更多一些，所以本提交流程是基于 Yarn 环境的
  - <img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/4.png" style="zoom:33%;" />
  - Spark 应用程序**提交到 Yarn 环境中执行的时候，一般会有两种部署执行的方式：Client 和 Cluster。两种模式主要区别在于：Driver 程序的运行节点位置**

- **Yarn Client模式**：Client 模式将**用于监控和调度的 Driver 模块在客户端执行，而不是在 Yarn 中，所以一 般用于测试**

  - Driver 在任务提交的本地机器上运行

  - Driver 启动后会和 ResourceManager 通讯申请启动 ApplicationMaster

  - ResourceManager 分配 container，在合适的 NodeManager 上启动 ApplicationMaster，负责向 ResourceManager 申请 Executor 内存

  - ResourceManager 接到 ApplicationMaster 的资源申请后会分配 container，然后ApplicationMaster 在资源分配指定的 NodeManager 上启动 Executor 进程

  - Executor 进程启动后会向 Driver 反向注册，Executor 全部注册完成后 Driver 开始执行 main 函数

    之后执行到 Action 算子时，触发一个 Job，并根据宽依赖开始划分 stage，每个 stage 生成对应的 TaskSet，之后将 task 分发到各个 Executor 上执行

- **Yarn Cluster模式：**Cluster 模式将**用于监控和调度的 Driver 模块启动在 Yarn 集群资源中执行。一般应用于实际生产环境**

  - 在 YARN Cluster 模式下，任务提交后会和 ResourceManager 通讯申请启动ApplicationMaster，
  - 随后 ResourceManager 分配 container，在合适的 NodeManager 上启动 ApplicationMaster， 此时的 ApplicationMaster 就是 Driver。
  - Driver 启动后向 ResourceManager 申请 Executor 内存，ResourceManager 接到ApplicationMaster 的资源申请后会分配 container，然后在合适的 NodeManager 上启动 Executor 进程
  - Executor 进程启动后会向 Driver 反向注册，Executor 全部注册完成后 Driver 开始执行 main 函数，
  - 之后执行到 Action 算子时，触发一个 Job，并根据宽依赖开始划分 stage，每个 stage 生 成对应的 TaskSet，之后将 task 分发到各个 Executor 上执行。

#### 运行流程及原理

- 使用**spark-submit提交一个Spark作业**之后，这个作业就会**启动一个对应的Driver进程**。

- 根据你使用的部署模式（deploy-mode）不同，Driver进程可能在本地启动，也可能在集群中某个工作节点上启动。

  - 而Driver进程要做的第一件事情，就是**向集群管理器**（可以是Spark Standalone集群，也可以是其他的资源管理集群，美团•大众点评使用的是YARN作为资源管理集群）**申请运行Spark作业需要使用的资源**，这里的**资源指的就是Executor进程**。
  - YARN集群管理器会根据我们为Spark作业设置的资源参数，**在各个工作节点上，启动一定数量的Executor进程**，每个Executor进程都占有一定数量的内存和CPU core

- 在申请到了作业执行所需的资源之后，Driver进程就会开始调度和执行我们编写的作业代码了。

  - Driver进程会将我们编写的Spark作业代码**分拆为多个stage**，**每个stage执行一部分代码片段，并为每个stage创建一批Task，然后将这些Task分配到各个Executor进程中执行。**

  - **Task是最小的计算单元，负责执行一模一样的计算逻辑**（也就是我们自己编写的某个代码片段），只是**每个Task处理的数据不同**而已。
  - **一个stage的所有Task都执行完毕之后**，会在各个节点本地的磁盘文件中**写入计算中间结果**，然后**Driver就会调度运行下一个stage**。
  - **下一个stage的Task的输入数据就是上一个stage输出的中间结果**。如此循环往复，直到将我们自己编写的代码逻辑全部执行完，并且计算完所有的数据，得到我们想要的结果为止。

- stage的划分

  - Spark是**根据shuffle类算子来进行stage的划分**。如果我们的代码中执行了某个shuffle类算子（比如reduceByKey、join等），那么就会在该算子处，划分出一个stage界限来。
  - 可以大致理解为，**shuffle算子执行之前的代码会被划分为一个stage，shuffle算子执行以及之后的代码会被划分为下一个stage**。因此一个stage刚开始执行的时候，它的每个Task可能都会从上一个stage的Task所在的节点，去通过网络传输拉取需要自己处理的所有key，然后对拉取到的所有相同的key使用我们自己编写的算子函数执行聚合操作（比如reduceByKey()算子接收的函数）。这个过程就是shuffle。

- 当我们在代码中执行了cache/persist等持久化操作时，根据我们选择的持久化级别的不同，每个Task计算出来的数据也会保存到Executor进程的内存或者所在节点的磁盘文件中。

- 因此**Executor的内存主要分为三块**：

  - 第一块是让Task执行我们自己编写的代码时使用，默认是占Executor总内存的20%；
  - 第二块是让Task通过shuffle过程拉取了上一个stage的Task的输出后，进行聚合等操作时使用，默认也是占Executor总内存的20%；
  - 第三块是让RDD持久化时使用，默认占Executor总内存的60%。

- **在实际编程中，我们不需关心以上调度细节。只需使用 Spark 提供的指定语言的编程接口调用相应的 API 即可**.

  - **在 Spark API 中**, 一个 应用(Application) 对应一个 SparkContext 的实例。一个 应用 可以用于单个 Job，或者分开的多个 Job 的 session，或者响应请求的长时间生存的服务器。与 MapReduce 不同的是，一个应用的进程（我们称之为 Executor)，会一直在集群上运行，即使当时没有 Job 在上面运行。

  - 而调用一个Spark内部的 Action 会产生一个 Spark job 来完成它。 为了确定这些job实际的内容，Spark 检查 RDD 的DAG再计算出执行 plan 。这个 plan 以最远端的 RDD 为起点（最远端指的是对外没有依赖的 RDD 或者 数据已经缓存下来的 RDD），产生结果 RDD 的 Action 为结束 。并根据是否发生 shuffle 划分 DAG 的 stage.

- **启动 Spark 运行程序主要有两种方式:**

  - 一种是使用 spark-submit 将脚本文件提交

  - 一种是打开 Spark 跟某种特定语言的解释器

    - spark-shell: 启动了 Spark 的 scala 解释器

    - pyspark: 启动了 Spark 的 python 解释器
    - sparkR: 启动了 Spark 的 R 解释器