#### 前言

通常，一个 Spark 应用程序包括两个 JVM 进程，**Driver**和**Executor**。Driver是主要的控制进程，负责创建Context、提交Job、将Job转换为Task、协调Executor之间的Task执行

Executor 主要负责执行具体的计算任务，并将结果返回给Driver。因为Driver的内存管理比较简单，和一般JVM程序区别不大，所以本文重点介绍Executor的内存管理。所以，本文所说的**内存管理是指Executor的内存管理。**

如下图所示：**Spark的内存管理划分可分为如下部分**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/31-2.jpg" style="zoom:33%;" />

#### 堆内On-Heap和堆外Off-Heap内存规划

- Executor acts as a JVM process, and its memory management is based on the JVM. So JVM memory management includes two methods:
  - **On-Heap** memory management: Objects are allocated on the JVM heap and bound by GC.
  - **Off-Heap** memory management: Objects are allocated in memory outside the JVM by serialization, managed by the application, and are not bound by GC. This memory management method can avoid frequent GC, but the disadvantage is that you have to write the logic of memory allocation and memory release.
- 作为一个 JVM 进程，Executor 的内存管理建立在 JVM 的内存管理之上，Spark 对 JVM 的堆内空间进行了更为详细的分配，以充分利用内存。
- 同时，Spark 引入了堆外内存，使之可以直接在工作节点的系统内存中开辟空间，进一步优化了内存的使用。**堆内内存受到 JVM 统一管理，堆外内存是直接向操作系统进行内存的申请和释放**。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/31-1.jpg" style="zoom:33%;" />

##### 堆内内存

- 堆内内存的大小 ，由Spark应用程序启动时的–executor-memory 或 spark.executor.memory 参数配置。Executor 内运行的并发任务共享 JVM堆内内存，大致分为四部分：**执行内存、存储内存、User Memory、Reserved Memory**
- **这些任务在缓存 RDD 数据和广播（Broadcast）数据时占用的内存被规划为存储（Storage）内存**
- **任务在执行 Shuffle 时占用的内存被规划为执行（Execution）内存**，剩余的部分不做特殊规划
- **User Memory：主要用于存放RDD转换操作所需的数据，如RDD依赖信息**
- **Reserved Memory：为系统保留的内存，用于存储Spark内部的对象**

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/31-3.jpg" style="zoom: 50%;" />

- Spark 对堆内内存的管理是一种逻辑上的”规划式”的管理，因为**对象实例占用内存的申请和释放都由 JVM 完成，Spark 只能在申请后和释放前记录这些内存**，我们来看其具体流程：
  - **申请内存流程**如下：
    - Spark 在代码中 new 一个对象实例；
    - JVM 从堆内内存分配空间，创建对象并返回对象引用；
  - **释放内存流程**如下：
    - Spark 记录该对象释放的内存，删除该对象的引用；
    - 等待 JVM 的垃圾回收 (GC) 机制释放该对象占用的堆内内存；

##### 堆外内存

- **为了进一步优化内存的使用以及提高 Shuffle 时排序的效率，Spark 引入了堆外（Offheap）内存，使之可以直接在工作节点的系统内存中开辟空间，存储经过序列化的二进制数据。**

- 默认情况下，堆外内存是禁用的，但我们可以通过`spark.memory.offHeap.enabled` 参数启用它 ，并通过`spark.memory.offHeap.size` 参数设置内存大小 
- 与On-heap memory相比，**Off-heap memory的模型比较简单，只有Storage memory和Execution memory**，其分布如下图所示：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/31-4.jpg" style="zoom:50%;" />

##### 统一内存管理与动态占用机制

- Spark1.6 之后引入的**统一内存管理机制，与静态内存管理的区别在于存储内存和执行内存共享同一块空间，可以动态占用对方的空闲区域，统一内存管理的堆内内存结构如图所示**：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/31-5.png" style="zoom:50%;" />

##### **动态占用机制，其规则如下：**

- 设定基本的存储内存和执行内存区域（spark.storage.storageFraction参数），该设定确定了双方各自拥有的空间的范围；

- 双方的空间都不足时，则存储到硬盘；若己方空间不足而对方空余时，可借用对方的空间；
- **执行内存的空间被对方占用后，可让对方将占用的部分转存到硬盘，然后归还借用的空间；**

- 存储内存的空间被对方占用后，无法让对方归还，因为需要考虑”Shuffle过程中的很多“因素，实现起来较为复杂。

凭借统一内存管理机制，Spark在一定程度上提高了堆内和堆外内存资源的利用率，降低了开发者维护 Spark 内存的难度，但并不意味着开发者可以高枕无忧。如果**存储内存的空间太大或者说缓存的数据过多，反而会导致频繁的全量垃圾回收，降低任务执行时的性能，因为 缓存的 RDD 数据通常都是长期驻留内存的。**

#### 存储内存管理

##### RDD的持久化机制

- 弹性分布式数据集（RDD）作为 Spark 最根本的数据抽象，是只读的分区记录（Partition）的集合，只能基于在稳定物理存储中的数据集上创建，或者在其他已有的 RDD 上执行转换 （Transformation）操作产生一个新的 RDD。
- 转换后的 RDD 与原始的 RDD 之间产生的依赖关系，构成了血统（Lineage）。凭借血统，Spark 保证了每一个 RDD 都可以被重新恢复。**但 RDD 的所有转换都是惰性的，即只有当一个返回结果给 Driver 的行动（Action）发生时， Spark 才会创建任务读取 RDD，然后真正触发转换的执行。**
- **Task 在启动之初读取一个分区时，会先判断这个分区是否已经被持久化，如果没有则需要检查 Checkpoint 或按照血统重新计算**。所以如果一个 RDD 上要执行多次行动，可以在第一次行动中使用 persist 或 cache 方法，在内存或磁盘中持久化或缓存这个 RDD，从而在后面的行动时提升计算速度。
- 事实上，**cache 方法是使用默认的 MEMORY_ONLY 的存储级别将 RDD 持久化到内存，故缓存是一种特殊的持久化**。 堆内和堆外存储内存的设计，便可以对缓存 RDD 时使用的内存做统一的规划和管理。

- RDD 的持久化由 Spark 的 Storage 模块负责，实现了 RDD 与物理存储的解耦合。Storage 模块负责管理 Spark 在计算过程中产生的数据，将那些在内存或磁盘、在本地或远程存取数据的功能封装了起来。**在具体实现时 Driver 端和 Executor 端的 Storage 模块构成了主从式 的架构，即 Driver 端的 BlockManager 为 Master，Executor 端的 BlockManager 为 Slave。Storage 模块在逻辑上以 Block 为基本存储单位，RDD 的每个 Partition 经过处理后唯一对应 一个 Block（BlockId 的格式为 rdd_RDD-ID_PARTITION-ID ）。Driver 端的 Master 负责整 个 Spark 应用程序的 Block 的元数据信息的管理和维护，而 Executor 端的 Slave 需要将 Block 的更新等状态上报到 Master，同时接收 Master 的命令，例如新增或删除一个 RDD**。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/31-6.png" style="zoom:50%;" />

- 在对 RDD 持久化时，Spark 规定了 MEMORY_ONLY、MEMORY_AND_DISK 等 7 种不同的存储级别，而存储级别是以下 5 个变量的组合：

> class StorageLevel private(
>
> ​	private var _useDisk: Boolean, //磁盘
>
> ​	private var _useMemory: Boolean, //这里其实是指堆内内存
>
> ​	private var _useOffHeap: Boolean, //堆外内存
>
> ​	private var _deserialized: Boolean, //是否为非序列化
>
> ​	private var _replication: Int = 1 //副本个数
>
> )

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0/31-7.png" style="zoom: 33%;" />

- 通过对数据结构的分析，可以看出存储级别从三个维度定义了 RDD 的 Partition（同时也就 是 Block）的存储方式：
  - **存储位置**：磁盘／堆内内存／堆外内存。
  - **存储形式：**Block缓存到存储内存后，是否为非序列化的形式
  - **副本数量**：大于1时需要远程冗余备份到其他节点

#### 执行内存管理

- **执行内存主要用来存储任务在执行 Shuffle 时占用的内存，Shuffle 是按照一定规则对 RDD 数据重新分区的过程**，Shuffle 的 Write 和 Read 两阶段对执行内存的使用：
- **Shuffle Write**
  - 若在 map 端选择普通的排序方式，会采用 ExternalSorter 进行外排，**在内存中存储数据时主要占用堆内执行空间**
  - 若在 map 端选择 Tungsten 的排序方式，则采用 ShuffleExternalSorter 直接对以序列化形式存储的数据排序，在内存中存储数据时可以占用堆外或堆内执行空间，取决于用户是否开启 了堆外内存以及堆外执行内存是否足够
- **Shuffle Read**
  - 在对 reduce 端的数据进行聚合时，要将数据交给 Aggregator 处理，**在内存中存储数据时占用堆内执行空间**
  - 如果需要进行最终结果排序，则要将再次将数据交给 ExternalSorter 处理，占用堆内执行空间