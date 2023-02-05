#### **闭包**

------

- 闭包是一个函数，该函数**返回值依赖于声明在函数外部的一个或多个变量**。
- 闭包通常来讲可以简单的认为是可以访问一个函数里面局部变量的另外一个函数。

```scala
val mult= (i:Int) => i * 10
```

- 如上匿名函数，函数体内有一个变量 i，它作为函数的一个参数。如下面的另一段代码：

```scala
var factor = 3 
val mult = (i:Int) => i * factor
```

- 在 mult 中有两个变量i 和 factor。i 是函数的形式参数，在mult 函数被调用时，i 被赋予一个新的值。然而，factor不是形式参数，而是自由变量
- 这样定义的函数变量 **mult 成为一个"闭包"，因为它引用到函数外面定义的变量**，定义这个函数的过程是将这个自由变量捕获而构成一个封闭的函数。

#### **RDD序列化**

------

- **从计算的角度, 算子以外的代码都是在 Driver 端执行, 算子里面的代码都是在 Executor 端执行。**
- 那么在 scala 的函数式编程中，就会导致算子内经常会用到算子外的数据，这样就 形成了闭包的效果**，如果使用的算子外的数据无法序列化，就意味着无法传值给 Executor 端执行，就会发生错误，所以需要在执行任务计算前，**检测闭包内的对象是否可以进行序列化，这个操作我们称之为闭包检测**。**
- 序列化方法和属性：见下面例子

```scala
object serializable02_function {

   def main(args: Array[String]): Unit = {
   //1.创建 SparkConf 并设置 App 名称
    val conf: SparkConf = new SparkConf().setAppName("SparkCoreTest").setMaster("local[*]")

   //2.创建 SparkContext，该对象是提交 Spark App 的入口
   val sc: SparkContext = new SparkContext(conf)

   //3.创建一个 RDD
    val rdd: RDD[String] = sc.makeRDD(Array("hello world", "hello spark","hive", "atguigu"))

   //3.1 创建一个 Search 对象
    val search = new Search("hello")

   //3.2 函数传递，打印：ERROR Task not serializable
    search.getMatch1(rdd).collect().foreach(println)

   //3.3 属性传递，打印：ERROR Task not serializable
   search.getMatch2(rdd).collect().foreach(println)
   //4.关闭连接
    sc.stop()
  }
}

class Search(query:String) extends Serializable {
   def isMatch(s: String): Boolean = {
    s.contains(query)
   }
   // 函数序列化案例
  def getMatch1 (rdd: RDD[String]): RDD[String] = {

   //rdd.filter(this.isMatch)
    rdd.filter(isMatch)
   }
   // 属性序列化案例
  def getMatch2(rdd: RDD[String]): RDD[String] = {
     //rdd.filter(x => x.contains(this.query))
     rdd.filter(x => x.contains(query))
     //val q = query
     //rdd.filter(x => x.contains(q))
   }
}
```

#### **Spark 广播变量**

------

- 默认情况下，**如果在一个算子的函数中使用到了某个外部的变量，那么这个变量的值会被拷贝到每个task中，此时每个task只能操作自己的那份变量副本，如果多个task想要共享某个变量，那么这种方式是做不到的。** 
- Spark为此提供了两种共享变量，一种是Broadcast Variable(广播变量)，另一种是Accumulator(累加变量)。**Broadcast Variable会将使用到的变量，仅仅为每个节点拷贝一份，而不是给节点上的每个task拷贝一份。这样可以优化性能，减少网络传输及内存消耗。**Accumulator与Broadcast Variable不同，Broadcast Variable主要用于共享读，是只读的，没法去写。Accumulator可以让多个task共同操作一份变量，主要可以进行累加操作 
- 在提交作业后，task在执行的过程中，有一个或多个值需要在计算的过程中多次从Driver端拿取时，此时会必然会发生大量的网络IO，这时，最好用广播变量的方式，将Driver端的变量的值事先广播到每一个Worker端；以后再计算过程中只需要从本地拿取该值即可，避免网络IO，提高计算效率
- 广播变量在广播的时候，将Driver端的变量广播到每一个每一个Worker端，一个Worker端会收到一份仅一份该变量的值；**注意：广播的值必须是一个确切的值，不能广播RDD（因为RDD是一个数据的描述，没有拿到确切的值）如果想要广播RDD对应的值，需要将该RDD对应的数据获取到Driver端然后再进行广播，例如collectAsMap等方法。**
- 广播变量的数据不可太大，如果太大，会在Executor占用大量的缓存，相对于计算的时候的缓存就少很多

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/Spark/2-1.jpg" style="zoom:50%;" />

