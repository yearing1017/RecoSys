### Spark Mllib Als

```scala
import com.xiaomi.bigdata.angelia.game.etl.KnightsGameInfoExtract
import com.xiaomi.bigdata.angelia.spark.SparkUtil
import com.xiaomi.bigdata.angelia.utils.{ArgumentParser, DateUtil, HDFSUtil}
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD


object GameAls {

  val logger: Logger = Logger.getLogger(GameAls.getClass)
  var modulePathRoot: String = _

  def train(userItemScore: RDD[(String, String, Double)]): (MatrixFactorizationModel, RDD[(String, Int)], RDD[(String, Int)]) = {
    val sc = userItemScore.sparkContext
    HDFSUtil.deletePath(sc, modulePathRoot)

    val (uidGidScore, userIndex, gameIndex) = UserGameScore.mappingIndex(userItemScore, false, false)
    val Array(train_date, test_date) = uidGidScore.map{ case (u, g, s) => Rating(u.toInt, g.toInt, s.toFloat) }.randomSplit(Array(0.9, 0.1))

    val game_model = ALS.train(train_date.cache(), 40, 20)
    val testUserGame = test_date.map(r => (r.user, r.product))
    val predictions = game_model.predict(testUserGame)
      .map { case Rating(user, product, rate) => ((user, product), rate) }
    val MSE = test_date.map { case Rating(user, product, rate) => ((user, product), rate) }
      .join(predictions)
      .map { case (_, (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean()
    logger.error("TRAIN MSE: " + MSE)

    game_model.save(sc, modulePathRoot + "/model")
    userIndex.map { case (user, idx) => user + "\t" + idx }.repartition(10).saveAsTextFile(modulePathRoot + "/user_index")
    gameIndex.map { case (game, idx) => game + "\t" + idx }.repartition(1).saveAsTextFile(modulePathRoot + "/game_index")
    (game_model, userIndex.mapValues(_.toInt), gameIndex.mapValues(_.toInt))
  }

  def loadModule(sc: SparkContext, modelRoot: String): (MatrixFactorizationModel, RDD[(String, Int)], RDD[(String, Int)]) = {
    val userIndex = sc.textFile(modelRoot + "/user_index").map(l => l.split("\t")).filter(_.length == 2).map(l => (l(0), l(1).toInt))
    val gameIndex = sc.textFile(modelRoot + "/game_index").map(l => l.split("\t")).filter(_.length == 2).map(l => (l(0), l(1).toInt))
    val module = MatrixFactorizationModel.load(sc, modelRoot + "/model")
    (module, userIndex, gameIndex)
  }

  def predict(userSet: RDD[String], module: MatrixFactorizationModel, userIndex: RDD[(String, Int)], pkgIndex: RDD[(String, Int)]): Unit = {
    doPredict(userSet, module, userIndex, pkgIndex)
      .map { case (uid, recItems) =>
        uid + "\t" + recItems.map { case (i, d) => f"$i:$d%.6f" }.mkString(" ")
      }
      .saveAsTextFile(modulePathRoot + "/predict")
  }

  private def doPredict(userSet: RDD[String], module: MatrixFactorizationModel, userIndex: RDD[(String, Int)], pkgIndex: RDD[(String, Int)]): RDD[(String, List[(Long, Double)])] = {
    //load被预估的ItemCF候选集合
    val sc = userSet.sparkContext
    val pkgIndexMap = pkgIndex.collectAsMap()

    val vaildGameFeatures = module.productFeatures

    val vaildUser = userSet.map(l => (l, true)).join(userIndex).map { case (_, (_, idx)) => (idx, true) }
    val userFeatures = module.userFeatures.join(vaildUser).map { case (idx, (arr, _)) => (idx, arr) }
    val rank = vaildGameFeatures.map(l => l._2.length).max()
    logger.error("predict rank is: " + rank)
    val vaildGameModule = new MatrixFactorizationModel(rank, userFeatures, vaildGameFeatures)
    val indexUser = userIndex.map(l => (l._2, l._1))
    vaildGameModule.recommendProductsForUsers(20).join(indexUser)
      .map {case (_, (recItems, uid)) =>
        (uid, recItems.map(r => (r.product.toLong, r.rating)).toList.sortBy(_._2)(Ordering.Double.reverse))
      }.filter(_._2.nonEmpty)
  }

  def loadPredict(sc: SparkContext, path: String): RDD[(String, Array[(Int, Double)])] = {
    sc.textFile(path).map(l => {
      val Array(item, recItems) = l.split("\t")
      val recs = recItems.split(" ")
        .map(recScore => {
          val Array(recItem, score) = recScore.split(":")
          (recItem.toInt, score.toDouble)
        })
      (item, recs)
    })
  }

  def recall(userSet: RDD[String], recallCount: Int = 20) = {
    val sc = userSet.sparkContext
    val modelPath = HDFSUtil.getLatestValidPathByDate(sc, "/user/s_youpin_search/game_push/ALSModule/miui")
    val (model, userIndex, gameIndex) = loadModule(sc, modelPath)
    doPredict(userSet, model, userIndex, gameIndex).mapValues(_.take(recallCount))
  }

  def main(args: Array[String]): Unit = {
    val sc = SparkUtil.createSparkContext(GameAls.getClass.getName)
    val ss = SparkUtil.createSparkSession(GameAls.getClass.getName)
    val argMap = ArgumentParser.parse(args)
    val sourceType = argMap.getOrElse("source", "game")
    val endDay = argMap.getOrElse("endDay", DateUtil.getYesterdayDateString)
    val days = argMap.getOrElse("days", "1").toInt
    val similar = argMap.contains("similar")
    modulePathRoot = argMap.getOrElse("outputRoot", "/user/s_youpin_search/game_push/ALSModule") + "/" + sourceType + "/date=" + endDay
    val userItemScore = UserGameScore.getUserItemScore(ss, sourceType, endDay, days)
    val (module, userIndex, gameIndex) = if (argMap.contains("train")) {
      if (similar) {
        train(UserGameScore.mappingSimilarPackage(UserGameScore.filterUserItemScore(userItemScore)))
      } else {
        train(UserGameScore.filterUserItemScore(userItemScore))
      }
    } else {
      loadModule(sc, modulePathRoot)
    }
    if (argMap.contains("predict")) {
      val usersetPath = argMap.getOrElse("userset", "")
      val userset = if (usersetPath.nonEmpty) sc.textFile(usersetPath) else userIndex.map(_._1)
      predict(userset, module, userIndex, gameIndex)
    }
  }
}

```

#### train

- 先获取打分矩阵 具体格式：`RDD[(uid, gid, score)]`

- 相似包名映射

- 打分矩阵到index的映射

  - ```scala
    val (uidGidScore, userIndex, gameIndex) = UserGameScore.mappingIndex(userItemScore, false, false)
    ```

- 9:1 随机划分训练+验证集

  - ```scala
    val Array(train_date, test_date) = uidGidScore.map{ case (u, g, s) => Rating(u.toInt, g.toInt, s.toFloat) }.randomSplit(Array(0.9, 0.1))
    ```

-  Als.train  ==> 得到 `MatrixFactorizationModel` 格式的模型

  - ```scala
    // rank参数 40 代表隐向量的维度 iterations 20 迭代的次数
    val game_model = ALS.train(train_date.cache(), 40, 20)
    ```

- 验证集验证；计算在验证集上的MSE损失

  - ```scala
    	val testUserGame = test_date.map(r => (r.user, r.product))
        val predictions = game_model.predict(testUserGame)
          .map { case Rating(user, product, rate) => ((user, product), rate) }
        val MSE = test_date.map { case Rating(user, product, rate) => ((user, product), rate) }
          .join(predictions)
          .map { case (_, (r1, r2)) =>
            val err = (r1 - r2)
            err * err
          }.mean()
        logger.error("TRAIN MSE: " + MSE)
    ```

- 保存模型 + index映射关系

#### loadModel

```scala
def loadModule(sc: SparkContext, modelRoot: String): (MatrixFactorizationModel, RDD[(String, Int)], RDD[(String, Int)]) = {
    val userIndex = sc.textFile(modelRoot + "/user_index").map(l => l.split("\t")).filter(_.length == 2)
    	.map(l => (l(0),l(1).toInt))
    val gameIndex = sc.textFile(modelRoot + "/game_index").map(l => l.split("\t")).filter(_.length == 2)
    	.map(l => (l(0),l(1).toInt))
    val module = MatrixFactorizationModel.load(sc, modelRoot + "/model")
    (module, userIndex, gameIndex)
}
```

#### predict

```scala
private def doPredict(userSet: RDD[String], module: MatrixFactorizationModel, userIndex: RDD[(String, Int)], pkgIndex: 		RDD[(String, Int)]): RDD[(String, List[(Long, Double)])] = {
    //load被预估的ItemCF候选集合
    val sc = userSet.sparkContext
    val pkgIndexMap = pkgIndex.collectAsMap()
	// RDD[(Int, Array[Double])] itemIndex + 其隐向量
    val vaildGameFeatures = module.productFeatures
	// jion 找到有效用户
    val vaildUser = userSet.map(l => (l, true)).join(userIndex).map { case (_, (_, idx)) => (idx, true) }
    // RDD[(Int, Array[Double])] 有效用户的userIndex + 其隐向量
    val userFeatures = module.userFeatures.join(vaildUser).map { case (idx, (arr, _)) => (idx, arr) }
    // 隐向量维度
    val rank = vaildGameFeatures.map(l => l._2.length).max()
    logger.error("predict rank is: " + rank)
    val vaildGameModule = new MatrixFactorizationModel(rank, userFeatures, vaildGameFeatures)
    val indexUser = userIndex.map(l => (l._2, l._1))
    // 每个用户最值得推荐的20个物品
    vaildGameModule.recommendProductsForUsers(20).join(indexUser)
      .map {case (_, (recItems, uid)) =>
        (uid, recItems.map(r => (r.product.toLong, r.rating)).toList.sortBy(_._2)(Ordering.Double.reverse))
      }.filter(_._2.nonEmpty)
  }
```

