import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.udf

object LinearRegressionAnalysis extends App {

  val spark = SparkSession.builder.master("local")
    .appName("Linear Regression Analysis")
    .getOrCreate()

  def loadTrainingData(data:DataFrame)():DataFrame={
    //val featureCols = Array("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15")
    val featureCols = data.schema.fieldNames.dropRight(1)
    val classLabel = data.schema.fieldNames.last

    val assembler =  new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val df2 = assembler.transform(data)
    val labelIndexer = new StringIndexer().setInputCol(classLabel).setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)
    return df3
  }

  def buildModel(trainingData: DataFrame)={
    val lr = new LinearRegression()

    val model = lr.fit(trainingData)

    displaySummaryStatistics(model)
  }

  def displaySummaryStatistics (model:LinearRegressionModel):Unit = {
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = model.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")


  }

  def calcVIF()={

  }

  var fileName:String = "dataset-p1.csv"
  //var fileName:String = args(0)

  val dataFile = spark.read.format("csv")
    .option("header","true")
    .option("inferSchema","true")
    .load(fileName) //replace with args(0)

  //  Run linear regression model with given dataset
  val trainingData = loadTrainingData(dataFile)

  buildModel(trainingData)

  // Augment X4^2 column to the data and run linear regression
  val squaredColumn : (Double) => Double = (x4:Double) => math.pow(x4,2)

  val addColumnUDF = udf(squaredColumn)

  val transformed_dataFile = dataFile.withColumn("X4sq",addColumnUDF(dataFile.col("X4")))

  val trainingDataAugmented = loadTrainingData(transformed_dataFile)

  buildModel(trainingDataAugmented)

  // removing redundunt features

  


}
