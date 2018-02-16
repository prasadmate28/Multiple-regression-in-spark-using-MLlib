import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.LabeledPoint


object LinearRegressionAnalysis extends App {


  import org.apache.spark.{SparkConf,SparkContext}
  import org.apache.spark.ml.regression.LinearRegression
  import org.apache.spark.sql.SparkSession

  def buildModel (fileName:String): LinearRegressionModel ={

    val conf = new SparkConf().setAppName("Linear regression Test").setMaster("local[*]")
    val sc = new SparkContext(conf)
    println(sc)

    val dataFile = spark.read.format("csv")
      .option("header","true")
      .option("inferSchema","true")
      .load("/home/prasad/R/Workspace/Project1/dataset-p1.csv") //replace with args(0)


    dataFile.show(10) // display first 10 entries

    val lr = new LinearRegression()
    val model = lr.fit(dataFile)
    return model
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


  val spark = SparkSession.builder.master("local")
    .appName("spark session example")
    .getOrCreate()


  //var fileName:String = args(0)
  var fileName:String = "/home/prasad/R/Workspace/Project1/dataset-p1.csv"

  val model = buildModel(fileName)

  /*val dataFile = spark.read.format("csv")
    .option("header","true")
    .option("inferSchema","true")
    .load("/home/prasad/R/Workspace/Project1/dataset-p1.csv") //replace with args(0)*/


  displaySummaryStatistics(model)


  // add column transformation
  def transform (x:Double):Double={
    return x*x
  }


}
