import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf


object LinearRegressionAnalysis extends App {

  val spark = SparkSession.builder.master("local")
    .appName("Linear Regression Analysis")
    .getOrCreate()

  def loadTrainingData(data:DataFrame, featureCols: Array[String], classLabel: String):DataFrame={
    //val featureCols = Array("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15")
    //val featureCols = data.schema.fieldNames.drop(data.schema.fieldNames.length)
    //dropRight(1)
    //val classLabel = data.schema.fieldNames.last
    //val classLabel = "Y"
    println("Features :" + featureCols.toList)
    println("last col "+classLabel)
    val assembler =  new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val df2 = assembler.transform(data)

    println("First 5 rows")
    df2.show(5)
    //df2.select("features").show(20,false)
    return df2
  }

  def buildModel(trainingData: DataFrame): LinearRegressionModel ={
    val lr = new LinearRegression().setFeaturesCol("features").setLabelCol("Y")
      //.setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    val model = lr.fit(trainingData)

    displaySummaryStatistics(model)

    return model
  }

  def displaySummaryStatistics (model:LinearRegressionModel):Unit = {
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")
    // Summarize the model over the training set and print out some metrics
    val trainingSummary = model.summary

    //println(s"numIterations: ${trainingSummary.totalIterations}")
    //println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
    println(s"Number of instances: ${trainingSummary.numInstances}")
    println(s"Degrees of freedom: ${trainingSummary.degreesOfFreedom}")
    println(s"MSE: ${trainingSummary.meanSquaredError}")
    print(s"SSE: ")
    print(trainingSummary.meanSquaredError * trainingSummary.numInstances +s"\n")
    //trainingSummary.residuals.show()
    println(s"Std errors: ${trainingSummary.coefficientStandardErrors.toList}")
    println(s"tValues: ${trainingSummary.tValues.toList}")
    println(s"pValues: ${trainingSummary.pValues.toList}")

  }

  def calcVIF(trainingData: DataFrame, model: LinearRegressionModel) = {

  }

  var fileName:String = "dataset-p1.csv"
  //var fileName:String = args(0)

  val dataFile = spark.read.format("csv")
    .option("header","true")
    .option("inferSchema","true")
    .load(fileName) //replace with args(0)

  val classLabel = dataFile.schema.fieldNames.last

  //var labelIndex = dataFile.schema.fieldIndex(classLabel)
  var featureCols = Array("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15")
    //dataFile.schema.fieldNames.drop(labelIndex)

  //  Run linear regression model with given dataset
  val trainingData = loadTrainingData(dataFile, featureCols, classLabel)
  println("======== Linear Regression --================")
  val model = buildModel(trainingData)

  println(" -~~~~~~~~   Calculating VIF  ~~~~~~~~~~")
  //calcVIF(trainingData, model)

  // Augment X4^2 column to the data and run linear regression
  val squaredColumn : (Double) => Double = (x4:Double) => math.pow(x4,2)
  val addColumnUDF = udf(squaredColumn)
  var transformed_dataFile = dataFile.withColumn("X4sq",addColumnUDF(dataFile.col("X4")))

  featureCols = Array("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X4sq")
    //dataFile.schema.fieldNames.drop(labelIndex)
  transformed_dataFile.show(5)
  val trainingDataAugmented = loadTrainingData(transformed_dataFile,featureCols,classLabel)
  println("======== Polynomial Regression --================")
  buildModel(trainingDataAugmented)

  // removing redundunt features
  transformed_dataFile = dataFile.select("X1", "X2", "X4", "X5", "X7", "X8", "X9", "X10", "X12", "X14", "X15", "Y")
  transformed_dataFile.show(5)
  //labelIndex = dataFile.schema.fieldIndex(classLabel)
  featureCols = Array("X1", "X2", "X4", "X5", "X7", "X8", "X9", "X10", "X12", "X14", "X15")
    //dataFile.schema.fieldNames.drop(labelIndex)
  val trainingData2 = loadTrainingData(transformed_dataFile,featureCols,classLabel)

  println("======== Reduced feature Regression ================")
  buildModel(trainingData2)

}
