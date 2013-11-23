import scala.collection.mutable.Map
import scala.io.Source
import java.io.File
import cc.factorie.la.DenseTensor1
import cc.factorie.optimize.ConjugateGradient
import cc.factorie.model.{Parameters, Weights}

class MaxEnt {

  def train(dirs:List[String]): Map[Tuple2[String, String], Double] = {
    val classes = dirs
    val model = Map[Tuple2[String, String], Double]().withDefaultValue(0)
    val constraints = Map[Tuple2[String, String], Double]().withDefaultValue(0)

    classes.foreach{ c =>
      model(c -> "DEFAULT") = 0

      for{
        file <- (new File(c)).listFiles.toList.map(Source.fromFile(_))
        word <- file.mkString.split(" ").map(_.toLowerCase)
      }{
        constraints(cls -> word) += 1
        classes.foreach( cls => model(cls -> word) = 0 )
      }
    }

    val originalLambdas = model.values.toArray
    val optimizedLambdas = CGWrapper.fminNCG(value, gradient, originalLambdas)

    model.keys.zip(optimizedLambdas).foldLeft(Map[Tuple2[String, String], Double]())(_+_) 
  }

  def gradient(lambdas: Array[Double], keys: Array[Tuple2[String, String]], dirs: Array[String]): Array[Double] = {
    val gradient = new Array[Double](55)
    //TODO Implement the gradient
    gradient
  }

  def value(lambdas: Array[Double], keys: Array[Tuple2[String, String]], dirs: Array[String]): Double = {
    val model = keys.zip(lambdas).foldLeft(Map[Tuple2[String, String], Double]())(_+_)
    val classes = dirs
    val totalLogProb = 0

    -((for{
        c <- classes
        file <- (new File(c)).listFiles.toList
    } yield Math.log(classify(model, classes, file)(c))
    ).toList.reduceLeft(_+_) + lambdas.toList.map(Math.log(_)).reduceLeft(_+_))
  }

  def classify(model: Map[Tuple2[String, String], Double], classes: Array[String], file: File): Map[String, Double] = {
    val scores = (for(c <- classes) yield (c ->
                                           (for(word <- Source.fromFile(file).mkString.split(" ").map(_.toLowerCase)) yield model(c -> word)
                                           ).toList.foldLeft(model(c -> "DEFAULT"))(_+_)
                                          )
                 ).toList.foldLeft(Map[String, Double]())(_+_)

    val minimum = scores.values.min
    val exScores = scores.mapValues( score => Math.exp(score - minimum) )
    val normalizer = exScores.values.reduceLeft(_+_)

    exScores.mapValues( score => score/normalizer ) //.toList.sortWith( (x,y) => x._2 > y._2 )
  }

  private object CGWrapper {
    def fminNCG(value: (Array[Double], Array[Tuple2[String, String]], Array[String]) => Double,
                gradient: (Array[Double], Array[Tuple2[String, String]], Array[String]) => Array[Double],
                initialWeights: Array[Double]
               ): Array[Double] = {
      val model = new Parameters { val weights = Weights(new DenseTensor1(initialWeights.size)) }

      model.weights.value := initialWeights

      val optimizer = new ConjugateGradient
      val gradientMap = model.parameters.blankDenseMap

      while (!optimizer.isConverged) {
        gradientMap(model.weights) = new DenseTensor1(gradient(model.weights.value.toArray))

        val currentValue = value(model.weights.value.toArray)

        optimizer.step(model.parameters, gradientMap, currentValue)
      }

      model.weights.value.toArray
    }
  }
}
