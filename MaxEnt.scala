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

    val optimizedLambdas = CGWrapper.fminNCG(value, gradient, model)

    model.keys.zip(optimizedLambdas).foldLeft(Map[Tuple2[String, String], Double]())(_+_) 
  }

  def gradient(lambdas: Array[Double], keys: List[Tuple2[String, String]], dirs: List[String]): Array[Double] = {
    val gradient = new Array[Double](55)
    //TODO Implement the gradient
    gradient
  }

  def value(lambdas: Array[Double], keys: List[Tuple2[String, String]], dirs: List[String]): Double = {
    val model = keys.zip(lambdas).foldLeft(Map[Tuple2[String, String], Double]())(_+_)
    val classes = dirs
    val totalLogProb = 0

    -((for{
        c <- classes
        file <- (new File(c)).listFiles.toList
    } yield Math.log(classify(model, classes, file)(c))
    ).toList.reduceLeft(_+_) + lambdas.toList.map(Math.log(_)).reduceLeft(_+_)) //TODO: Figure out how to actually do the Gaussian prior
  }

  def classify(model: Map[Tuple2[String, String], Double], classes: List[String], file: File): Map[String, Double] = {
    val scores = (for(c <- classes) yield (c ->
                                           (for(word <- Source.fromFile(file).mkString.split(" ").map(_.toLowerCase)) yield model(c -> word)
                                           ).toList.foldLeft(model(c -> "DEFAULT"))(_+_)
                                          )
                 ).toList.foldLeft(Map[String, Double]())(_+_)

    val minimum = scores.values.min
    val exScores = scores.map( score => score._1 -> Math.exp(score._2 - minimum) )
    val normalizer = exScores.values.reduceLeft(_+_)

    exScores.map( score => score._1 -> score._2/normalizer ) //.toList.sortWith( (x,y) => x._2 > y._2 )
  }

  private object CGWrapper {
    def fminNCG(value: (Array[Double], List[Tuple2[String, String]], List[String]) => Double,
                gradient: (Array[Double], List[Tuple2[String, String]], List[String]) => Array[Double],
                oldModel: Map[Tuple2[String, String], Double]
               ): List[Double] = {
      val initialWeights = oldModel.values
      val keys = oldModel.keys.toList
      val dirs = keys.map( x => x._1 )
      val model = new Parameters { val weights = Weights(new DenseTensor1(initialWeights.size)) }

      model.weights.value := initialWeights

      val optimizer = new ConjugateGradient
      val gradientMap = model.parameters.blankDenseMap

      while (!optimizer.isConverged) {
        gradientMap(model.weights) = new DenseTensor1(gradient(model.weights.value.asArray, keys, dirs))

        val currentValue = value(model.weights.value.asArray, keys, dirs)

        optimizer.step(model.parameters, gradientMap, currentValue)
      }

      model.weights.value.asArray
    }
  }
}