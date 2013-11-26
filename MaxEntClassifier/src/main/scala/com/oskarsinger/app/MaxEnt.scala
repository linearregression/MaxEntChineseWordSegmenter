package MaxEntClassification

import scala.collection.mutable.Map
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File
import play.api.libs.json._
import cc.factorie.la.DenseTensor1
import cc.factorie.optimize.ConjugateGradient
import cc.factorie.model.{Parameters, Weights}

class MaxEnt {

  def train(dirs:List[String]): Map[Tuple2[String, String], Double] = {
    val classes = dirs
    val model = Map[Tuple2[String, String], Int]()
    val stats = new ArrayBuffer[Tuple2[Int, Double]]()

    classes.foreach{ c =>
      model(c -> "DEFAULT") = 0

      for{
        file <- (new File(c)).listFiles.toList.map(Source.fromFile(_))
        word <- file.mkString.split(" ").map(_.toLowerCase)
      }{
        val feature = c -> word
        if( model.contains(feature) ){
          val old = stats(model(feature))
          stats(model(feature)) = ((old._1 + 1) -> old._2)
        } else {
          stats += (1 -> 0)
          model(feature) = stats.size - 1
        }
      }
    }

    val optimizedLambdas = CGWrapper.fminNCG(value, gradient, stats.toArray)

    (for((feature, index) <- model) 
      yield (feature -> optimizedLambdas(index))
    ).toList.foldLeft(Map[Tuple2[String, String], Double]())(_+_)
  }

  def gradient(stats: Array[Tuple2[Int, Double]]): Array[Double] = 
    stats.toList.map( stat => stat._1 - (stat._2 * stat._1) ).toArray

  def value(stats: Array[Tuple2[Int, Double]]): Double = {
    val totalLogProb = stats.toList.map( stat => stat._1 * stat._2 ).sum
    //TODO: Make these ex-normalized
    //TODO: Figure out how to actually do the Gaussian prior
    val gaussianPrior = stats.toList.map( stat => stat._2 ).toList.map(Math.log(_)).reduceLeft(_+_)

    -(totalLogProb + gaussianPrior)
  }

  def classify(model: Map[Tuple2[String, String], Double], file: File): Map[String, Double] = {
    val classes = model.keys.map( key => key._1 ).toList.distinct
    val scores = (for(c <- classes) 
                    yield (c -> (for(word <- Source.fromFile(file).mkString.split(" ").map(_.toLowerCase)) 
                                   yield model(c -> word)
                                ).toList.foldLeft(model(c -> "DEFAULT"))(_+_)
                          )
                 ).toList.foldLeft(Map[String, Double]())(_+_)

    expNormalize(scores)
  }

  def expNormalize(scores: Map[String, Double]): Map[String, Double] = {
    val minimum = scores.values.min
    val exScores = scores.map( score => score._1 -> Math.exp(score._2 - minimum) )
    val normalizer = exScores.values.sum
    val normScores = exScores.map( score => score._1 -> score._2/normalizer )

    assert( normScores.values.sum == 1 )

    normScores
  }

  def parseDatum(string: String): Tuple3[String, String, Map[String, Boolean]] = {
    val json = Json.parse(string)

    val refDom = (json \ "referer_domain")(0).as[String]
    val hasVisited = (json \ "visited_shared_page")(0).as[Boolean]
    val copyChoices = (json \ "copy_choices")(0).as[List[String]]
    val landingPages = (for(c <- copyChoices) yield ( c -> (json \ c)(1).as[Boolean] )).foldLeft(Map[String, Boolean]())(_+_)
    
    (refDom, (if(hasVisited) "TRUE" else "FALSE"), landingPages)
  }

  private object CGWrapper {
    def fminNCG(value: (Array[Tuple2[Int, Double]]) => Double,
                gradient: (Array[Tuple2[Int, Double]]) => Array[Double],
                stats: Array[Tuple2[Int, Double]]
               ): Array[Double] = {
      val initialWeights = stats.map( stat => stat._2 ).toArray
      val features = stats.map( stat => stat._1 ).toArray
      val model = new Parameters { val weights = Weights(new DenseTensor1(initialWeights.size)) }

      model.weights.value := initialWeights.toArray

      val optimizer = new ConjugateGradient
      val gradientMap = model.parameters.blankDenseMap

      while (!optimizer.isConverged) {
        val newStats = features.toList.zip(model.weights.value.asArray.toList).toArray

        gradientMap(model.weights) = new DenseTensor1(gradient(newStats))

        val currentValue = value(newStats)

        optimizer.step(model.parameters, gradientMap, currentValue)
      }

      model.weights.value.asArray
    }
  }
}
