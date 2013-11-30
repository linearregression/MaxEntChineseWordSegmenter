package MaxEntClassification

import scala.collection.mutable.Map
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File
import cc.factorie.la.DenseTensor1
import cc.factorie.optimize.ConjugateGradient
import cc.factorie.model.{Parameters, Weights}

class MaxEntChineseWordSegmenter {

  def train(filePath: String): Map[(String, String), Double] = {
    val classes = List(filePath)
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

  def getLabeledDataSet(filePath: String): List[(Char, String)] = {
    (for{
       line <- Source.fromFile(filePath).getLines.toList  
       i <- 0 until line.size
       if !isWhiteSpace(line(i))
     } yield getTaggedCharacter(i, line)
    ).toList.foldLeft(List[(Char, String)]())(_:+_)
  }

  def getTaggedCharacter(i: Int, line: String): (Char, String) = {
    val tag =
      if(isFirst(i, line) && isLast(i, line)) "LR"
      else if(isFirst(i, line)) "LL"
      else if(isLast(i, line)) "RR"
      else if(!isWhiteOrPunct(line(i))) "MM"
      else if(isPunctuation(line(i))) "PP"
      else "<INVALID>"
    
    (line(i) -> tag)
  }

  def isFirst(i: Int, line: String): Boolean = (i == 0 || isWhiteOrPunct(line(i-1))) && !isWhiteOrPunct(line(i))

  def isLast(i: Int, line: String): Boolean = (i == (line.size - 1) || isWhiteOrPunct(line(i+1))) && !isWhiteOrPunct(line(i))

  def isWhiteOrPunct(character: Char): Boolean = isPunctuation(character) || isWhiteSpace(character)

  def isPunctuation(character: Char): Boolean = {
    val punctuationChars = 
      List( (0x3000 to 0x303F), (0x2400 to 0x243F), (0xFF00 to 0xFFEF), (0x2000 to 0x206F) )
    
    punctuationChars.exists( _.contains(character) )
  }

  def isWhiteSpace(character: Char): Boolean = (0x0000 to 0x000F).contains(character)

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
