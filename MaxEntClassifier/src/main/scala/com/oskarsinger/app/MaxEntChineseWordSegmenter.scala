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
    val taggedTraining = getLabeledDataSet(filePath).toArray
    val weights = new ArrayBuffer[Double]()
    val model = Map[(String, String), Int]()
    val cache = Map[String, ArrayBuffer[List[String]]]().withDefault( x => new ArrayBuffer[List[String]]() )

    ( 0 until taggedTraining.size ).foreach{ i =>
      val current = taggedTraining(i) 
      val character = current._1.toString
      val tag = current._2
      val features: List[(String, String)] = 
        if( i == 0 ) List( (tag -> character), (tag -> ("NEXT" + taggedTraining(i+1)._1))) 
        else if(i == taggedTraining.size - 1) List( (tag -> ("PREV" + taggedTraining(i-1)._1)), (tag -> character) )
        else List( (tag -> ("PREV" + taggedTraining(i-1)._1)), (tag -> character), (tag -> ("NEXT" + taggedTraining(i+1)._1)) )

      cache(tag) = cache(tag)
      cache(tag) += features.map( feature => feature._2 )

      features.foreach{ feature =>
        if(!model.contains(feature)){
          weights += 0.0
          model(feature) = weights.size - 1
        }
      }
    }

    cache.keys.foreach{ tag =>
      weights += 0.0
      model(tag -> "DEFAULT") = weights.size
    }

    val listCache = cache.map( tagEntry => tagEntry._1 -> tagEntry._2.toList )
    val optimizedWeights = CGWrapper.fminNCG(value, gradient, listCache, model, weights.toArray)

    (for((feature, index) <- model) 
      yield (feature -> optimizedWeights(index))
    ).toList.foldLeft(Map[Tuple2[String, String], Double]())(_+_)
  }

  def getLabeledDataSet(filePath: String): List[(Char, String)] = {
    (for{
       line <- Source.fromFile(filePath).getLines.toList  
       i <- 0 until line.size
       if !isWhiteSpace(line(i))
     } yield getTaggedCharacter(i, line)
    ).toList.foldRight(List[(Char, String)]())(_+:_)
  }

  def getTaggedCharacter(i: Int, line: String): (Char, String) = {
    val tag =
      if(isFirst(i, line) && isLast(i, line)) "LR"
      else if(isFirst(i, line)) "LL"
      else if(isLast(i, line)) "RR"
      else if(!isPunctuation(line(i))) "MM"
      else if(isPunctuation(line(i))) "PP"
      else "<INVALID>"
    
    (line(i) -> tag)
  }

  def isFirst(i: Int, line: String): Boolean = (i == 0 || isWhiteSpace(line(i-1))) && !isPunctuation(line(i))

  def isLast(i: Int, line: String): Boolean = (i == (line.size - 1) || isWhiteSpace(line(i+1))) && !isPunctuation(line(i))

  def isPunctuation(character: Char): Boolean = {
    val punctuationChars = 
      List( (0x3000, 0x303F), (0x2400, 0x243F), (0xFF00, 0xFF0F), (0xFF1A, 0xFFEF), (0x2000, 0x206F) )
    
    punctuationChars.exists( range => character >= range._1 && character <= range._2 )
  }

  def isWhiteSpace(character: Char): Boolean = List( (0x0000, 0x0020) ).exists( range => character >= range._1 && character <= range._2)

  def gradient(cache: Map[String, List[List[String]]], model: Map[(String, String), Int], weights: Array[Double]): Array[Double] = {
    val classes = cache.keys.toList
    val featureCounts = new Array[Double](weights.size)

    for{
      (tag, entries) <- cache
      entry <- entries
      feature <- entry
    } featureCounts(model(tag -> feature)) += 1
    
    val probability =
      (for(tag <- classes)
         yield (tag -> (for{
                          entries <- cache.values
                          entry <- entries
                        } yield tagScores(entry, model, weights, classes)(tag)
                       ).toList.reduceLeft(_+_)
               )
      ).toList.foldLeft(Map[String, Double]())(_+_)

    for{
      (feature, index) <- model
      tag = feature._1
    } featureCounts(index) = featureCounts(index) - (probability(tag) * featureCounts(index)) - (weights(index) / 2)
     
    featureCounts
  }

  def value(cache: Map[String, List[List[String]]], model: Map[(String, String), Int], weights: Array[Double]): Double = {
    val classes = cache.keys.toList
    val totalLogProb =
      (for{
         (tag, entries) <- cache
         entry <- entries
       } yield Math.log(tagScores(entry, model, weights, classes)(tag))
      ).toList.reduceLeft(_+_)
    val gaussianPrior = weights.toList.map( weight => weight * weight ).sum
    
    -(totalLogProb - gaussianPrior)
  }

  def tagScores(features: List[String], model: Map[(String, String), Int], weights: Array[Double], classes: List[String]): Map[String, Double] = {
    val scores =
      (for( c <- classes )
         yield (c -> (for( feature <- features )
                        yield (if(model.contains(c -> feature)) weights(model(c -> feature)) else 0.0)
                     ).toList.foldLeft(weights(model(c -> "DEFAULT")))(_+_)
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
    def fminNCG(value: (Map[String, List[List[String]]], Map[(String, String), Int], Array[Double]) => Double,
                gradient: (Map[String, List[List[String]]], Map[(String, String), Int], Array[Double]) => Array[Double],
                cache: Map[String, List[List[String]]],
                featureMap: Map[(String, String), Int],
                initialWeights: Array[Double]
               ): Array[Double] = {
      val model = new Parameters { val weights = Weights(new DenseTensor1(initialWeights.size)) }

      model.weights.value := initialWeights

      val optimizer = new ConjugateGradient
      val gradientMap = model.parameters.blankDenseMap

      while (!optimizer.isConverged) {
        gradientMap(model.weights) = new DenseTensor1(gradient(cache, featureMap, model.weights.value.asArray))

        val currentValue = value(cache, featureMap, model.weights.value.asArray)

        optimizer.step(model.parameters, gradientMap, currentValue)
      }

      model.weights.value.asArray
    }
  }
}
