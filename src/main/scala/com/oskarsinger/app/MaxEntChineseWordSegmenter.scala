package com.oskarsinger.app

import scala.collection.mutable.Map
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File
import cc.factorie.la.DenseTensor1
import cc.factorie.optimize.ConjugateGradient
import cc.factorie.model.{Parameters, Weights}

class MaxEntChineseWordSegmenter {

  //Returns a list of 2-tuples that holds the characters of a phrase
  //original order mapped to their maxent-classified tags
  def segment(text: String, model: Map[(String, String), Double]): List[(String, String)] = {
    val characters = text.toList.map( character => character.toString ).toArray
    val weightsBuffer = new ArrayBuffer[Double]()
    val featureMap = Map[(String, String), Int]()
    val classes = model.keys.map( key => key._1 ).toList.distinct

    model.foreach{ case (feature, weight) =>
      weightsBuffer += weight
      featureMap(feature) = weightsBuffer.size - 1
    }
    
    assert( weightsBuffer.size == model.size )
    assert( featureMap.size == model.size )

    val weights = weightsBuffer.toArray

    val taggedData =
      (for( i <- 0 until characters.size )
         yield (characters(i) -> 
                  tagScores(getFeatures(i, characters), featureMap, weights, classes).toList.sortWith( (x,y) => x._2 > y._2 )(0)._1
               )
      ).toList.foldRight(List[(String, String)]())(_+:_)

    assert( taggedData.size == characters.size )

    taggedData
  }

  def train(filePath: String): Map[(String, String), Double] = {
    val taggedTraining = getLabeledDataSet(filePath).toArray
    val trainingChars = taggedTraining.map( word => word._1 )

    assert(taggedTraining.size >= 1)
    assert(taggedTraining.size == trainingChars.size)

    val weights = new ArrayBuffer[Double]()
    val model = Map[(String, String), Int]()
    val cache = Map[String, ArrayBuffer[List[String]]]().withDefault( x => new ArrayBuffer[List[String]]() )

    ( 0 until taggedTraining.size ).foreach{ i =>
      val current = taggedTraining(i) 
      val character = current._1.toString
      val tag = current._2
      val features: List[String] = getFeatures(i, trainingChars)

      cache(tag) = cache(tag)
      cache(tag) += features

      features.foreach{ feature =>
        if(!model.contains(tag -> feature)){
          weights += 0.0
          model(tag -> feature) = weights.size - 1
        }
      }
    }

    assert( cache.keys.size == 5 )
    assert( cache.values.reduceLeft(_++_).size == taggedTraining.size )

    cache.keys.foreach{ tag =>
      weights += 0.0
      model(tag -> "DEFAULT") = weights.size - 1
    }

    assert(weights.size == model.size)

    val listCache = cache.map( tagEntry => tagEntry._1 -> tagEntry._2.toList )

    assert(listCache.values.reduceLeft(_++_).size == taggedTraining.size)

    val optimizedWeights = CGWrapper.fmaxNCG(value, gradient, listCache, model, weights.toArray)

    val newModel =
      (for((feature, index) <- model) 
        yield (feature -> optimizedWeights(index))
      ).toList.foldLeft(Map[Tuple2[String, String], Double]())(_+_)

    assert( newModel.size == model.size )

    newModel
  }
  
  //Returns the list of features for a character in an unsegmented data set
  def getFeatures(i: Int, characters: Array[String]): List[String] = {
    val features =
      if( i == 0 ) List( characters(i), ("NEXT" + characters(i+1))) 
      else if(i == characters.size - 1) List( ("PREV" + characters(i-1)), characters(i) )
      else List( ("PREV" + characters(i-1)), characters(i), ("NEXT" + characters(i+1)) )
    
    assert(features.size >= 2)

    features
  }

  //Labels a pre-segmented training set based on this tag set: 
  //LL (first) MM (middle) RR (right) LR (solitary) PP (punctuation)
  def getLabeledDataSet(filePath: String): List[(String, String)] = {
    (for{
       line <- Source.fromFile(filePath).getLines.toList  
       i <- 0 until line.size
       if !isWhiteSpace(line(i))
     } yield getTaggedCharacter(i, line)
    ).toList.foldRight(List[(String, String)]())(_+:_)
  }

  //Returns a 2-tuple of an instance of character from a training set mapped to its tag
  def getTaggedCharacter(i: Int, line: String): (String, String) = {
    val tag =
      if(isFirst(i, line) && isLast(i, line)) "LR"
      else if(isFirst(i, line)) "LL"
      else if(isLast(i, line)) "RR"
      else if(!isPunctuation(line(i))) "MM"
      else if(isPunctuation(line(i))) "PP"
      else "<INVALID>"
    
    (line.slice(i, i+1) -> tag)
  }

  //Checks if a character in a training set is first in a word
  def isFirst(i: Int, line: String): Boolean = (i == 0 || isWhiteSpace(line(i-1))) && !isPunctuation(line(i))

  //Checks if a character in a training set is last in a word
  def isLast(i: Int, line: String): Boolean = (i == (line.size - 1) || isWhiteSpace(line(i+1))) && !isPunctuation(line(i))

  def isPunctuation(character: Char): Boolean = {
    val punctuationChars = 
      List( (0x3000, 0x303F), (0x2400, 0x243F), (0xFF00, 0xFF0F), (0xFF1A, 0xFFEF), (0x2000, 0x206F) )
    
    punctuationChars.exists( range => character >= range._1 && character <= range._2 )
  }

  def isWhiteSpace(character: Char): Boolean = List( (0x0000, 0x0020) ).exists( range => character >= range._1 && character <= range._2)

  def gradient(cache: Map[String, List[List[String]]], 
               model: Map[(String, String), Int], 
               weights: Array[Double],
               globalFeatureCounts: Array[Double]
              ): Array[Double] = {
    val classes = cache.keys.toList
    
    for{
      (tag, entries) <- cache
      entry <- entries
      score = tagScores(entry, model, weights, classes)(tag)
      feature <- entry
      index = model(tag -> feature)
    } globalFeatureCounts(index) = globalFeatureCounts(index) - score

    for( (feature, index) <- model )
      globalFeatureCounts(index) = globalFeatureCounts(index) - (weights(index) * 2)
     
    globalFeatureCounts
  }

  def value(cache: Map[String, List[List[String]]], model: Map[(String, String), Int], weights: Array[Double]): Double = {
    val classes = cache.keys.toList
    val totalLogProb =
      (for{
         (tag, entries) <- cache
         entry <- entries
       } yield Math.log(tagScores(entry, model, weights, classes)(tag))
      ).toList.sum
    val gaussianPrior = weights.toList.map( weight => weight * weight ).sum
    
    (totalLogProb - gaussianPrior)
  }

  //Gives a map from tags to exponentially normalized scores for a character based on its features and the corresponding weights
  def tagScores(features: List[String], model: Map[(String, String), Int], weights: Array[Double], classes: List[String]): Map[String, Double] = {
    val scores =
      (for( c <- classes )
         yield (c -> (for( feature <- features )
                        yield (if(model.contains(c -> feature)) weights(model(c -> feature)) else 0.0)
                     ).toList.foldLeft(weights(model(c -> "DEFAULT")))(_+_)
               )
      ).toList.foldLeft(Map[String, Double]())(_+_)

    assert( scores.size == classes.size )

    expNormalize(scores)
  }

  //Exponentially normalizes the scores for each tag
  def expNormalize(scores: Map[String, Double]): Map[String, Double] = {
    val minimum = scores.values.min
    val exScores = scores.map( score => score._1 -> Math.exp(score._2 - minimum) * 1.0 )
    val normalizer = exScores.values.sum * 1.0
    val normScores = exScores.map( score => score._1 -> score._2/normalizer )
    val scoreSum = normScores.values.sum

    //assert( normScores.values.sum == 1 )

    normScores
  }

  private object CGWrapper {
    def fmaxNCG(value: (Map[String, List[List[String]]], 
                        Map[(String, String), Int], 
                        Array[Double]
                       ) => Double,
                gradient: (Map[String, List[List[String]]], 
                           Map[(String, String), Int], 
                           Array[Double], 
                           Array[Double]
                          ) => Array[Double],
                cache: Map[String, List[List[String]]],
                featureMap: Map[(String, String), Int],
                initialWeights: Array[Double]
               ): Array[Double] = {
      val model = new Parameters { val weights = Weights(new DenseTensor1(initialWeights.size)) }
      val featureCounts = new Array[Double](initialWeights.size)

      for{
        (tag, entries) <- cache
        entry <- entries
        feature <- entry
      } featureCounts(featureMap(tag -> feature)) += 1

      model.weights.value := initialWeights

      val optimizer = new ConjugateGradient
      val gradientMap = model.parameters.blankDenseMap

      while (!optimizer.isConverged) {
        gradientMap(model.weights) = new DenseTensor1(gradient(cache, featureMap, model.weights.value.asArray, featureCounts))

        val currentValue = value(cache, featureMap, model.weights.value.asArray)
        println(currentValue)

        optimizer.step(model.parameters, gradientMap, currentValue)
      }

      model.weights.value.asArray
    }
  }
}
