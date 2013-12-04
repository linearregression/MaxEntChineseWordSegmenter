package com.oskarsinger.app

import scala.collection.mutable.Map
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File
import cc.factorie.la._
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
                  scoreChar(getFeatures(i, characters), featureMap, weights, classes).toList.sortWith( (x,y) => x._2 > y._2 )(0)._1
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

    val uniqueFeatures = scala.collection.mutable.Set[String]()
    val cache = Map[(String, Int), ArrayBuffer[List[String]]]().withDefault( x => new ArrayBuffer[List[String]]() )
    val tagToIndex = Map[String, Int]()

    ( 0 until taggedTraining.size ).foreach{ i =>
      val current = taggedTraining(i) 
      val character = current._1.toString
      val tag = current._2
      val maximum = if(tagToIndex.isEmpty) -1 else tagToIndex.values.max
      
      if(!tagToIndex.contains(tag)) tagToIndex(tag) = maximum + 1

      val cacheTag = tag -> tagToIndex(tag)
      val features: List[String] = getFeatures(i, trainingChars)

      uniqueFeatures ++= features
      cache(cacheTag) = cache(cacheTag)
      cache(cacheTag) += features
    }

    var count = 0
    val model =
      (for{
         (tag, index) <- cache.keys.toList.sortWith( (x,y) => x._2 < y._2 )
         feature <- uniqueFeatures.toList
       } yield {
           val temp = count
           count += 1
           (tag -> feature) -> temp
       }
      ).toList.foldLeft(Map[(String, String), Int]())(_+_)

    val weights = ArrayBuffer.fill(model.size)(0.0)

    assert( cache.keys.size == 5 )
    assert( cache.values.reduceLeft(_++_).size == taggedTraining.size )
    
    val numFeatures = uniqueFeatures.size
    val listCache = cache.map( 
                      tagEntry => tagEntry._1 -> tagEntry._2.toList.map( 
                        entry => entry.map( 
                          feature => model(tagEntry._1._1 -> feature) ) ) )

    assert(listCache.values.reduceLeft(_++_).size == taggedTraining.size)

    val optimizedWeights = CGWrapper.fminNCG(value, gradient, listCache, model, weights.toArray)

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

  def gradient(fullSparseCache: SparseBinaryTensor2,
               probTables: Map[(String, Int), Array[Array[Double]]], 
               model: Map[(String, String), Int],
               weights: Array[Double], 
               globalFeatureCounts: Array[Double]
              ): Array[Double] = {
    val fullProbTable = new DenseTensor2(probTables.values.reduceLeft(_++_))
    val transProbTable = fullProbTable.transpose
    val expectations = fullSparseCache.leftMultiply(transProbTable).transpose.asArray.reduceLeft(_++_)

    for{
      (feature, index) <- model
      tag = feature._1
    } globalFeatureCounts(index) = globalFeatureCounts(index) - expectations(index) - (weights(index) / 2)
     
    globalFeatureCounts
  }

  def value(probTables: Map[(String, Int), Array[Array[Double]]], weights: Array[Double]): Double = {
    val totalLogProb =
      (for{
         (key, entries) <- probTables
         index = key._2
         entry <- entries
       } yield Math.log(entry(index))
      ).toList.sum
    val gaussianPrior = weights.toList.map( weight => weight * weight ).sum
    
    (totalLogProb - gaussianPrior)
  }

  //Gives a map from tags to exponentially normalized scores for a character based on its features and the corresponding weights
  def scoreChar(features: List[String], model: Map[(String, String), Int], weights: Array[Double], classes: List[String]): Map[String, Double] = {
    val scores =
      (for( c <- classes )
         yield (c -> ( for( feature <- features ) 
                         yield (if(model.contains(c -> feature)) weights(model(c->feature)) else 0.0) 
                     ).toList.sum
               )
      ).toList.foldLeft(Map[String, Double]())(_+_)

    assert( scores.size == classes.size )

    expNormalize(scores)
  }

  //Exponentially normalizes the scores for each tag
  def expNormalize(scoreMap: Map[String, Double]): Map[String, Double] = {
    val tags = scoreMap.keys.toList
    val scores = scoreMap.values.toArray

    (for( item <- tags.zip(expNormalize(scores).toList)) yield item).toList.foldLeft(Map[String, Double]())(_+_)
  }

  def expNormalize(scores: Array[Double]): Array[Double] = {
    val minimum = scores.min
    val exScores = scores.map( score => Math.exp(score - minimum) * 1.0 )
    val normalizer = exScores.sum

    exScores.map( score => score/normalizer )
  }

  private object CGWrapper {
    def fminNCG(value: (Map[(String, Int), Array[Array[Double]]], 
                        Array[Double]
                       ) => Double,
                gradient: (SparseBinaryTensor2,
                           Map[(String, Int), Array[Array[Double]]], 
                           Map[(String, String), Int],
                           Array[Double], 
                           Array[Double]
                          ) => Array[Double],
                cache: Map[(String, Int), List[List[Int]]],
                featureMap: Map[(String, String), Int],
                initialWeights: Array[Double]
               ): Array[Double] = {
      
      val globalFeatureCounts = getGlobalFeatureCounts(cache, featureMap)
      val numFeatures = globalFeatureCounts.size/cache.keys.toList.size
      val modCache = cache.map( 
                       tagEntry => tagEntry._1 -> tagEntry._2.map(
                         _.map( feature => feature % numFeatures ) 
                       ) 
                     )
      val fullSparseCache = getFullSparseCache(modCache, numFeatures)
      val model = new Parameters { val weights = Weights(new DenseTensor1(initialWeights.size)) }

      model.weights.value := initialWeights

      val optimizer = new ConjugateGradient
      val gradientMap = model.parameters.blankDenseMap

      while (!optimizer.isConverged) {
        val weights = model.weights.value.asArray
        val probMatrix = getProbMatrix(weights, modCache)
        
        gradientMap(model.weights) = new DenseTensor1(gradient(fullSparseCache, probMatrix, featureMap, weights, globalFeatureCounts))

        val currentValue = value(probMatrix, weights)

        optimizer.step(model.parameters, gradientMap, currentValue)
      }

      model.weights.value.asArray
    }
  }

  def getGlobalFeatureCounts(cache: Map[(String, Int), List[List[Int]]], model: Map[(String, String), Int]): Array[Double] = {
    val featureCounts = new Array[Double](model.keys.toList.size) 

    for{
      (tag, entries) <- cache
      entry <- entries
      feature <- entry
    } featureCounts(feature) += 1

    featureCounts
  }

  def getFullSparseCache(modCache: Map[(String, Int), List[List[Int]]], numFeatures: Int): SparseBinaryTensor2 = {
    val fullCache = modCache.values.reduceLeft(_++_)
    val docTensor = new SparseBinaryTensor2(numFeatures, fullCache.size)

    var count = 0

    fullCache.foreach{ entry =>
      entry.foreach( feature => docTensor +=(feature, count) )
      count += 1
    }

    docTensor
  }

  def getProbMatrix(weights: Array[Double], cache: Map[(String, Int), List[List[Int]]]): Map[(String, Int), Array[Array[Double]]] = {
    val numClasses = cache.keys.toList.size
    val numFeatures = weights.size/numClasses

    val weightsTable = 
      (for( i <- 0 until numClasses ) 
         yield weights.slice(0 + i * numFeatures, numFeatures + i * numFeatures)
      ).toArray

    val weightsTensor = new DenseTensor2(weightsTable)

    (for( (tag, entries) <- cache )
       yield (tag -> (for(entry <- entries)
                        yield {
                          val docTensor = new SparseBinaryTensor1(numFeatures)
                          docTensor ++= entry
                          expNormalize(weightsTensor.leftMultiply(docTensor).asArray)
                        }
                     ).toArray
             )
    ).toList.foldLeft(Map[(String, Int), Array[Array[Double]]]())(_+_)
  }
}
