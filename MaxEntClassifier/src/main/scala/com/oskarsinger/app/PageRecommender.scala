package MaxEntClassification

import scala.collection.mutable.Map
import scala.io.Source
import java.io.File

class PageClassification {

  def classify(datum: (String, String, Map[String, Boolean]), 
               model: Map[(String, String), Double] = Map[(String, String), Double]().withDefaultValue(0)
              ): (String, Map[(String, String), Double]) = {
    val refDom = datum._1
    val hasVisitedSharePage = datum._2
    val copyChoices = datum._3
    
    //TODO: Implement Thompson sampling 
    val scores = (for(c <- copyChoices.keys)
                   yield (c -> (model(c -> refDom) + model(c -> hasVisitedSharePage)))
                 ).toList.foldLeft(Map[String, Double]())(_+_)

    val suggestion = expNormalize(scores).toList.sortWith( (x,y) => x._2 > y._2 )(0)._1

    val delta = if(copyChoices(suggestion)) 1 else 0 

    model(suggestion -> refDom) += 1
    model(suggestion -> hasVisitedSharePage) += 1

    (suggestion -> optimize(model, gradient))
  }

  def expNormalize(scores: Map[String, Double]): Map[String, Double] = {
    val minimum = scores.values.min
    val expScores = scores.map( score => score._1 -> Math.exp(score._2 - minimum) )
    val normalizer = expScores.values.sum
    val normScores = expScores.map( score => score._1 -> score._2/normalizer )

    assert(normScores.values.sum == 1)

    normScores
  }

  def gradient(): Array[Double] = {
    new Array[Double](55) 
  }

  def optimize(model: Map[(String, String), Double], gradient: () => Array[Double]): Map[(String, String), Double] = {
    model 
  }
} 
