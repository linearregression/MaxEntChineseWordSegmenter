package com.oskarsinger.app

import scala.io.Source

object MaxEntExample extends App {

  val segmenter = new MaxEntChineseWordSegmenter()
  val trainingCorpusPath = "/home/oskar/GitRepos/SegmentationData/training/pku_training.utf8"
  val model = segmenter.train(trainingCorpusPath)

  assert( model.size > 0 )

  val testingCorpusPath = "/home/oskar/GitRepos/SegmentationData/testing/pku_test.utf8"  
  val testingFile = new java.io.File(testingCorpusPath)
  val testingTags = segmenter.segment(testingFile, model).map( taggedCharacter => taggedCharacter._2 )

  assert( testingTags.size > 0 )

  val goldCorpusPath = "/home/oskar/GitRepos/SegmentationData/gold/pku_test_gold.utf8"
  val goldTags = segmenter.getLabeledDataSet(goldCorpusPath).map( character => character._2 )

  assert( testingTags.size == goldTags.size )

  val correspondingTags = testingTags.zip(goldTags)

  val numMatches = correspondingTags.count( tags => tags._1 equals tags._2 ) * 1.0

  println(numMatches / correspondingTags.size) 
}
