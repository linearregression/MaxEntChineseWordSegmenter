import MaxEntClassification.MaxEntChineseWordSegmenter

object MaxEntExample extends App {

  val segmenter = new MaxEntChineseWordSegmenter()
  val trainingCorpusPath = "/home/oskar/GitRepos/SegmentationData/training/pku_training.utf8"
  val model = segmenter.train(trainingCorpusPath)
  println("Model size: " + model.size)
}
