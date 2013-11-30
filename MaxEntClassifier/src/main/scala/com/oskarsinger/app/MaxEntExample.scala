import MaxEntClassification.MaxEntChineseWordSegmenter

object MaxEntExample extends App {

  val maxEnt = new MaxEntChineseWordSegmenter()

  maxEnt.getLabeledDataSet("/home/oskar/GitRepos/SegmentationData/training/pku_training.utf8").foreach(println)

}
