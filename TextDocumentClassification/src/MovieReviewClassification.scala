/**
  * Created by litian on 11/8/16.
  */

import bayesian.BayesianClassifier

object MovieReviewClassification {
    val trainfiles = Array("fisher_2topic/fisher_train_2topic.txt", "movie_review/rt-train.txt", "fisher_40topic/fisher_train_40topic.txt")
    val testfiles = Array("fisher_2topic/fisher_test_2topic.txt", "movie_review/rt-test.txt", "fisher_40topic/fisher_test_40topic.txt")
    val modeltype = Array("multinomial", "bernoulli")
    def main(args: Array[String]): Unit = {
        val b = new BayesianClassifier()
        b.initialize(trainfiles(2), (0 until 40).toArray)
        b.train(modeltype(1))
        b.test(testfiles(2), modeltype(1))
    }
}
