/**
  * Created by litian on 11/8/16.
  */

import bayesian.BayesianClassifier

object MovieReviewClassification {
    def main(args: Array[String]): Unit = {
        val b = new BayesianClassifier()
        b.initialize("fisher_2topic/fisher_train_2topic.txt", Array(-1, 1))
        b.train()
        b.test("fisher_2topic/fisher_test_2topic.txt")
    }
}
