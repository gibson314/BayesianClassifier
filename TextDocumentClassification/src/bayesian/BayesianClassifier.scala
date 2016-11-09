package bayesian


import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.math.log
/**
  * Created by litian on 11/8/16.
  */


class BayesianClassifier {
    private var _class_name = Array[Int]()
    private val _vocabulary = mutable.HashMap[String, Int]()
    private val _class_count = mutable.HashMap[Int, Int]()
    private val _class_word_sum = mutable.HashMap[Int, Int]()
    private val _class_word_count = mutable.HashMap[(Int, Int), Int]()

    private val _prior = mutable.HashMap[Int, Double]()
    private val _feature_likelihood = mutable.HashMap[(Int, Int), Double]()

    private var _vocabulary_index = 0
    private var _train_case_count = 0

    private val _posterior = mutable.HashMap[Int, Double]()

    def initialize(test_file: String, class_name: Array[Int]): Unit ={
    /*
        @Parameter: test_file - filename of test cases

    */
        _class_name = class_name
        println("Read Test File...")
        Source.fromFile(test_file).getLines().foreach(
            line => {
                val inputline = line.split(" ")
                val _class = inputline(0).toInt
                _class_count.update(_class, _class_count.getOrElse(_class, 0) + 1)
                _train_case_count += 1

                var word_sum = 0
                for (i <- 1 until inputline.length){
                    val Array(word, count) = inputline(i).split(":")

                    // Add new word and its corresponding index to vocabulary
                    if (!_vocabulary.contains(word)) {
                        _vocabulary.put(word, _vocabulary_index)
                        class_name.foreach(key => _class_word_count.put((key, _vocabulary_index), 0))
                        _vocabulary_index += 1
                    }
                    word_sum += count.toInt
                    val word_index = _vocabulary(word)
                    _class_word_count.update((_class, word_index),
                        count.toInt + _class_word_count.getOrElse((_class, word_index), 0))
                }

                _class_word_sum.update(_class, _class_word_sum.getOrElse(_class, 0) + word_sum)
            }
        )
        println("Initialized dictionary from " + _train_case_count + " test cases." )
        //_vocabulary.foreach(println)
        //println(_class_word_count)
    }



    def train(): Unit ={
        println("Training...")
        _class_name.foreach(cl => _prior.put(cl, _class_count(cl).toDouble / _train_case_count))
        //println(_prior)
        //println(_class_word_sum)
        //println(_class_word_count)
        val K = 1
        _class_word_count.keys.foreach(key =>
            _feature_likelihood.put(key,
                (_class_word_count(key).toDouble + K) / (_class_word_sum(key._1) + K * _vocabulary.size))
        )


        //println(_feature_likelihood.maxBy(_._2))
        //println(_vocabulary.find(_._2 == 50))
    }

    def test_single(test_label: Int, document: Array[(String, Int)]): Boolean = {
        _posterior.clear()
        _class_name.foreach(c => {
            var likelihood = 0.0
            document.foreach(doc => {
                if (_vocabulary.contains(doc._1))
                    likelihood += doc._2 * log(_feature_likelihood((c, _vocabulary(doc._1))))
            })

            likelihood += log(_prior(c))
            _posterior.put(c, likelihood)
        })
        //print(_posterior + "-" + _posterior.maxBy(_._2)._1)
        //println()
        _posterior.maxBy(_._2)._1 == test_label
    }

    def test(test_file: String): Unit ={
        var test_case_count = 0
        var correct_case_count = 0

        Source.fromFile(test_file).getLines().foreach(
            line => {
                print("Test Case " + test_case_count + ": ")
                var document = new ArrayBuffer[(String, Int)]()
                val testline = line.split(" ")
                val test_label = testline(0).toInt
                for (i <- 1 until testline.length) {
                    val Array(word, count) = testline(i).split(":")
                    document += ((word, count.toInt))
                }

                if (test_single(test_label, document.toArray)) {
                    correct_case_count += 1
                    print("\tResult: Correct\n")
                }
                else
                    print("\tResult: Wrong\n")
                test_case_count += 1

            }
        )

        println("Accuracy: " + correct_case_count.toDouble / test_case_count)
    }



}
