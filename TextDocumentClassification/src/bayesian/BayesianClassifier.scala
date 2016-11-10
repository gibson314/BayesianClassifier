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

    // (class, word_index) -> count
    private val _class_word_count = mutable.HashMap[(Int, Int), Int]()

    private val _prior = mutable.HashMap[Int, Double]()


    private val _feature_likelihood = mutable.HashMap[(Int, Int), Double]()

    private var _vocabulary_index = 0
    private var _train_case_count = 0

    private val _posterior = mutable.HashMap[Int, Double]()


    // Bernoulli Naive Bayesian helper hashmap
    private val _bernoulli_class_word_count = mutable.HashMap[(Int, Int), Int]()
    private val _bernoulli_feature_likelihood = mutable.HashMap[(Int, Int), (Double, Double)]()

    def initialize(test_file: String, class_name: Array[Int]): Unit ={
    /*
        @Parameter: test_file - filename of test cases

    */
        _class_name = class_name
        println("Read Training File...")
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
                        // add to vocabulary
                        _vocabulary.put(word, _vocabulary_index)
                        // initialize each entry of _class_wod_count (or Bernoulli)
                        class_name.foreach(cl => _class_word_count.put((cl, _vocabulary_index), 0))
                        class_name.foreach(cl => _bernoulli_class_word_count.put((cl, _vocabulary_index), 0))
                        _vocabulary_index += 1
                    }
                    word_sum += count.toInt

                    // Add count to (class, word_index) -> count
                    val word_index = _vocabulary(word)
                    _class_word_count.update((_class, word_index),
                        count.toInt + _class_word_count.getOrElse((_class, word_index), 0))

                    // Add count to bernoulli_word_count
                    _bernoulli_class_word_count.update((_class, word_index),
                        1 + _bernoulli_class_word_count.getOrElse((_class, word_index), 0))

                }

                _class_word_sum.update(_class, _class_word_sum.getOrElse(_class, 0) + word_sum)
            }
        )
        println("Initialized dictionary from " + _train_case_count + " train cases." )
        println(_vocabulary.size + " Words Vocabulary Set.")
        //_vocabulary.foreach(println)
        //println(_class_word_count)
    }



    def train(train_type: String): Unit ={
        train_type match {
            case "multinomial" =>
                println("Training using Multinomial Naive Bayes...")
                _class_name.foreach(cl => _prior.put(cl, _class_count(cl).toDouble / _train_case_count))
                val K = 1
                _class_word_count.keys.foreach(key =>
                    _feature_likelihood.put(key,
                        (_class_word_count(key).toDouble + K) / (_class_word_sum(key._1) + K * _vocabulary.size))
                )
            case "bernoulli" =>
                println("Training using Bernoulli Naive Bayes...")
                _class_name.foreach(cl => _prior.put(cl, _class_count(cl).toDouble / _train_case_count))
                val K = 1
                _bernoulli_class_word_count.keys.foreach(key =>
                    _bernoulli_feature_likelihood.put(key,
                        (log((_bernoulli_class_word_count(key).toDouble + K) / (_class_word_sum(key._1) + K * _vocabulary.size)),
                                log(1 - (_bernoulli_class_word_count(key).toDouble + K) / (_class_word_sum(key._1) + K * _vocabulary.size))))
                )

        }
    }



        //println(_feature_likelihood.maxBy(_._2))
        //println(_vocabulary.find(_._2 == 50))


    def test_single(document: Array[(String, Int)]): Int = {
        _posterior.clear()
        _class_name.foreach(c => {
            var likelihood = 0.0
            document.foreach(doc => {
                if (_vocabulary.contains(doc._1))
                    likelihood += doc._2 * _feature_likelihood((c, _vocabulary(doc._1)))
            })

            likelihood += log(_prior(c))
            _posterior.put(c, likelihood)
        })
        _posterior.maxBy(_._2)._1
    }


    def test_single_bernoulli(document: Set[String]) : Int = {
        _posterior.clear()
        // accelerate by parallel processing
        _class_name.par.foreach(c => {
            var likelihood = 0.0
//            val t1 = System.currentTimeMillis
            _vocabulary.keys.foreach(v => {
                val v_index = _vocabulary(v)

                if (document.contains(v)){
                    likelihood += _bernoulli_feature_likelihood((c, v_index))._1
                }
                else {
                    likelihood += _bernoulli_feature_likelihood((c, v_index))._2
                }
            })
//            val t2 = System.currentTimeMillis
//            println("XXX "+ (t2 - t1))
            likelihood += log(_prior(c))
            _posterior.put(c, likelihood)
        })
        _posterior.maxBy(_._2)._1
    }

    def test(test_file: String, test_type: String): Unit ={
        var test_case_count = 0
        var correct_case_count = 0


        var testcase = new ArrayBuffer[(Int, Set[String], Array[(String, Int)])]()
        Source.fromFile(test_file).getLines().foreach(
            line => {

                var document = new ArrayBuffer[(String, Int)]()
                var doc_word = new ArrayBuffer[String]()
                val testline = line.split(" ")
                val test_label = testline(0).toInt
                for (i <- 1 until testline.length) {
                    val Array(word, count) = testline(i).split(":")
                    document += ((word, count.toInt))
                    doc_word += word
                }
                testcase += ((test_label, doc_word.toSet, document.toArray))
            })
        println(testcase.size + " Test case initialized.")


        val confusionmatrix = Array.ofDim[Int](_class_name.length, _class_name.length)


        testcase.foreach(tc => {
                test_type match {
                    case "multinomial" =>
                        print("Test Case " + test_case_count + ": ")
                        val t1 = System.currentTimeMillis
                        val predict_label = test_single(tc._3)
                        if (predict_label == tc._1) {
                            correct_case_count += 1
                            print("\tResult: Correct\t")
                        }
                        else
                            print("\tResult: Wrong\t")
                        val t2 = System.currentTimeMillis
                        println("[" + (t2 - t1) + " ms]")

                        confusionmatrix(_class_name.indexOf(tc._1))(_class_name.indexOf(predict_label)) += 1

                    case "bernoulli" =>
                        print("Test Case " + test_case_count + ": ")
                        val t1 = System.currentTimeMillis
                        val predict_label = test_single_bernoulli(tc._2)
                        if (predict_label == tc._1) {
                            correct_case_count += 1
                            print("\tResult: Correct\t")
                        }
                        else
                            print("\tResult: Wrong\t")
                        val t2 = System.currentTimeMillis
                        println("[" + (t2 - t1) + " ms]")
                        confusionmatrix(_class_name.indexOf(tc._1))(_class_name.indexOf(predict_label)) += 1

                }
                test_case_count += 1

            }
        )
        println("Confusion Matrix: ")
        confusionmatrix.foreach(r => {r.foreach(c => print(c + "\t"))
            println})
        println("Accuracy: " + correct_case_count.toDouble / test_case_count)
    }



}
