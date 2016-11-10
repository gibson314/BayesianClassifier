# BayesianClassifier
## Multinomial Naive Bayes
a document of length N has variables W1, ... , WN and each variable Wi takes on values from 1 to V, where V is the size of the vocabulary. You can estimate the likelihoods P(Wi = k | class) by frequency counts.
## Bernoulli Naive Bayes
In this model, every document is described by V binary variables W1, ... , WV, and Wi = 1 if the i-th word appears at least once in the document, and 0 otherwise. You can estimate the likelihoods P(Wi = 1 | class) as the proportion of documents from that class that feature the i-th word.
## Datasets:
Train and test data are both preprocessed. Words that carry little meaning, like "the" and "and", have been removed (these are called "stop words"); the remaining words are sorted, counted, and presented in the following format:

label word_1:count_1 word_2:count_2 ... word_n:count_n

where label denotes the target topic label, word_n:count_n pair denotes the form of a word, and the number of times that it occurred in the document. The word:count features are each separated by a single space.
### Sentiment analysis of movie reviews
This corpus has 4000 training documents (2000 positive and 2000 negative), and 1000 test documents (500 positive and 500 negative), where a label of -1 denotes a negative review, and 1 denotes a positive review.
### Binary conversation topic classification
 This corpus is generated from transcripts of two-person telephone conversations. Before the start of each conversation, two participants are assigned with a specific topic, for example, "Minimum Wage", "Reality TV Shows", "Pets", etc. In this binary classification task, the corpus has been filtered so that it only contains conversations on two topics: "Minimum Wage" and "Life Partners". The corpus has 878 training conversations (440 for Minimum Wage, 438 for Life Partners), and 98 test conversations (49 for Minimum Wage, 49 for Life Partners). For the target labels, -1 denotes Life Partners, and 1 denotes a topic relevant to Minimum Wage.

## Implementation:
Developped in Scala 2.11.8.

Successfully compiled in Ubuntu 16.04
