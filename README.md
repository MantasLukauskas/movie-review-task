# Movie Review task

## Task
Classify movie reviews into positive and negative. Classification task should be done
using two different classification methods (e.g. logistic regression and Naive Bayes)
## Requirements
Use Python programming language
High accuracy is not required for this task
## Dataset
http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
## Questions
__1. Describe text processing pipeline you have selected.__
- Import positive and negative reviews
- Remove noninformative (punctuation, stop words) from text
- TFIDFVectorizer (same as CountVectorizer + TFIDFTransformer)
- Review classification with RandomForestClassifier and KNeighborsClassifier

If accuracy were important then additional PCA analysis could be performed, as well as more popular SVM, XGBoost or NB methods. Likewise, accurate results are obtained using hybrid models. Various other methods of vectorization or "noise filtering" and its methods like HARF are also available.

__2. Why you have selected these two classification methods?__

__3. Compare selected classification methods. Which one is better? Why?__

| Method | Accuracy | F1 Score | Precision | Recall |
| --- | --- | --- | --- | --- |
| RF | 0.688701 | 0.664646 | 0.686848 | 0.643836 |
| KNN | 0.662916 | 0.487527 | 0.897638 | 0.334638 |

__4. How would you compare selected classification methods if the dataset was imbalanced?__

