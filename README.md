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

__2. Why you have selected these two classification methods?__

__3. Compare selected classification methods. Which one is better? Why?__

__4. How would you compare selected classification methods if the dataset was imbalanced?__

## Results
Everything (source code, answers to questions, etc.) should be packed into single github
repository
