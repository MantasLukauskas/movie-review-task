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

In fact, these 2 methods were chosen to be said by chance. The SVM, NB methods are commonly used in Sentiment Analysis [1], so they were not used in this exercise and RB and KNN methods were tested for accuracy. talking about RB method, the ensemble segment algorithms are suitable when the available data contains noise or when the model is to be avoided learning

__3. Compare selected classification methods. Which one is better? Why?__

As we can see information in table below accuracy of both models pretty similar KNN model have better precision that means that this model have high true positive over (true + false) positives but have low recall rate (a lot false negatives). If we are looking for bad reviews this model is not suitable for this task. Overall we can say that RF model is better in this example.

| Method | Accuracy | F1 Score | Precision | Recall |
| --- | --- | --- | --- | --- |
| RF | 0.688701 | 0.664646 | 0.686848 | 0.643836 |
| KNN | 0.662916 | 0.487527 | 0.897638 | 0.334638 |

__4. How would you compare selected classification methods if the dataset was imbalanced?__


## Python code usage
If you are running `classification.py` using command promt (terminal) there are 2 options: 
- `--train` with 2 arguments: 1) positive review .csv file 2) negative review .csv file. This option will processes data and do 3-fold CV on preprocessed data, find best model, create model with best parameters and save model in same directory as `classification.py`
- `--text` with your text will use saved model and test it with new text and give positive, negative reviews probability.

Use `git status` to list all new or modified files that haven't yet been committed.

## References / Further readings
1. Kumar, A., & Jaiswal, A. (2019). Systematic literature review of sentiment analysis on Twitter using soft computing techniques. Concurrency and Computation: Practice and Experience, e5107.
