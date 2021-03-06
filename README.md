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
- Import positive and negative reviews (merge to one dataset, positive reviews predictable var (Y) value 1, negative - 0 (binary classification task) and etc.)
- Remove punctuation, stop words and other noninformative elements from text
- TFIDFVectorizer (work same as CountVectorizer + TFIDFTransformer in one function)
- Review classification with RandomForestClassifier and KNeighborsClassifier

If accuracy were important then additional PCA analysis could be performed, as well as more popular SVM, XGBoost or NB classification methods. Likewise, accurate results are obtained using hybrid models. Various other methods of vectorization or "noise filtering" and its methods like HARF are also available for better accuracy.

__2. Why you have selected these two classification methods?__

In fact, these methods were chosen randomly. The SVM, NB methods are commonly used in text classification [1], so they were not used in this exercise and RB and KNN methods were tested for accuracy. Talking about RB method, the ensemble segment algorithms are suitable when the available data contains noise or we want to avoid overfitting. In real world task I would prefer to test more than 2 methods (DT, DNN and etc.) to be sure that the method chosen is the most appropriate for the task but in this task in source code (Jupyter and .py) I show only 2 methods.

__3. Compare selected classification methods. Which one is better? Why?__

As we can see information in table below accuracy of both models pretty similar KNN model have better precision that means that this model have high true positive over (true + false) positives but have low recall rate (a lot false negatives). If we are looking for bad reviews this model is not suitable for this task. Overall we can say that RF model is better in this example.

| Method | Accuracy | F1 Score | Precision | Recall |
| --- | --- | --- | --- | --- |
| RF | 0.688701 | 0.664646 | 0.686848 | 0.643836 |
| KNN | 0.662916 | 0.487527 | 0.897638 | 0.334638 |

__4. How would you compare selected classification methods if the dataset was imbalanced?__

Depending on the imbalance ratio because low disbalance can be solved with methods like RandomForest for example. Also in data processing, there are oversampling, undersampling algorithms to solve the imbalanced data problem. In ML methods possible solution for the imbalance is assigning weights to classes. If you are just comparing models that were created on imbalanced data you can use different metrics like in table above Precision, Recall because sure that if we have 10000 obs of one class and 100 obs another accuracy will be high even if the model will show that all obs is from first class.

## Python code usage
If you are running `classification.py` using command promt (terminal) there are 2 options: 
- `--train` with 2 arguments: 1) positive review .csv file 2) negative review .csv file. This option will prepare data and do 3-fold CV on preprocessed data, find best model, create model with best parameters and save model in same directory as `classification.py`
- `--text` with your text will use saved model and test it with new text and give positive, negative reviews probability.

## References / Further readings
1. Kumar, A., & Jaiswal, A. (2019). Systematic literature review of sentiment analysis on Twitter using soft computing techniques. Concurrency and Computation: Practice and Experience, e5107.
