** AdaBoost in sklearn **
Building an AdaBoost model in sklearn is no different than building any other model. You can use scikit-learn's AdaBoostClassifier class. This class provides the functions to define and fit the model to your data.

>>> from sklearn.ensemble import AdaBoostClassifier
>>> model = AdaBoostClassifier()
>>> model.fit(x_train, y_train)
>>> model.predict(x_test)

In the example above, the model variable is a decision tree model that has been fitted to the data x_train and y_train. The functions fit and predict work exactly as before.

Hyperparameters
When we define the model, we can specify the hyperparameters. In practice, the most common ones are

base_estimator: The model utilized for the weak learners (Warning: Don't forget to import the model that you decide to use for the weak learner).
n_estimators: The maximum number of weak learners used.
For example, here we define a model which uses decision trees of max_depth 2 as the weak learners, and it allows a maximum of 4 of them.

>>> from sklearn.tree import DecisionTreeClassifier
>>> model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)
                                                          

Recap
In this lesson, you learned about a number of techniques used in ensemble methods. Before looking at the techniques, you saw that there are two variables with tradeoffs Bias and Variance.

High Bias, Low Variance models tend to underfit data, as they are not flexible. Linear models fall into this category of models.

High Variance, Low Bias models tend to overfit data, as they are too flexible. Decision trees fall into this category of models.

Ensemble Models
In order to find a way to optimize for both variance and bias, we have ensemble methods. Ensemble methods have become some of the most popular methods used to compete in competitions on Kaggle and used in industry across applications.

There were two randomization techniques you saw to combat overfitting:

Bootstrap the data - that is, sampling the data with replacement and fitting your algorithm and fitting your algorithm to the sampled data.

Subset the features - in each split of a decision tree or with each algorithm used an ensemble only a subset of the total possible features are used.

Techniques
You saw a number of ensemble methods in this lesson including:

BaggingClassifier
RandomForestClassifier
AdaBoostClassifier
Another really useful guide for ensemble methods can be found in the documentation here. These methods can also all be extended to regression problems, not just classification.

Additional Resources
Additionally, here are some great resources on AdaBoost if you'd like to learn some more!

Here is the original paper from Freund and Schapire.
A follow-up paper from the same authors regarding several experiments with Adaboost.
A great tutorial by Schapire.

                                                          
