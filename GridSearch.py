** Grid Search in sklearn **
Grid Search in sklearn is very simple. We'll illustrate it with an example. Let's say we'd like to train a support vector machine, and we'd like to decide between the following parameters:

kernel: poly or rbf.
C: 0.1, 1, or 10.
  
 
** Code:
  
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
scorer = make_scorer(f1_score)

# Create the object.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
# Fit the data
grid_fit = grid_obj.fit(X, y)

best_clf = grid_fit.best_estimator_


** Use grid search to improve this model **
In here, we'll do the following steps:

First define some parameters to perform grid search on. We suggest to play with max_depth, min_samples_leaf, and min_samples_split.
Make a scorer for the model using f1_score.
Perform grid search on the classifier, using the parameters and the scorer.
Fit the data to the new classifier.
Plot the model and find the f1_score.
If the model is not much better, try changing the ranges for the parameters and fit it again.
Optional Step - Put the steps 2-6 mentioned above inside a function calculate_F1_Score(parameters) to make it reusable.


** An Intriguing Result!! **
In the cell above, you have used the following hyper-parameter grid with a step-size 2 to perform the grid search:

parameters = {'max_depth':[2,4,6,8,10],'min_samples_leaf':[2,4,6,8,10], 'min_samples_split':[2,4,6,8,10]}
Let's run the model again with the following finer parameters grid (super-set) with a step-size 1 before coming to a Conclusion.
parameters = {'max_depth':[1,2,3,4,5,6,7,8,9,10],'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10],'min_samples_split':[2,3,4,5,6,7,8,9,10]}
