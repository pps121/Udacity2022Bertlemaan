Hyperparameters for Decision Trees
In order to create decision trees that will generalize to new problems well, we can tune a number of different aspects about the trees. We call the different aspects of a decision tree "hyperparameters". These are some of the most important hyperparameters used in decision trees:

Maximum Depth
The maximum depth of a decision tree is simply the largest possible length between the root to a leaf. A tree of maximum length kk can have at most 2^k leaves.

Minimum number of samples to split
A node must have at least min_samples_split samples in order to be large enough to split. If a node has fewer samples than min_samples_split samples, it will not be split, and the splitting process stops.

However, min_samples_split doesn't control the minimum size of leaves. As you can see in the example on the right, above, the parent node had 20 samples, greater than min_samples_split = 11, so the node was split. But when the node was split, a child node was created with that had 5 samples, less than min_samples_split = 11.

Minimum number of samples per leaf
When splitting a node, one could run into the problem of having 99 samples in one of them, and 1 on the other. This will not take us too far in our process, and would be a waste of resources and time. If we want to avoid this, we can set a minimum for the number of samples we allow on each leaf.

This number can be specified as an integer or as a float. If it's an integer, it's the minimum number of samples allowed in a leaf. If it's a float, it's the minimum percentage of samples allowed in a leaf. For example, 0.1, or 10%, implies that a particular split will not be allowed if one of the leaves that results contains less than 10% of the samples in the dataset.

If a threshold on a feature results in a leaf that has fewer samples than min_samples_leaf, the algorithm will not allow that split, but it may perform a split on the same feature at a different threshold, that does satisfy min_samples_leaf.

** Decision Trees in sklearn **
In this section, you'll use decision trees to fit a given sample dataset.
Before you do that, let's go over the tools required to build this model.
For your decision tree model, you'll be using scikit-learn's Decision Tree Classifier class. This class provides the functions to define and fit the model to your data.

>>> from sklearn.tree import DecisionTreeClassifier
>>> model = DecisionTreeClassifier()
>>> model.fit(x_values, y_values)
>>> print(model.predict([ [0.2, 0.8], [0.5, 0.4] ]))
[[ 0., 1.]]

>>> model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 10)

# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier(max_depth = 94, min_samples_leaf = 1)
model.fit(X, y)

# TODO: Fit the model.

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y,y_pred)
