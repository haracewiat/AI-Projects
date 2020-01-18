
import pandas as pd
import numpy as np
import graphviz
from sklearn import tree
from sklearn.model_selection import cross_val_score

# Read the file
d = pd.read_csv('./dataset/student-por.csv', sep=';')


# Add columns for pass/fail
d['pass'] = d.apply(lambda row: 1 if (
    row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)
d = d.drop(['G1', 'G2', 'G3'], axis=1)
d.head()

""" 
    In computer science, the most important, defining characteristic of a lambda
    expression is that it is used as data. That is, the function is passed as
    an argument to another function, returned as a value from a function, or
    assigned to variables or data structures.
"""

# Convert columns such as 'romantic', 'internet' etc. into numbers
d = pd.get_dummies(d, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])

"""
    One-hot encoding is a process by which categorical variables are converted 
    into a form that could be provided to ML algorithms to do a better job in 
    prediction.
"""


# Produce a training set and a test set
d = d.sample(frac=1)

d_train = d[:500]
d_test = d[500:]

d_train_att = d_train.drop(['pass'], axis=1)
d_train_pass = d_train['pass']

d_test_att = d_test.drop(['pass'], axis=1)
d_test_pass = d_test['pass']

d_att = d.drop(['pass'], axis=1)
d_pass = d['pass']

"""
    Because the pass/fail statistic is around 50%, the set is a well-balanced one.
"""

# Fit a decision tree
t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
t = t.fit(d_train_att, d_train_pass)

"""
    Entropy or information gain is used to decide when to split. The tree will split
    at a depth of 5 questions.
"""


# Visualize the tree
dot_data = tree.export_graphviz(t, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(d_train_att), class_names=["fail", "pass"],
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)


# Check the score of the tree
t.score(d_test_att, d_test_pass)


# Cross-verify the result (to be sure the dataset is trained well)
scores = cross_val_score(t, d_att, d_pass, cv=5)

"""
    Performing cross-validation on the entire dataset splits the data on a 20/80 basis, 
    where 20% is the testing set and 80% is the training set.

    To see how the number of depths affects the above value, the following code can be 
    run:

    for max_depth in range(1, 20):
        t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
        scores = cross_val_score(t, d_att, d_pass, cv=5)
        print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2))

    Choosing the right depth is important as it helps to avoid overfitting and underfitting.

    Overfitting is a modeling error which occurs when a function is too closely fit to a limited set of 
    data points. Overfitting the model generally takes the form of making an overly complex model to 
    explain idiosyncrasies in the data under study.

    Underfitting occurs when the model or the algorithm does not fit the data well enough. Specifically, 
    underfitting occurs if the model or algorithm shows low variance but high bias.
"""
