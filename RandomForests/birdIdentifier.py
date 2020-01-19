import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.model_selection import cross_val_score

# Read the file
imgatt = pd.read_csv("dataset/image_attribute_labels.txt",
                     sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     usecols=[0, 1, 2], names=['imgid', 'attid', 'present'])

"""
    Description from dataset README:

    The set of attribute labels as perceived by MTurkers for each image
    is contained in the file attributes/image_attribute_labels.txt, with
    each line corresponding to one image/attribute/worker triplet:

    <image_id> <attribute_id> <is_present> <certainty_id> <time>

    where <image_id>, <attribute_id>, <certainty_id> correspond to the IDs
    in images.txt, attributes/attributes.txt, and attributes/certainties.txt
    respectively.  <is_present> is 0 or 1 (1 denotes that the attribute is
    present).  <time> denotes the time spent by the MTurker in seconds.
"""

"""
    We can see how many rows and columns we have by running the following code:

    imgatt.shape

    This will output (3677856, 3), which means we have about 3.7 million rows and 3 columns.
"""


# Reorganize the table
imgatt2 = imgatt.pivot(index='imgid', columns='attid', values='present')

"""
   Because we want the attributes to be columns, not rows, we need to change the table. For this,
   we have to use pivot:

   1. Pivot on the image ID and make one row for each image ID.
   2. Turn the attributes into distinct columnt, and the vaues will be ones or twos. 

   Now, imgatt2.shape will output (11788, 312). 
"""


# Load the true image classes
imglabels = pd.read_csv("dataset/image_class_labels.txt",
                        sep=' ', header=None, names=['imgid', 'label'])

imglabels = imglabels.set_index('imgid')

"""
    Description from dataset README:

    The ground truth class labels (bird species labels) for each image are contained
    in the file image_class_labels.txt, with each line corresponding to one image:

    <image_id> <class_id>

    where <image_id> and <class_id> correspond to the IDs in images.txt and classes.txt,
    respectively.
"""


# Attach the labels to the attribute data set
df = imgatt2.join(imglabels)


# Shuffle the data
df = df.sample(frac=1)

df_att = df.iloc[:, :312]
df_label = df.iloc[:, 312:]

# Seperate two sets: a training set and a testing set
df_train_att = df_att[:8000]
df_train_label = df_label[:8000]
df_test_att = df_att[8000:]
df_test_label = df_label[8000:]

df_train_label = df_train_label['label']
df_test_label = df_test_label['label']


# Create a Random Forest Classifier
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100)

"""
    max_features: the number of different columnes each tree can look at
    n_estimators: the number of trees created 
"""


# Build the classifier
clf.fit(df_train_att, df_train_label)


# Predict a few cases and check the accuracy
"""
    The following code predicts 5 species based on the attributes from the first 5 rows
    of the training set:

    clf.predict(df_train_att.head())

    The output is: [ 73  32  86 169  46 ]

    
    Now, to check the accuracy of this prediction, the following code is run:
    
    clf.score(df_test_att, df_test_label)

    This outputs an accuracy level of 0.447. This is not the best score.

"""


# Create a confusion matrix
pred_labels = clf.predict(df_test_att)
cm = confusion_matrix(df_test_label, pred_labels)


# Visualize the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Load the names of the birds
birds = pd.read_csv("dataset/classes.txt",
                    sep='\s+', header=None, usecols=[1], names=['birdname'])
birds = birds['birdname']


# Plot the confusion matrix
"""
    np.set_printoptions(precision=2)
    plt.figure(figsize=(60, 60), dpi=300)
    plot_confusion_matrix(cm, classes=birds, normalize=True)
    plt.show()
"""


# Test accuracy of other methods

"""
    Decision Tree:

    from sklearn import tree
    clftree = tree.DecisionTreeClassifier()
    clftree.fit(df_train_att, df_train_label)
    clftree.score(df_test_att, df_test_label)

    The accuracy of a decision tree is 0.27 (0.2695353748680042)

"""


"""
    Support Vector Machine (neural network approach)

    from sklearn import svm
    clfsvm = svm.SVC()
    clfsvm.fit(df_train_att, df_train_label)
    clfsvm.score(df_test_att, df_test_label)

    The accuracy of SVM is 0.49 (0.48653643083421333)
    * The authors of the book achived an accuracy of 0.29
"""

# Perform cross-validation to make sure the accuracy is reliable

"""
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scorestree = cross_val_score(clftree, df_train_att, df_train_label, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" %  (scorestree.mean(), scorestree.std() * 2))

    scoressvm = cross_val_score(clfsvm, df_train_att, df_train_label, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))

    This gives the following results:
    Accuracy: 0.44 (+/- 0.01)  (random forest)
    Accuracy: 0.26 (+/- 0.01)  (simple decision tree)
    Accuracy: 0.47 (+/- 0.02)  (svm)

    * The authors of the book received the following results: 0.44, 0.25 and 0.27
"""


# Perform a test to find the best number of attributes and estimators
"""
    In order to test the impact of different amount of attributes and estimators, 
    the following code is run:

    max_features_opts = range(5, 50, 5)
    n_estimators_opts = range(10, 200, 20)
    rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts), 4), float)
    i = 0
    for max_features in max_features_opts:
        for n_estimators in n_estimators_opts:
            clf = RandomForestClassifier(
                max_features=max_features, n_estimators=n_estimators)
            scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
            rf_params[i, 0] = max_features
            rf_params[i, 1] = n_estimators
            rf_params[i, 2] = scores.mean()
            rf_params[i, 3] = scores.std() * 2
            i += 1
            print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" %
                (max_features, n_estimators, scores.mean(), scores.std() * 2))


    This results in the output:
    {
        Max features: 5, num estimators: 10, accuracy: 0.26 (+/- 0.02)
        Max features: 5, num estimators: 30, accuracy: 0.35 (+/- 0.01)
        Max features: 5, num estimators: 50, accuracy: 0.38 (+/- 0.01)
        Max features: 5, num estimators: 70, accuracy: 0.41 (+/- 0.02)
        Max features: 5, num estimators: 90, accuracy: 0.42 (+/- 0.01)
        Max features: 5, num estimators: 110, accuracy: 0.42 (+/- 0.01)
        Max features: 5, num estimators: 130, accuracy: 0.43 (+/- 0.01)
        Max features: 5, num estimators: 150, accuracy: 0.44 (+/- 0.00)
        Max features: 5, num estimators: 170, accuracy: 0.44 (+/- 0.02)
        Max features: 5, num estimators: 190, accuracy: 0.44 (+/- 0.02)
        Max features: 10, num estimators: 10, accuracy: 0.29 (+/- 0.02)
        Max features: 10, num estimators: 30, accuracy: 0.38 (+/- 0.02)
        Max features: 10, num estimators: 50, accuracy: 0.41 (+/- 0.01)
        Max features: 10, num estimators: 70, accuracy: 0.42 (+/- 0.01)
        Max features: 10, num estimators: 90, accuracy: 0.43 (+/- 0.01)
        Max features: 10, num estimators: 110, accuracy: 0.44 (+/- 0.02)
        Max features: 10, num estimators: 130, accuracy: 0.45 (+/- 0.01)
        Max features: 10, num estimators: 150, accuracy: 0.45 (+/- 0.01)
        Max features: 10, num estimators: 170, accuracy: 0.45 (+/- 0.01)
        Max features: 10, num estimators: 190, accuracy: 0.45 (+/- 0.01)
        Max features: 15, num estimators: 10, accuracy: 0.31 (+/- 0.03)
        Max features: 15, num estimators: 30, accuracy: 0.38 (+/- 0.02)
        Max features: 15, num estimators: 50, accuracy: 0.42 (+/- 0.01)
        Max features: 15, num estimators: 70, accuracy: 0.43 (+/- 0.02)
        Max features: 15, num estimators: 90, accuracy: 0.44 (+/- 0.01)
        Max features: 15, num estimators: 110, accuracy: 0.45 (+/- 0.01)
        Max features: 15, num estimators: 130, accuracy: 0.44 (+/- 0.02)
        Max features: 15, num estimators: 150, accuracy: 0.45 (+/- 0.02)
        Max features: 15, num estimators: 170, accuracy: 0.45 (+/- 0.00)
        Max features: 15, num estimators: 190, accuracy: 0.45 (+/- 0.01)
        Max features: 20, num estimators: 10, accuracy: 0.32 (+/- 0.02)
        Max features: 20, num estimators: 30, accuracy: 0.39 (+/- 0.01)
        Max features: 20, num estimators: 50, accuracy: 0.42 (+/- 0.02)
        Max features: 20, num estimators: 70, accuracy: 0.43 (+/- 0.01)
        Max features: 20, num estimators: 90, accuracy: 0.44 (+/- 0.01)
        Max features: 20, num estimators: 110, accuracy: 0.44 (+/- 0.02)
        Max features: 20, num estimators: 130, accuracy: 0.45 (+/- 0.01)
        Max features: 20, num estimators: 150, accuracy: 0.45 (+/- 0.01)
        Max features: 20, num estimators: 170, accuracy: 0.46 (+/- 0.02)
        Max features: 20, num estimators: 190, accuracy: 0.46 (+/- 0.01)
        Max features: 25, num estimators: 10, accuracy: 0.32 (+/- 0.01)
        Max features: 25, num estimators: 30, accuracy: 0.40 (+/- 0.01)
        Max features: 25, num estimators: 50, accuracy: 0.42 (+/- 0.01)
        Max features: 25, num estimators: 70, accuracy: 0.44 (+/- 0.01)
        Max features: 25, num estimators: 90, accuracy: 0.44 (+/- 0.01)
        Max features: 25, num estimators: 110, accuracy: 0.45 (+/- 0.01)
        Max features: 25, num estimators: 130, accuracy: 0.45 (+/- 0.02)
        Max features: 25, num estimators: 150, accuracy: 0.45 (+/- 0.01)
        Max features: 25, num estimators: 170, accuracy: 0.45 (+/- 0.01)
        Max features: 25, num estimators: 190, accuracy: 0.46 (+/- 0.01)
        Max features: 30, num estimators: 10, accuracy: 0.32 (+/- 0.01)
        Max features: 30, num estimators: 30, accuracy: 0.40 (+/- 0.02)
        Max features: 30, num estimators: 50, accuracy: 0.42 (+/- 0.02)
        Max features: 30, num estimators: 70, accuracy: 0.43 (+/- 0.01)
        Max features: 30, num estimators: 90, accuracy: 0.44 (+/- 0.02)
        Max features: 30, num estimators: 110, accuracy: 0.45 (+/- 0.02)
        Max features: 30, num estimators: 130, accuracy: 0.45 (+/- 0.02)
        Max features: 30, num estimators: 150, accuracy: 0.45 (+/- 0.01)
        Max features: 30, num estimators: 170, accuracy: 0.45 (+/- 0.01)
        Max features: 30, num estimators: 190, accuracy: 0.45 (+/- 0.01)
        Max features: 35, num estimators: 10, accuracy: 0.33 (+/- 0.01)
        Max features: 35, num estimators: 30, accuracy: 0.40 (+/- 0.01)
        Max features: 35, num estimators: 50, accuracy: 0.42 (+/- 0.01)
        Max features: 35, num estimators: 70, accuracy: 0.44 (+/- 0.01)
        Max features: 35, num estimators: 90, accuracy: 0.44 (+/- 0.02)
        Max features: 35, num estimators: 110, accuracy: 0.44 (+/- 0.01)
        Max features: 35, num estimators: 130, accuracy: 0.45 (+/- 0.01)
        Max features: 35, num estimators: 150, accuracy: 0.45 (+/- 0.01)
        Max features: 35, num estimators: 170, accuracy: 0.45 (+/- 0.01)
        Max features: 35, num estimators: 190, accuracy: 0.46 (+/- 0.01)
        Max features: 40, num estimators: 10, accuracy: 0.34 (+/- 0.01)
        Max features: 40, num estimators: 30, accuracy: 0.40 (+/- 0.01)
        Max features: 40, num estimators: 50, accuracy: 0.42 (+/- 0.02)
        Max features: 40, num estimators: 70, accuracy: 0.43 (+/- 0.02)
        Max features: 40, num estimators: 90, accuracy: 0.44 (+/- 0.02)
        Max features: 40, num estimators: 110, accuracy: 0.45 (+/- 0.01)
        Max features: 40, num estimators: 130, accuracy: 0.45 (+/- 0.00)
        Max features: 40, num estimators: 150, accuracy: 0.45 (+/- 0.02)
        Max features: 40, num estimators: 170, accuracy: 0.45 (+/- 0.01)
        Max features: 40, num estimators: 190, accuracy: 0.45 (+/- 0.01)
        Max features: 45, num estimators: 10, accuracy: 0.33 (+/- 0.00)
        Max features: 45, num estimators: 30, accuracy: 0.40 (+/- 0.01)
        Max features: 45, num estimators: 50, accuracy: 0.42 (+/- 0.01)
        Max features: 45, num estimators: 70, accuracy: 0.43 (+/- 0.02)
        Max features: 45, num estimators: 90, accuracy: 0.44 (+/- 0.02)
        Max features: 45, num estimators: 110, accuracy: 0.44 (+/- 0.02)
        Max features: 45, num estimators: 130, accuracy: 0.44 (+/- 0.01)
        Max features: 45, num estimators: 150, accuracy: 0.45 (+/- 0.01)
        Max features: 45, num estimators: 170, accuracy: 0.45 (+/- 0.01)
        Max features: 45, num estimators: 190, accuracy: 0.45 (+/- 0.01)
    }


    To plot this output, the following code is used:

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    fig.clf()
    ax = fig.gca(projection='3d')
    x = rf_params[:,0]
    y = rf_params[:,1]
    z = rf_params[:,2]
    ax.scatter(x, y, z)
    ax.set_zlim(0.2, 0.5)
    ax.set_xlabel('Max features')
    ax.set_ylabel('Num estimators')
    ax.set_zlabel('Avg accuracy')
    plt.show()


    We can see that increasing the number of trees produces a better outcome. Also, 
    increasing the number of features produces better outcomes if we are able to see
    more features, but ultimately, if we're at about 20 to 30 features and we have 
    about 75 to 100 trees, that's as good as we're going to get an accuracy of 45%.
"""
