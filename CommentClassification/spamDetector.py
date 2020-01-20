"""
    Spam Detector using Random Forests and bag-of-words. 
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV


# Read the file
d = pd.read_csv("dataset/YouTube/Youtube01-Psy.csv")

"""
    To check count of rows with a particular feature, the following code can
    be run:

    len(d.query('CLASS == 1'))
    len(d.query('CLASS == 0'))

    Their output is 175 and 175 respectively.
"""


# Create a vector and an analyzer
vectorizer = CountVectorizer()
analyze = vectorizer.build_analyzer()

dvec = vectorizer.fit_transform(d['CONTENT'])

"""
    The first step is to discover which words are present in the dataset. Next step
    is to create a bag of words for these phrases. 
"""


# Shuffle the data
dshuf = d.sample(frac=1)


# Create training and test set
d_train = dshuf[:300]
d_test = dshuf[300:]
d_train_att = vectorizer.fit_transform(
    d_train['CONTENT'])  # fit bag-of-words on training set
d_test_att = vectorizer.transform(d_test['CONTENT'])  # reuse on testing set
d_train_label = d_train['CLASS']
d_test_label = d_test['CLASS']

"""
    In the above code, vectorizer.fit_transform(d_train['CONTENT']) is an important step. 
    At that stage, we have a training set that we want to perform a fit transform on, 
    which means it will learn the words and also produce the matrix. However, for the 
    testing set, we don't perform a fit transform again, since we don't want the model 
    to learn from different words for the testing data. We will use the same words it 
    learned on the training set. 
"""


# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=80)
clf.fit(d_train_att, d_train_label)
clf.score(d_test_att, d_test_label)                 # output: 0.96


# Create a confusion matrix
pred_labels = clf.predict(d_test_att)
confusion_matrix(d_test_label, pred_labels)


# Perform cross-validation
scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)

"""
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    Outputs:

    Accuracy: 0.95 (+/- 0.05)
"""


# Combine all datasets
d = pd.concat([pd.read_csv("dataset/YouTube/Youtube01-Psy.csv"),
               pd.read_csv("dataset/YouTube/Youtube02-KatyPerry.csv"),
               pd.read_csv("dataset/YouTube/Youtube03-LMFAO.csv"),
               pd.read_csv("dataset/YouTube/Youtube04-Eminem.csv"),
               pd.read_csv("dataset/YouTube/Youtube05-Shakira.csv")])


# Shuffle the data
dshuf = d.sample(frac=1)
d_content = dshuf['CONTENT']
d_label = dshuf['CLASS']


# Create a pipeline
pipeline = Pipeline([
    ('bag-of-words', CountVectorizer()),
    ('random forest', RandomForestClassifier()),
])

"""
    Alternative code:
    
    make_pipeline(CountVectorizer(), RandomForestClassifier())
"""


# Fit the pipeline
pipeline.fit(d_content[:1500], d_label[:1500])

"""
    print(pipeline.score(d_content[1500:], d_label[1500:]))

    outputs:

    0.9451754385964912

    To test a single comment, following code can be run:
    
    pipeline.predict(["plz subscribe to my channel"])
"""


# Perform cross-validation
scores = cross_val_score(pipeline, d_content, d_label, cv=5)

"""
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    outputs:

    Accuracy: 0.96 (+/- 0.03)
"""


# Add TF-IDF (term frequency–inverse document frequency)
pipeline2 = make_pipeline(CountVectorizer(),
                          TfidfTransformer(norm=None),
                          RandomForestClassifier())

scores = cross_val_score(pipeline2, d_content, d_label, cv=5)

"""
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    outputs:

    Accuracy: 0.96 (+/- 0.01)
"""


# Parameter search
parameters = {
    'countvectorizer__max_features': (None, 1000, 2000),
    # unigrams or bigrams
    'countvectorizer__ngram_range': ((1, 1), (1, 2)),
    'countvectorizer__stop_words': ('english', None),
    # effectively turn on/off tfidf
    'tfidftransformer__use_idf': (True, False),
    'randomforestclassifier__n_estimators': (20, 50, 100)
}

grid_search = GridSearchCV(pipeline2, parameters, n_jobs=-1, verbose=1)

grid_search.fit(d_content, d_label)

"""
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    outputs:
    
    Best score: 0.961
    Best parameters set:
        countvectorizer__max_features: 2000
        countvectorizer__ngram_range: (1, 2)
        countvectorizer__stop_words: None
        randomforestclassifier__n_estimators: 100
        tfidftransformer__use_idf: False

"""
