import re
import os
import numpy as np
import gensim
import logging
import random

from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# Set up default logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Load pre-build Word2Vec model
#gmodel = gensim.models.KeyedVectors.load_word2vec_format('dataset/GoogleNews-vectors-negative300.bin', binary=True)

"""
    We can see different vectors by running the code:

    gmodel['cat']
    gmodel['dog']

    Also, we can investigate the similarity between different words:

    gmodel.similarity('cat', 'dog')     
    gmodel.similarity('cat', 'spatula')    

    The output is 76% and 12% respectively.
"""


# Create a function to format the document
def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent)  # strip html tags
    sent = re.sub(r'(\w)\'(\w)', '\1\2', sent)  # remove apostrophes
    sent = re.sub(r'\W', ' ', sent)  # remove punctuation
    sent = re.sub(r'\s+', ' ', sent)  # remove repeated spaces
    sent = sent.strip()
    return sent.split()


"""
    The above function will lowercase the text in the document, remove all HTML tags,
    apostrophes, punctuation, spaces and repeated spaces, and then ultimately break
    it apart by words.
"""


# Unsupervised training data
unsup_sentences = []

# source: http://ai.stanford.edu/~amaas/data/sentiment/, data from IMDB
for dirname in ["train/pos", "train/neg", "train/unsup", "test/pos", "test/neg"]:
    for fname in sorted(os.listdir("dataset/IMDB/" + dirname)):
        if fname[-4:] == '.txt':
            with open("dataset/IMDB/" + dirname + "/" + fname, encoding='UTF-8') as f:
                sent = f.read()
                words = extract_words(sent)
                unsup_sentences.append(TaggedDocument(
                    words, [dirname + "/" + fname]))

# source: http://www.cs.cornell.edu/people/pabo/movie-review-data/
for dirname in ["dataset/review_polarity/txt_sentoken/pos", "dataset/review_polarity/txt_sentoken/neg"]:
    for fname in sorted(os.listdir(dirname)):
        if fname[-4:] == '.txt':
            with open(dirname + "/" + fname, encoding='UTF-8') as f:
                for i, sent in enumerate(f):
                    words = extract_words(sent)
                    unsup_sentences.append(TaggedDocument(
                        words, ["%s/%s-%d" % (dirname, fname, i)]))

# source: https://nlp.stanford.edu/sentiment/, data from Rotten Tomatoes
with open("dataset/stanfordSentimentTreebank/original_rt_snippets.txt", encoding='UTF-8') as f:
    for i, line in enumerate(f):
        words = extract_words(sent)
        unsup_sentences.append(TaggedDocument(words, ["rt-%d" % i]))

# Shuffle the data


class PermuteSentences(object):
    def __init__(self, sents):
        self.sents = sents

    def __iter__(self):
        shuffled = list(self.sents)
        random.shuffle(shuffled)
        for sent in shuffled:
            yield sent


permuter = PermuteSentences(unsup_sentences)
model = Doc2Vec(permuter, dm=0, hs=1, size=50)


# Free up some space in the training data
model.delete_temporary_training_data(keep_inference=True)


# Save the model
model.save('reviews.d2v')


"""
    To infer the vector, we can run the code:
    model.infer_vector(extract_words("This place is not worth your time, let alone Vegas."))

    To test the model, we can run the following code:
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarity(
        [model.infer_vector(extract_words("This place is not worth your time, let alone Vegas."))],
        [model.infer_vector(extract_words("Service sucks."))])

    This will result in such output: array([[ 0.48211202]], dtype=float32)

    
    cosine_similarity(
        [model.infer_vector(extract_words("Highly recommended."))],
        [model.infer_vector(extract_words("Service sucks."))])

    This code will result in a slightlly different output: array([[ 0.28899333]], dtype=float32)
"""


# Load the real dataset for prediction
sentences = []
sentvecs = []
sentiments = []
for fname in ["yelp", "amazon_cells", "imdb"]:
    with open("dataset/sentiment labelled sentences/%s_labelled.txt" % fname, encoding='UTF-8') as f:
        for i, line in enumerate(f):
            line_split = line.strip().split('\t')
            sentences.append(line_split[0])
            words = extract_words(line_split[0])
            # create a vector for this document
            sentvecs.append(model.infer_vector(words, steps=10))
            sentiments.append(int(line_split[1]))

# shuffle sentences, sentvecs, sentiments together
combined = list(zip(sentences, sentvecs, sentiments))
random.shuffle(combined)
sentences, sentvecs, sentiments = zip(*combined)


# Create a K-Neighbours Classifier
clf = KNeighborsClassifier(n_neighbors=9)
clfrf = RandomForestClassifier()


# Compare the accuracy with a Random Forest and bag-of-words
scores = cross_val_score(clf, sentvecs, sentiments, cv=5)
np.mean(scores), np.std(scores)
"""
    Output: (0.75900000000000012, 0.016950909513454807)
"""

scores = cross_val_score(clfrf, sentvecs, sentiments, cv=5)
np.mean(scores), np.std(scores)
"""
    Output: (0.69766666666666655, 0.019988885800753264)
"""

pipeline = make_pipeline(
    CountVectorizer(), TfidfTransformer(), RandomForestClassifier())
scores = cross_val_score(pipeline, sentences, sentiments, cv=5)
np.mean(scores), np.std(scores)
"""
    Output: (0.73733333333333329, 0.015937377450509209)
"""
