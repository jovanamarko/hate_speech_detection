import nltk
import openpyxl as openpyxl
import pandas as pd
import numpy as np
import math

from sklearn.utils import shuffle
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from collections import defaultdict

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0.

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    # Initialize a counter
    counter = 0.

    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    # Loop through the reviews
    for review in reviews:

        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))

        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)

        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


stop_words = set(stopwords.words('english'))

ps = PorterStemmer()

dataset = pd.read_csv('processed_dataset.csv')

out = openpyxl.load_workbook('outcome.xlsx', read_only=False)
sh = out.get_sheet_by_name('Sheet1')

# did not do randomize to the data !!!
# did not used cross validation - consider this in future extraction of the project

clean_tweets = []
data1 = shuffle(dataset)
data2 = shuffle(data1)
data = shuffle(data2)
print(data[:8])

for head, row in data.iterrows():
    string = str(row[1])
    text = ''
    for word in string.split(' '):
        if '@' in word:  # filtering the tags
            continue
        else:
            if text == '':
                text = word
            else:
                text = text + ' ' + word

    words = word_tokenize(text)
    filtered_words = [w.lower() for w in words if not w in stop_words]  # filter the stop words IN ROW AFTER ROW
    filtered_wordsA = [word for word in filtered_words if word.isalpha()]  # filter the non alphabetic symbols
    filtered_wordsB = [word for word in filtered_wordsA if word != 'rt']  # filter the rt word
    filtered_wordsC = [word for word in filtered_wordsB if "http" not in word and "www" not in word]  # filter the urls
    filtered_wordsD = [word.replace('\n', ' ') for word in filtered_wordsC]
    filtered_wordsE = [word.replace('  ', ' ') for word in filtered_wordsD]
    filtered_wordsF = [word.replace('   ', ' ') for word in filtered_wordsE]

    filtered_words_final = [ps.stem(word) for word in filtered_wordsF]  # stemming

    tweet = ''
    for i in filtered_words_final:
        if tweet == '':
            tweet = i
        else:
            tweet = tweet + ' ' + i

    clean_tweets.append(tweet)

print(clean_tweets[:10])

# Vector transformation

# ***Bag-of-words***
# vectorizer = CountVectorizer(analyzer="word",
#                              tokenizer=None,
#                              preprocessor=None,
#                              stop_words=None,
#                              max_features=5000)
#
# tweet_features = vectorizer.fit_transform(clean_tweets)
# np.asarray(tweet_features)

num_features = 150  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print("Training Word2Vec model...")
model = Word2Vec(clean_tweets, workers=num_workers,
                 size=num_features, min_count=min_word_count,
                 window=context, sample=downsampling, seed=1)

tweet_features = getAvgFeatureVecs(clean_tweets, model, num_features)

# Assigning classifiers
forest = RandomForestClassifier(n_estimators=100)
regression = LogisticRegression(verbose=0)
extraTrees = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, verbose=0, class_weight='balanced', bootstrap=True)
# gnb = GaussianNB()

np.asarray(tweet_features)
print("PRINTING RESULT OF VECTORIZED TWEETS")
print(tweet_features[:10])
print(np.any(np.isnan(tweet_features)))
print(np.all(np.isfinite(tweet_features)))

tweet_features = np.nan_to_num(tweet_features)

print("PRINTING AGAIN IF THERE ARE NANS")
print(np.any(np.isnan(tweet_features)))
print(np.all(np.isfinite(tweet_features)))

# Training
forest = forest.fit(tweet_features[:26820], data['class'][:26820])
regression = regression.fit(tweet_features[:26820], data['class'][:26820])
extraTrees = extraTrees.fit(tweet_features[:26820], data['class'][:26820])
# gnb = gnb.fit(tweet_features[:26820], data['class'][:26820])

# Prediction
result = forest.predict(tweet_features[26820:])
resultReg = regression.predict(tweet_features[26820:])
resultET = extraTrees.predict(tweet_features[26820:])
# resultGnb = gnb.predict(tweet_features[26820:])

# Results
accuracyForest = metrics.accuracy_score(data['class'][26820:], result)
accuracyReg = metrics.accuracy_score(data['class'][26820:], resultReg)
accuracyET = metrics.accuracy_score(data['class'][26820:], resultET)

# Write the results into excel file
string = accuracyForest * 100
sh['B2'] = string
sh['B3'] = str(accuracyReg * 100) + "%"
sh['B4'] = str(accuracyET * 100) + "%"

print("Random Forest: " + str(round(accuracyForest * 100)) + "%")
print("Logistic Regression: " + str(round(accuracyReg * 100)) + "%")
print("Extra Trees: " + str(round(accuracyET * 100)) + "%")

precision = dict()
recall = dict()
average_precision = dict()
# for i in range(data['class']):
#     precision[i], recall[i], _ = precision_recall_curve(data['class'][17300:][:, i], resultReg[:, i])
#     average_precision[i] = average_precision_score(data['class'][17300:][:, i], resultReg[:, i])
#
# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(data['class'][17300:].ravel(), resultReg.ravel())
# average_precision["micro"] = average_precision_score(data['class'][17300:], resultReg, average="micro")
# print('Average precision score, micro-averaged over all classes: {0:0.2f}'
#       .format(average_precision["micro"]))
