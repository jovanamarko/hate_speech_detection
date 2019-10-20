import numpy as np
import openpyxl as openpyxl
from gensim.models import Word2Vec
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from clean_tweets import clean_data, get_data
from word2vec_features import getAvgFeatureVecs, get_features
from ngrams import get_n_grams, get_features_tfidf

out = openpyxl.load_workbook('outcome.xlsx', read_only=False)
sh = out.get_sheet_by_name('Sheet1')
data = get_data()

clean_tweets = clean_data()

print(clean_tweets[:10])


num_features = 150  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

# word2vec features
tweet_features = get_features()
# KeyedVectors.load_word2vec_format()

# ***TfidfVectorizer***

tweet_features_tfidf = get_features_tfidf()
np.asarray(tweet_features_tfidf)
tweet_features_tfidf = np.nan_to_num(tweet_features_tfidf)

ngrams = get_n_grams()


# Assigning classifiers
forest = RandomForestClassifier(n_estimators=100)
regression = LogisticRegression(verbose=0)
extraTrees = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, verbose=0, class_weight='balanced', bootstrap=True)
# gnb = GaussianNB()

# Training
forest = forest.fit(tweet_features_tfidf[:26820], data['class'][:26820])
regression = regression.fit(tweet_features_tfidf[:26820], data['class'][:26820])
extraTrees = extraTrees.fit(tweet_features_tfidf[:26820], data['class'][:26820])
# gnb = gnb.fit(tweet_features[:26820], data['class'][:26820])

# Prediction
result = forest.predict(tweet_features_tfidf[26820:])
resultReg = regression.predict(tweet_features_tfidf[26820:])
resultET = extraTrees.predict(tweet_features_tfidf[26820:])
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
