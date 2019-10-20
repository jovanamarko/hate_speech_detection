import numpy as np
import openpyxl as openpyxl
from gensim.models import Word2Vec
from sklearn import metrics
from clean_tweets import clean_data, get_data


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


clean_tweets = clean_data()

num_features = 150  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

# ***Word2Vec***
# Initialize and train the model (this will take some time)
print("Training Word2Vec model...")
model = Word2Vec(clean_tweets, workers=num_workers,
                 size=num_features, min_count=min_word_count,
                 window=context, sample=downsampling, seed=1)


def get_features():
    tweet_features = getAvgFeatureVecs(clean_tweets, model, num_features)
    return tweet_features
