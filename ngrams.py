from sklearn.feature_extraction.text import TfidfVectorizer
from clean_tweets import clean_data, get_data
import pandas as pd

clean_tweets = clean_data()
vectorizer = TfidfVectorizer(ngram_range=(2, 3), max_features=1500)


def get_n_grams():
    tweet_features_tfidf = get_features_tfidf()
    ngrams = pd.DataFrame(tweet_features_tfidf.todense(), columns=vectorizer.get_feature_names())
    # print('Printing n-grams:')
    # print(ngrams)
    return ngrams


def get_features_tfidf():
    tweet_features_tfidf = vectorizer.fit_transform(clean_tweets)

    return tweet_features_tfidf


get_n_grams()
