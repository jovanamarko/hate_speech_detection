import openpyxl as openpyxl
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.utils import shuffle


def get_data():
    dataset = pd.read_csv('processed_dataset.csv')
    data1 = shuffle(dataset)
    data2 = shuffle(data1)
    data = shuffle(data2)

    return data


def clean_data():
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    clean_tweets = []
    data = get_data()

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
        filtered_wordsC = [word for word in filtered_wordsB if
                           "http" not in word and "www" not in word]  # filter the urls
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

    return clean_tweets
