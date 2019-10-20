from clean_tweets import clean_data, get_data
from textblob import TextBlob
import nltk

nltk.download('averaged_perceptron_tagger')
data = clean_data()


def get_postags():
    pos_tags = []
    pos_tags_nltk = []
    for tweet in data:
        blob = TextBlob(tweet)
        pos_tags.append(blob.tags)

        pos_tags_nltk.append(nltk.pos_tag(tweet))

    return pos_tags


def get_postags_nltk():
    pos_tags_nltk = []
    for tweet in data:
        pos_tags_nltk.append(nltk.pos_tag(tweet))

    return pos_tags_nltk


print(get_postags()[:10])
print('\n')
print(get_postags_nltk()[:10])
