from distutils.archive_util import make_archive
import numpy as np
import pandas as pd
import pickle as pkl
import re
# Model training and evaluation
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


def preprocess(data_set):
    # Remove links
    data_set["posts"] = data_set["posts"].apply(lambda x: re.sub(
        r'https?:\/\/.*?[\s+]', '', x.replace("|", " ") + " "))

    # Keep the End Of Sentence characters
    data_set["posts"] = data_set["posts"].apply(
        lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    data_set["posts"] = data_set["posts"].apply(
        lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    data_set["posts"] = data_set["posts"].apply(
        lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))

    # Strip Punctation
    data_set["posts"] = data_set["posts"].apply(
        lambda x: re.sub(r'[\.+]', ".", x))

    # Remove Non-words
    data_set["posts"] = data_set["posts"].apply(
        lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # Remove multiple fullstops
    data_set["posts"] = data_set["posts"].apply(
        lambda x: re.sub(r'[^\w\s]', '', x))

    # Convert posts to lowercase
    data_set["posts"] = data_set["posts"].apply(lambda x: x.lower())

    # Remove multiple letter repeating words !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    data_set["posts"] = data_set["posts"].apply(
        lambda x: re.sub(r'([a-z])\1{2,}[\s|\w]*', '', x))

    # Remove very short or long words
    data_set["posts"] = data_set["posts"].apply(
        lambda x: re.sub(r'(\b\w{0,3})?\b', '', x))
    data_set["posts"] = data_set["posts"].apply(
        lambda x: re.sub(r'(\b\w{30,1000})?\b', '', x))

    # Remove MBTI Personality Words
    pers_types = ['infp', 'infj', 'intp', 'intj', 'entp', 'enfp', 'istp',
                  'isfp', 'entj', 'istj', 'enfj', 'isfj', 'estp', 'esfp', 'esfj', 'estj']
    p = re.compile("(" + "|".join(pers_types) + ")")
    data_set['posts'] = data_set["posts"].apply(lambda x: re.sub(p, '', x))

    return data_set


def preprocess_sentence(sentence):
    # Remove links
    sentence = re.sub(r'https?:\/\/.*?[\s+]', '', sentence.replace("|", " ") + " ")

    # Keep the End Of Sentence characters
    sentence = re.sub(r'\.', ' EOSTokenDot ', sentence + " ")
    sentence = re.sub(r'\?', ' EOSTokenQuest ', sentence + " ")
    sentence = re.sub(r'!', ' EOSTokenExs ', sentence + " ")

    # Strip Punctation
    sentence = re.sub(r'[\.+]', ".", sentence)

    # Remove Non-words
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)

    # Remove multiple fullstops
    sentence = re.sub(r'[^\w\s]', '', sentence)

    # Convert posts to lowercase
    sentence = sentence.lower()

    # Remove multiple letter repeating words !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sentence = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', sentence)

    # Remove very short or long words
    sentence = re.sub(r'(\b\w{0,3})?\b', '', sentence)
    sentence = re.sub(r'(\b\w{30,1000})?\b', '', sentence)

    # Remove MBTI Personality Words
    pers_types = ['infp', 'infj', 'intp', 'intj', 'entp', 'enfp', 'istp',
                  'isfp', 'entj', 'istj', 'enfj', 'isfj', 'estp', 'esfp', 'esfj', 'estj']
    p = re.compile("(" + "|".join(pers_types) + ")")
    sentence = re.sub(p, '', sentence)

    return sentence


def get_data():
    # loading dataset
    # data_set = pd.read_csv("C:\\Users\Amalia\\Desktop\\Inteligenta-Artificiala-facultate\\Proiect\\mbti_1.csv")
    data_set = pd.read_csv(
        "C:\\Users\\ioana\\Documents\\GitHub\\Inteligenta-Artificiala-facultate\\Proiect\\mbti_1.csv")
    data_set = preprocess(data_set)

    # Remove posts with less than X words
    min_words = 15
    print("Before : Number of posts", len(data_set))
    data_set["no. of. words"] = data_set["posts"].apply(
        lambda x: len(re.findall(r'\w+', x)))
    data_set = data_set[data_set["no. of. words"] >= min_words]

    types = list(data_set['type'])
    posts = list(data_set['posts'])

    # ######################
    # Convert the sentences into a numerical representation

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(posts)
    y = types

    # Split the dataset into a training set and a test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4)

    # return preprocess(data_set)
    return X_train, X_test, Y_train, Y_test, vectorizer
