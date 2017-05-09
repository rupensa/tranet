import json
import logging
import os
import string

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer

import config
from utilities.domain_dependent_stopwords import scientific_stopwords_list


def get_stopwords():
    '''
    Method created for 
    :return: a string or a list of stopwords
    '''
    return scientific_stopwords_list + stopwords.words('english')


def compute_tf(data, stopwords_list, use_lemmer=True, min_df=2, max_df=0.8):
    lemmer_tokenizer = None

    if use_lemmer:
        lemmer_tokenizer = LemNormalize

    min_df = min_df if len(data) > min_df else 1
    max_df = max_df if max_df * len(data) >= min_df else 1.0

    # tf
    tf_vectorizer = CountVectorizer(tokenizer=lemmer_tokenizer,
                                    max_df=max_df, min_df=min_df,
                                    max_features=None,
                                    stop_words=stopwords_list, token_pattern="[a-zA-Z]{3,}")

    try:
        tf = tf_vectorizer.fit_transform(data)
        tf_features_names = tf_vectorizer.get_feature_names()
    except:
        logging.warning('The computed tf matrix is empty. Check stopwords.')
        tf = []
        tf_features_names = []

    return tf, tf_features_names


def save_matrix(filename, matrix):
    if issparse(matrix):
        matrix = matrix.toarray()

    with open(os.path.join(config.__data_folder_path, filename), 'w') as f:
        for row in range(len(matrix)):
            line = " ".join(map(str, matrix[row])) + '\n'
            f.write(line)


def save_json(filename, data):
    """
    Save a json file containing the dump of the data
    :param filename: the file name
    :param data: the data to cache
    :return:
    """
    cache_path = os.path.join('..', config.__data_folder_path, filename)
    with open(cache_path, 'w') as json_file:
        json.dump(data, json_file)


def get_json(filename):
    """
    Retrieves a cached file
    :param filename: the name of the file to retrieve
    :return: the retrieved data
    """
    cache_path = os.path.join('..', config.__data_folder_path, filename)
    if os.path.exists(cache_path):
        with open(cache_path) as json_data:
            return json.load(json_data)

    return None


def read_from_file(filename):
    matrix = np.loadtxt(os.path.join(config.__data_folder_path, filename),
                        dtype=float, delimiter=' ')
    return matrix


def print_top_words_per_topic(topic_word_matrix, feature_names, n_top_words=10):
    print("Print top {0} words per topic".format(n_top_words))
    for topic_idx, topic in enumerate(topic_word_matrix):
        print("Topic #{0}:".format(topic_idx))
        print("\t".join([feature_names[i]
                         for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print('\t'.join([str(int(i)) for i in sorted(topic, reverse=True)[:n_top_words]]))
    print()


def LemTokens(tokens):
    lemmer = WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]


def LemNormalize(text):
    # convert non ascii characters
    text = text.encode('ascii', 'replace').decode()
    # remove punctuation and digits
    remove_punct_and_digits = dict([(ord(punct), ' ') for punct in string.punctuation + string.digits])
    transformed = text.lower().translate(remove_punct_and_digits)
    # shortword = re.compile(r'\W*\b\w{1,2}\b')
    # transformed = shortword.sub('', transformed)

    # tokenize the transformed string
    tokenized = nltk.word_tokenize(transformed)

    # remove short words (less than 3 char)
    tokenized = [w for w in tokenized if len(w) > 3]
    tokenizer = LemTokens(tokenized)

    return tokenizer
