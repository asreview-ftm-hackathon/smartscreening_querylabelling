import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer
from pythonds.basic import Stack
from pythonds.trees import BinaryTree


# python -m spacy download nl_core_news_sm

def load_data(filepath):
    """This basic function loads csv/excel data into a Pandas dataframe
    containing strings.
    """

    file_type = filepath.split('.')[-1]
    if file_type == 'csv':
        return pd.read_csv(filepath, delimiter=';',
                                     encoding='utf-8')
    elif file_type == 'xlsx':
        return pd.read_excel(filepath)
    else:
        raise ValueError('Filetype not supported.')

def parse_query(raw_query):
    """This function parses the search query into a Binary Tree, where each
    leaf node is a search term and internal nodes represent boolean
    operators.
    """
    parsed_query = None
    return parsed_query

def normalize_text(text_vector):
    """"""
    nlp = spacy.load('nl_core_news_sm')

    normed_text_vector = []
    for document in text_vector:
        clean_doc = re.sub('[^0-9a-zA-Z]+', ' ', document)
        clean_doc = clean_doc.strip().lower()
        normed_text_vector.append(' '.join([w.lemma_ for w in nlp(clean_doc)]))

    return normed_text_vector

def get_vocabulary(query):
    """ ."""
    # extract terms from the leaf nodes of the query object.
    terms = None #TODO:


    normed_terms = normalize_text(terms)

    # remove duplicates.
    vocabulary = list(set(normed_terms))

    return vocabulary

def get_vectorizer(vocabulary):
    """ """
    # find the n-gram range of the vocabulary.
    min_n = min([len(term.split()) for term in vocabulary])
    max_n = max([len(term.split()) for term in vocabulary])

    return CountVectorizer(vocabulary=vocabulary, ngram_range=(min_n, max_n))

def vectorize_data(data, vectorizer):
    #TODO loop over columns of data
    column_data = None # maak list van strings uit series
    normed_data = normalize_text(column_data)
    pass

def select_subset():


    selection = None # is boolean array
    return selection
    pass

def save_subset(data, selection):
    pass

if __name__ == '__main__':
    data = load_data('test_data.csv')