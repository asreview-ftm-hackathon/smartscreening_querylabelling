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

    if not (raw_query.count("(") == raw_query.count(")") and raw_query.count("(") == raw_query.count("AND") + raw_query.count("OR") + raw_query.count("NOT")):
        raise ValueError('Unmatched brackets or operators')
    #TODO hier kunnen nog een heleboel andere dingen misgaan, dan krijg je gewoon een gekke boom

    raw_query = raw_query.replace("(", "}(}").replace(")", "})}").replace("AND", "}AND}").replace("OR", "}OR}").replace("NOT", "}NOT}")
    split_query = raw_query.split("}")
    query_list = [x for x in split_query if x != '' and x != ' ']

    qStack = Stack()
    parsed_query = BinaryTree('')
    qStack.push(parsed_query)
    currentTree = parsed_query

    # if no boolean operations are given, the list only contains brackets and the key word
    if len(query_list) == 1:
        currentTree.setRootVal(query_list[0])

    else:

        for q in query_list:
            if q == '(':
                currentTree.insertLeft('')
                qStack.push(currentTree)
                currentTree = currentTree.getLeftChild()

            elif q in ['AND', 'OR']:
                currentTree.setRootVal(q)
                currentTree.insertRight('')
                qStack.push(currentTree)
                currentTree = currentTree.getRightChild()

            elif q == 'NOT':
                currentTree = qStack.pop()
                currentTree.setRootVal(q)
                qStack.push(currentTree)
                currentTree = currentTree.getLeftChild()

            elif q == ')':
                currentTree = qStack.pop()

            elif q not in ['AND', 'OR', 'NOT', ')']:
                currentTree.setRootVal(q)
                parent = qStack.pop()
                currentTree = parent

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
    """ This function extracts the leaf nodes from the search query into
    ..........."""

    # extract terms from the leaf nodes of the query object.
    terms = []
    if query.getLeftChild() is None and query.getRightChild() is None:
        terms.append(query.getRootVal())
    if query.getLeftChild():
        terms.append(get_vocabulary(query.getLeftChild()))
    if query.getRightChild():
        terms.append(get_vocabulary(query.getRightChild()))

    #TODO: dit is wat  ik nu heb, column names moeten nog geschrapt worden dan I guess
    # ik weet niet zo goed waarom deze nodig is help


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
    data = load_data('C:/Users/Hendr076/FTM_hackathon/smartscreening_FUNnelyourdata_TheFUNnel/test_data.csv')
