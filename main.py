import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer
from pythonds.basic import Stack
from pythonds.trees import BinaryTree
# paste in CLI:      python -m spacy download nl_core_news_sm

def load_data(filepath, columns=['title','abstract']):
    """This basic function loads specified columns of csv/excel data into a
    Pandas dataframe.
    """

    file_type = filepath.split('.')[-1]
    if file_type == 'csv':
        return pd.read_csv(filepath, delimiter=';',
                                     encoding='utf-8',
                                     usecols=columns)
    elif file_type == 'xlsx':
        return pd.read_excel(filepath, usecols=columns)
    else:
        raise ValueError('Filetype not supported.')

def parse_query(raw_query):
    """This function parses the search query into a Binary Tree, where each
    leaf node is a search term and internal nodes represent boolean
    operators (AND, OR and NOT). (It uses basic Stack- and BinaryTree objects
    of the pythonds package.)

    Example: the raw_query '(X AND (Y OR (NOT Z)))'
             turns into:
                             AND
                        X           OR
                                 Y      NOT
                                      Z
    """

    if not (raw_query.count("(") == raw_query.count(")")
                and raw_query.count("(") == (raw_query.count("AND")
                                             + raw_query.count("OR")
                                             + raw_query.count("NOT"))):
        raise ValueError('Invalid Query: Unmatched brackets or operators.')

    # add splitting points using private unicode character
    raw_query = raw_query.replace('\uf026', '')
    subs = [('(','\uf026(\uf026'),
            (')','\uf026)\uf026'),
            ('AND','\uf026AND\uf026'),
            ('OR','\uf026OR\uf026'),
            ('NOT','\uf026NOT\uf026')]

    # get nodes
    for substitution in subs:
        raw_query = raw_query.replace(substitution[0], substitution[1])
    split_query = raw_query.split('\uf026')
    node_list = [x for x in split_query if x != '' and x != ' ']

    qStack = Stack()
    parsed_query = BinaryTree('')
    qStack.push(parsed_query)
    currentTree = parsed_query

    if len(node_list) == 1: # query only contains a search term
        currentTree.setRootVal(node_list[0])
    else:
        for node in node_list:
            if node == '(':
                currentTree.insertLeft('')
                qStack.push(currentTree)
                currentTree = currentTree.getLeftChild()
            elif node in ['AND', 'OR']:
                currentTree.setRootVal(node)
                currentTree.insertRight('')
                qStack.push(currentTree)
                currentTree = currentTree.getRightChild()
            elif node == 'NOT':
                currentTree = qStack.pop()
                currentTree.setRootVal(node)
                qStack.push(currentTree)
                currentTree = currentTree.getLeftChild()
            elif node == ')':
                currentTree = qStack.pop()
            elif node not in ['AND', 'OR', 'NOT', ')']: # leaf node
                currentTree.setRootVal(node)
                currentTree = qStack.pop()

    return parsed_query

def normalize_text(text_vector, language_model):
    """This function normalizes a vector of documents using the Spacy package.
    In particular: it replaces all non-alphanumeric characters with spaces
    before stripping whitespace from the edges of each document and setting
    everything to lowercase. It then uses a Spacy language object to lemmatize
    each dutch word in each document. For more information on these language
    objects see:
    https://spacy.io/usage/models
    """
    nlp = language_model

    normed_text_vector = []
    for document in text_vector:
        clean_doc = re.sub('[^0-9a-zA-Z]+', ' ', document)
        clean_doc = clean_doc.strip().lower()
        normed_text_vector.append(' '.join([w.lemma_ for w in nlp(clean_doc)]))

    return normed_text_vector


def get_vocabulary(query, language_model):
    """This function extracts the search terms (leaf nodes) from the parsed
    query, and constructs the vocabulary for the text vectorizer object by
    applying text normalization to them.
    """
    def _getleafnodes(query):
        terms = []
        if query.isLeaf():
            return terms + [query.getRootVal()]
        elif query.leftChild and not query.rightChild:
            return terms + _getleafnodes(query.getLeftChild())
        elif query.rightChild and not query.leftChild:
            return terms + _getleafnodes(query.getRightChild())
        else:   # has two children
            return terms + _getleafnodes(query.getLeftChild()) \
                         + _getleafnodes(query.getRightChild())

    # extract terms from the leaf nodes of the query object.
    terms = _getleafnodes(query)

    # remove column keys
    for i, term in enumerate(terms):
        column_key = False
        for j, char in enumerate(term):
            if char == ':':
                column_key = True
                split_index = j
        if column_key:
            terms[i] = term[split_index:]

    normed_terms = normalize_text(terms, language_model)

    # remove duplicates.
    vocabulary = list(set(normed_terms))

    return vocabulary

def get_vectorizer(vocabulary):
    """This function constructs a text vectorization object fit to a specified
     vocabulary, using the ScikitLearn package. This vectorization uses a
     binary (present/not present) bag-of-words approach.
     """
    # find the n-gram range of the vocabulary.
    min_n = min([len(term.split()) for term in vocabulary])
    max_n = max([len(term.split()) for term in vocabulary])

    return CountVectorizer(vocabulary=vocabulary,
                           ngram_range=(min_n, max_n),
                           binary=True)

def vectorize_data(data, vectorizer):
    """This function takes tabular data in the form of a Pandas Dataframe,
     as well as a sklearn text-vectorization object-- and returns a dictionary
     containing columnname -> vectorizedtextdata key/value pairs. Since the
     (binary) vectorization object is fit to a specified vocabulary, the
     vectorized text data consists of a binary array representing the presence
     of each word (column) of the vocabulary in each document (row).
     """
    vectorized_data = {}
    for col in data.columns:
        column_data = data[col].apply(str)
        normed_data = normalize_text(column_data, language_model)
        vectorized_data[col] = vectorizer.transform(normed_data)

    return vectorized_data

def select_subset():


    selection = None # is boolean array
    return selection
    pass

def save_subset(data, selection):
    pass

if __name__ == '__main__':
    # Dutch language model (has to be installed separately on CLI, see line 7)
    language_model = spacy.load('nl_core_news_sm')

    data = load_data('test_data.csv', columns=['type','title','abstract'])

    test_queries = ['(abstract:zalm AND (abstract:evi OR (NOT type:help)))',
                    '((abstract:evi OR (NOT type:help)) AND abstract:zalm)',
                    '(abstract:zalm pasta AND (abstract:evi OR (NOT type:help me)))',
                    '(abstract:zalm pasta AND (evi OR (NOT type:do not help me)))',
                    'abstract:zalm pasta',
                    '(abstract:evi AND type:help)',
                    '(abstract:42222*& OR type:life)'
                    ]
    test_trees = [parse_query(x) for x in test_queries]
    test_vocabs = [get_vocabulary(x, language_model) for x in test_trees]
    test_vects = [get_vectorizer(x) for x in test_vocabs]
    tested_data = vectorize_data(data, test_vects[0])
    print(tested_data)