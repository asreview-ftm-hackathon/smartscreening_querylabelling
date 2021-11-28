import numpy as np
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

def normalize_text(text_vector, nlp):
    """This function normalizes a vector of documents using the Spacy package.
    In particular: it replaces all non-alphanumeric characters with spaces
    before stripping whitespace from the edges of each document and setting
    everything to lowercase. It then uses a Spacy language object to lemmatize
    each dutch word in each document. For more information on these language
    objects see:
    https://spacy.io/usage/models
    """

    normed_text_vector = []
    for document in text_vector:
        clean_doc = re.sub('[^0-9a-zA-Z]+', ' ', document)
        clean_doc = clean_doc.strip().lower()
        normed_text_vector.append(' '.join([w.lemma_ for w in nlp(clean_doc)]))

    return normed_text_vector


def get_vocabulary(query, nlp):
    """This function extracts the search terms (leaf nodes) from the parsed
    query, and constructs the vocabulary for the text vectorizer object by
    applying text normalization to them.
    """
    def _getleafnodes(query):
        terms = []
        if query.isLeaf():
            return terms + [query.key]
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
                break
        if column_key:
            terms[i] = term[split_index:]

    normed_terms = normalize_text(terms, nlp)

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
    vectorizer = CountVectorizer(vocabulary=vocabulary,
                           ngram_range=(min_n, max_n),
                           binary=True)
    # get vocabulary_ attribute
    vectorizer.get_feature_names_out()
    return vectorizer

def vectorize_data(data, vectorizer, nlp):
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
        normed_data = normalize_text(column_data, nlp)
        vectorized_data[col] = vectorizer.transform(normed_data)

    return vectorized_data

def select_subset(vec_data, query, legend, nlp):
    """This recursive function takes vectorized data and a parsed query as
     input, and returns a boolean array that indexes the original data s.t.
     the True values refer to entries that conform to the searched query.
     It does this through boolean operations between Pandas Series while
     traversing the tree-structured query.
     The additionally required argument 'legend' should be a dictionary which
     maps the vocabulary to feature indices--
     (this is generally the vocabulary_ attribute of the CountVectorizer object
     obtained from get_vectorizer()).
     """
    if query.isLeaf(): # we found a search term.
        term = query.key.strip()

        # get column key. (e.g 'abstract' in 'abstract:Mozambique')
        column_key = False
        for i, char in enumerate(term):
            if char == ':':
                split_index = i
                column_key = term[:split_index]
                break

        if column_key:
            if column_key not in vec_data.keys():
                raise ValueError(f'Query contains invalid column key. '
                                 f'Column key: {column_key}')

            term = term[split_index+1:]
            term_index = legend[normalize_text([term],nlp)[0]]
            # get column data.
            column_data = vec_data[column_key].toarray()
            return column_data[:,term_index].astype(bool)
        else:   # search term can be in any column.
            term_index = legend[normalize_text([term],nlp)[0]]
            # OR operation between all columns.
                # get nr of entries in the data
            N = list(vec_data.values())[0].shape[0]
            accumulator = np.zeros(N).astype(bool)
            for column_key in vec_data.keys():
                # get column data.
                column_data = vec_data[column_key].toarray()
                accumulator = (accumulator
                                | column_data[:,term_index].astype(bool))
            return accumulator
    elif query.key == 'AND':
        return (select_subset(vec_data, query.getLeftChild(), legend, nlp)
                    & select_subset(vec_data, query.getRightChild(), legend, nlp))
    elif query.key == 'OR':
        return (select_subset(vec_data, query.getLeftChild(), legend, nlp)
                    | select_subset(vec_data, query.getRightChild(), legend, nlp))
    elif query.key == 'NOT':
        return ~ (select_subset(vec_data, query.getLeftChild(), legend, nlp))
    else:
        raise ValueError('Query was not parsed correctly.')

if __name__ == '__main__':
    # Dutch language model (has to be installed separately on CLI, see line 7)
    language_model = spacy.load('nl_core_news_sm')

    filepath = input('what is the filepath to your dataset? (e.g data.csv)\n'
                     'The script handles csv or excel files.\n')
    raw_query = input('What is your query?\n')

    # load data
    data = load_data('test_data.csv', columns=['type','title','abstract'])

    # parse the query
    parsed_query = parse_query(raw_query)

    # construct the text vectorizer
    vocabulary = get_vocabulary(parsed_query, language_model)
    vectorizer = get_vectorizer(vocabulary)
    feature_legend = vectorizer.vocabulary_

    # vectorize the data according to the restricted vocabulary
    searched_data = vectorize_data(data, vectorizer, language_model)

    selection = select_subset(searched_data,
                              parsed_query,
                              feature_legend,
                              language_model)

    # select and save pruned dataset
    data_subset = data.iloc[selection]

    # TODO: THIS DOESNT WORK WELL RIGHT NOW
    data_subset.to_csv('data_subset.csv')