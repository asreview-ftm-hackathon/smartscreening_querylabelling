#MIT License
#
#Copyright (c) 2021 ASReview hackathon for Follow the Money
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer
from pythonds.basic import Stack
from pythonds.trees import BinaryTree
import en_core_web_sm # paste in CLI:  python -m spacy download en_core_web_sm


def load_data(filepath, columns=['title','abstract']):
    """Loads specified columns of csv/excel data.

    Arguments
    ---------
    filepath: str
        Path to file (e.g. 'data.csv')
    columns: list
        List of strings specifying the column names in the data to load.

    Returns
    -------
    pandas.DataFrame
        Pandas object containing the loaded tabular data. If labels are not
        loaded, a 'label_included' column is added (filled with -1).

    """

    file_type = filepath.split('.')[-1]
    if file_type == 'csv':
        df = pd.read_csv(filepath, delimiter=';',
                                     encoding='utf-8',
                                     usecols=columns)
    elif file_type == 'xlsx':
        df = pd.read_excel(filepath, usecols=columns)
    else:
        raise ValueError('Filetype not supported.')

    if 'label_included' not in df.columns:
        df['label_included'] = np.full(df.shape[0], -1, dtype=int)

    return df

def normalize_text(docs, lang_model):
    """Normalizes a vector of documents using the Spacy package.
    In particular: replaces all non-alphanumeric characters with spaces before
    stripping whitespace from the edges of each document and setting everything
    to lowercase. It then uses a Spacy language object to lemmatize each word
    in each document. For more information on these language objects see:
    https://spacy.io/usage/models

    Arguments
    ---------
    docs: iterable
        Iterable of strings, where each string represents a single document.
    lang_model: spacy.Language
        Spacy language object which allows for the lemmatization.

    Returns
    -------
    normed_docs: list
        List of strings, where each string represents a normalized document.

    """

    normed_docs = []
    for doc in docs:
        clean_doc = re.sub('[^0-9a-zA-Z]+', ' ', doc)
        clean_doc = clean_doc.strip().lower()
        normed_docs.append(' '.join([w.lemma_ for w in lang_model(clean_doc)]))

    return normed_docs

def parse_query(raw_query, lang_model):
    """Parses a search query into a Binary Tree, using the pythonds package.

    Arguments
    ---------
    raw_query: str
        The raw query of search terms connected by boolean operations.
        Formatting is strict:
            All boolean operations (AND, OR, NOT) need to be uppercase.
            Each operation needs to match enclosing parentheses.
            Within a search term, the first ':' splits the target column name
            (to the left) and the search term itself (to the right).
        Example:    '(X AND (abstract:Y OR (NOT Z)))'

    lang_model: spacy.Language
        Spacy language object passed to normalize_text() for lemmatization.

    Returns
    -------
    query_tree: pythonds.trees.BinaryTree
        The binary tree object representing the parsed search query. Each leaf
        node is a search term and internal nodes represent boolean operations.
        The 'key' attribute of each node is a string containing either the
        boolean operation or the raw search term. Leaf nodes have the
        attributes 'targetcolumn' and 'normedterm', which contain the target
        column name and the normalized search term respectively.
        Example:
                             AND
                        X            OR
                           abstract:Y      NOT
                                         Z
    """

    if not (raw_query.count("(") == raw_query.count(")")
                and raw_query.count("(") == (raw_query.count("AND")
                                             + raw_query.count("OR")
                                             + raw_query.count("NOT"))):
        raise ValueError('Invalid Query: Unmatched brackets or operators.')

    # get nodes
    split_query = re.split(r"(\(|\)|AND|OR|NOT)", raw_query)
    node_list = [s.strip() for s in split_query if s != '' and s != ' ']

    qStack = Stack()
    query_tree = BinaryTree('')
    qStack.push(query_tree)
    current_tree = query_tree

    if len(node_list) == 1: # query only contains a search term
        node = node_list[0]
        current_tree.setRootVal(node)

        # extract column specification and normalize search term.
        split_term = node.split(':')
        if len(split_term) == 1:  # no column specification
            current_tree.targetcolumn = None
            term = node
        else:
            current_tree.targetcolumn = split_term[0]
            term = ':'.join(split_term[1:])

        current_tree.normedterm = normalize_text([term], lang_model)[0]
    else:
        for node in node_list:
            if node == '(':
                current_tree.insertLeft('')
                qStack.push(current_tree)
                current_tree = current_tree.getLeftChild()
            elif node in ['AND', 'OR']:
                current_tree.setRootVal(node)
                current_tree.insertRight('')
                qStack.push(current_tree)
                current_tree = current_tree.getRightChild()
            elif node == 'NOT':
                current_tree = qStack.pop()
                current_tree.setRootVal(node)
                qStack.push(current_tree)
                current_tree = current_tree.getLeftChild()
            elif node == ')':
                current_tree = qStack.pop()
            elif node not in ['AND', 'OR', 'NOT', ')']: # leaf node
                current_tree.setRootVal(node)

                # extract column specification and normalize search term.
                split_term = node.split(':')
                if len(split_term) == 1: # no column specification
                    current_tree.targetcolumn = None
                    term = node
                else:
                    current_tree.targetcolumn = split_term[0]
                    term = ':'.join(split_term[1:])

                current_tree.normedterm = normalize_text([term], lang_model)[0]

                current_tree = qStack.pop()

    return query_tree

def get_vocabulary(query_tree):
    """Extracts the normalized search terms from the leaf nodes of a parsed
    query to construct the vocabulary for the text vectorization.

    Arguments
    ---------
    query_tree: pythonds.trees.BinaryTree
        The binary tree object representing a parsed search query. Each leaf
        node is a search term and internal nodes represent boolean operations.
        See parse_query() for details.

    Returns
    -------
    vocabulary: list
        List of strings representing unique normalized search terms.

    """
    def _getleafnodes(node):
        terms = []
        if node.isLeaf():
            return terms + [node.normedterm]
        elif node.leftChild and not node.rightChild:
            return terms + _getleafnodes(node.getLeftChild())
        elif node.rightChild and not node.leftChild:
            return terms + _getleafnodes(node.getRightChild())
        else:   # has two children
            return terms + _getleafnodes(node.getLeftChild()) \
                         + _getleafnodes(node.getRightChild())

    # extract terms from the leaf nodes of the query object.
    terms = _getleafnodes(query_tree)

    # remove duplicates.
    vocabulary = list(set(terms))

    return vocabulary

def get_vectorizer(vocabulary):
    """This function constructs a text vectorization object fit to a specified
     vocabulary of n-gram search terms, using the ScikitLearn package. This
     vectorizer uses a binary (present/not present) bag-of-words approach to
     transform a collection of documents into a document-term matrix.

    Arguments
    ---------
    vocabulary: list
       List of strings representing unique normalized search terms.

    Returns
    -------
    vectorizer: sklearn.feature_extraction.text.CountVectorizer
       Text vectorization object of ScikitLearn. It is fit to the vocabulary,
       which means that subsequent transformations of collections of documents
       will amount to searching the collections of documents for the presence
       of the terms in the vocabulary.

    """
    # find the n-gram range of the vocabulary.
    min_n = min([len(term.split()) for term in vocabulary])
    max_n = max([len(term.split()) for term in vocabulary])
    vectorizer = CountVectorizer(vocabulary=vocabulary,
                           ngram_range=(min_n, max_n),
                           binary=True)

    # init vocabulary_ attribute
    vectorizer.get_feature_names_out()
    return vectorizer

def search_data(data, vectorizer, lang_model):
    """Searches collections of documents for the presence of unique search
    terms. This is implemented through the transform() method of a
    text vectorization objects fit to a vocabulary of unique search terms.
    Treats each column of the input data as a collection of documents, and
    fills a dictionary with columnname -> vectorizedtextdata key/value pairs.

    Arguments
    ---------
    data: pandas.DataFrame
        Pandas object containing the tabular data.
    vectorizer: sklearn.feature_extraction.text.CountVectorizer
        Text vectorization object of ScikitLearn. It is fit to a vocabulary of
        search terms, which means that subsequent vectorization of a collection
        of documents amounts to searching the documents for the presence of
        search terms in the vocabulary.
    lang_model: spacy.Language
        Spacy language object passed to normalize_text() for lemmatization.

    Returns
    -------
    vectorized_data: dict
        Dictionary containing columnname -> vectorizedtextdata key/value pairs.
        The vectorizedtextdata consists of a binary document-term matrix
        representing the presence of each search term (column) in each
        document (row).

     """
    vectorized_data = {}
    for col in data.columns:
        column_data = data[col].apply(str)
        normed_data = normalize_text(column_data, lang_model)
        vectorized_data[col] = vectorizer.transform(normed_data)

    return vectorized_data

def select_subset(vectorized_data, query_tree, legend):
    """Recursively chains boolean operations between pandas Series while
    traversing a tree-structured query to obtain a boolean index representing
    the entries in the original dataframe which conform to the query.

    Arguments
    ---------
    vectorized_data: dict
        Dictionary of columnname -> vectorizedtextdata key/value pairs.
        The vectorizedtextdata consists of a binary document-term matrix
        representing the presence of each search term (column) in each
        document (row).
    query_tree: pythonds.trees.BinaryTree
        The binary tree object representing a parsed search query. Each leaf
        node is a search term and internal nodes represent boolean operations.
        See parse_query() for details.
    legend: dict
        A dictionary which maps the vocabulary containing search terms to the
        term indices in the document-term matrices of vectorized_data.
        (this is generally the vocabulary_ attribute of the CountVectorizer
        object obtained from get_vectorizer()).

    Returns
    -------
    pandas.Series
        A boolean index for the original dataframe, representing the outcome
        of the query.

     """
    if query_tree.isLeaf(): # we found a search term.
        term = query_tree.normedterm

        # get column key. (e.g 'abstract' in 'abstract:Mozambique')
        column_key = query_tree.targetcolumn

        if column_key:
            if column_key not in vectorized_data.keys():
                raise ValueError(f'Query contains invalid column key. '
                                 f'Column key: {column_key}')

            term_index = legend[term]
            # get column data.
            column_data = vectorized_data[column_key].toarray()
            return column_data[:,term_index].astype(bool)
        else:   # search term can be in any column.
            term_index = legend[term]
            # OR operation between all columns.
                # get nr of entries in the data
            N = list(vectorized_data.values())[0].shape[0]
            accumulator = np.zeros(N).astype(bool)
            for column_key in vectorized_data.keys():
                # get column data.
                column_data = vectorized_data[column_key].toarray()
                accumulator = (accumulator
                                | column_data[:,term_index].astype(bool))
            return accumulator
    elif query_tree.key == 'AND':
        return (select_subset(vectorized_data,
                              query_tree.getLeftChild(), legend)
                    & select_subset(vectorized_data,
                                    query_tree.getRightChild(), legend))
    elif query_tree.key == 'OR':
        return (select_subset(vectorized_data,
                              query_tree.getLeftChild(), legend)
                    | select_subset(vectorized_data,
                                    query_tree.getRightChild(), legend))
    elif query_tree.key == 'NOT':
        return ~ (select_subset(vectorized_data,
                                query_tree.getLeftChild(), legend))
    else:
        raise ValueError('Query was not parsed correctly.')

def label_subset(data, index, label):
    """Changes the labels of the entries in the data which conform to the
    query.

    Arguments
    ---------
    data: pandas.DataFrame
        Pandas object containing the tabular data.
    index: pandas.Series
        A boolean index for the original dataframe, representing the outcome
        of the query.
    label: str
        The label (include/exclude/unlabelled) to be given to queried entries.

    Returns
    -------
    labeled_data: pandas.DataFrame
        The tabular data with updated 'label_included' column.
        Unlabelled = -1. Relevant = 1, irrelevant = 0.
    """
    if 'label_included' not in data.columns:
        raise ValueError("Data does not contain 'label_included' column.")

    labeled_data = data.copy()
    if label == 'include':
        labeled_data.loc[index,'label_included'] = 1
    elif label == 'exclude':
        labeled_data.loc[index, 'label_included'] = 0
    elif label == 'unlabelled':
        labeled_data.loc[index, 'label_included'] = -1
    else:
        raise ValueError("Label not recognized. include/exclude/unlabelled")

    return labeled_data


if __name__ == '__main__':
    # English language model (to be installed separately on CLI, see line 7)
    lang_model = en_core_web_sm.load()

    filepath = input('what is the filepath to your dataset? (e.g data.csv)\n'
                     'The script handles csv or excel files.\n')

    raw_query = input('What is your query? See README for details.\n')

    label = input("What label do you want to apply to the subset? "
                  "  include/exclude/unlabelled\n")

    # load data
    data = load_data(filepath, columns=['title',
                                        'abstract',
                                        'label_included'])

    # parse the query
    query_tree = parse_query(raw_query, lang_model)

    # construct the text vectorizer
    vocabulary = get_vocabulary(query_tree)
    vectorizer = get_vectorizer(vocabulary)
    feature_legend = vectorizer.vocabulary_

    # vectorize the data according to the restricted vocabulary
    vectorized_data = search_data(data, vectorizer, lang_model)

    selection = select_subset(vectorized_data,
                              query_tree,
                              feature_legend)

    labeled_data = label_subset(data,selection,label)

    labeled_data.to_csv('labeled_' + filepath,
                        index=False,
                        encoding='utf-8',
                        sep=';')