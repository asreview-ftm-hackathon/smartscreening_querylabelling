from main import *
from pythonds.basic import Stack
from pythonds.trees import BinaryTree

# keywords in test data
# x-index           keyword
# 0
# 1                 zalm (in abstract)
# 2                 help (in type)
# 3                 life (in type)
# 4                 help me (in type)
# 7                 ZALM + evi (in abstract)


test_queries = ['(abstract:zalm AND (abstract:evi OR (NOT type:help)))',
                '((abstract:evi OR (NOT type:help)) AND abstract:zalm)',
                '(abstract:zalm pasta AND (abstract:evi OR (NOT type:help me)))',
                '(abstract:zalm AND (evi OR (type:do not help me)))',
                'abstract:zalm pasta',
                '(abstract:evi AND type:help)',
                '(abstract:42222*& OR type:life)'
                ]

test_trees = [parse_query(x) for x in test_queries]
test_vocabs = [get_vocabulary(x) for x in test_trees]
test_vects = [get_vectorizer(x) for x in test_vocabs]


language_model = spacy.load('nl_core_news_sm')
data = load_data('test_data.csv', columns=['type','title','abstract'])
test_query = '(abstract:zalm AND (evi OR type:do not help me))'
test_tree = parse_query(test_query)
test_vocab = get_vocabulary(test_tree, language_model)
test_vect = get_vectorizer(test_vocab)
test_vec_data = vectorize_data(data, test_vect, language_model)
output = select_subset(test_vec_data, test_tree, test_vect.vocabulary_, language_model)

def test(query):
    pt = buildParseTree(query)
    print(pt.postorder())  # defined and explained in the next section

    output = pt
    success = True
    return success, output



for query in test_queries:
    success, output = test(query)
    if not success:
        print('Test Failed')
        print(f'Query: {query}')
        print(f'Output: {output}')