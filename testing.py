from main import *


test_queries = ['(abstract:zalm AND (abstract:evi OR (NOT type:help)))',
                '((abstract:evi OR (NOT type:help)) AND abstract:zalm)',
                '(abstract:zalm pasta AND (abstract:evi OR (NOT type:help me)))',
                '(abstract:zalm pasta AND (evi OR (NOT type:do not help me)))',
                '(abstract:zalm pasta)',
                '(abstract:evi AND type:help)',
                '(abstract:42222*& OR type:life)'
                ]


def test(query):
    output = query
    success = True
    return success, output

for query in test_queries:
    success, output = test(query)
    if not success:
        print('Test Failed')
        print(f'Query: {query}')
        print(f'Output: {output}')








