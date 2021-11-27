from main import *


test_queries = ['()','()','()']


def test():
    output = None
    success = False
    return success, output

for query in test_queries:
    success, output = test(query)
    if not success:
        print('Test Failed')
        print(f'Query: {query}')
        print(f'Output: {output}')








