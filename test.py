from tests.get_ngrams_test import testGetNGrams
from tests.mle_test import testMLE
from tests.addK_prob_test import testAddKProb
from tests.interpolation_prob_test import testInterpolationProb
from tests.surprisal_test import testSurprisal
from tests.entropy_test import testEntropy
from tests.perplexity_test import testPerplexity
from tests.generate_test import testGenerate
import argparse

parser = argparse.ArgumentParser(prog='test.py', 
                                 description='Test your tokenizer')


parser.add_argument('--test',
                    default='all',
                    nargs='?',
                    choices=['ngrams', 'mle', 'addK', 
                             'interpolate', 'surprisal', 
                             'entropy', 'perplexity', 
                             'generate', 'all'],
                    help='run test of ngrams, mle, addK, interpolate, '\
                    'surprisal, entropy, perplexity, generate, or all ' 
                    '(default: all)')

args = parser.parse_args()

if args.test == 'ngrams' or args.test == 'all':
    print('Testing get_ngrams()...')
    testGetNGrams()
    print('Passed!')

if args.test == 'mle' or args.test == 'all':
    print('Testing mle()...')
    testMLE()
    print('Passed!')

if args.test == 'addK' or args.test == 'all':
    print('Testing addK_prob()...')
    testAddKProb()
    print('Passed!')

if args.test == 'interpolate' or args.test == 'all':
    print('Testing interpolation_prob()...')
    testInterpolationProb()
    print('Passed!')

if args.test == 'surprisal' or args.test == 'all':
    print('Testing surprisal()...')
    testSurprisal()
    print('Passed!')

if args.test == 'entropy' or args.test == 'all':
    print('Testing entropy()...')
    testEntropy()
    print('Passed!')

if args.test == 'perplexity' or args.test == 'all':
    print('Testing perplexity()...')
    testPerplexity()
    print('Passed!')

if args.test == 'generate' or args.test == 'all':
    print('Testing generate()...')
    testGenerate()
    print('Passed!')
