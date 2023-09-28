import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ngram import NGram 

def testSurprisal():

    model = NGram(vocab_file='vocab.txt')

    trigrams = {'1gram': {'the': 4, '<unk>': 6, 'of': 1, 'book': 1, 'was': 1, 'and': 2, '</s>': 3, 'cat': 1, 'sleeps': 1, '.': 1, 'man': 1, '!': 1}, '2gram': {'<s>': {'the': 2, 'and': 1}, 'the': {'<unk>': 1, 'book': 1, 'cat': 1, 'man': 1}, '<unk>': {'of': 1, '<unk>': 1, 'was': 1, 'and': 1, '</s>': 1, '!': 1}, 'of': {'the': 1}, 'book': {'<unk>': 1}, 'was': {'<unk>': 1}, 'and': {'<unk>': 1, 'the': 1}, 'cat': {'sleeps': 1}, 'sleeps': {'.': 1}, '.': {'</s>': 1}, 'man': {'<unk>': 1}, '!': {'</s>': 1}}, '3gram': {'<s> <s>': {'the': 2, 'and': 1}, '<s> the': {'<unk>': 1, 'cat': 1}, 'the <unk>': {'of': 1}, '<unk> of': {'the': 1}, 'of the': {'book': 1}, 'the book': {'<unk>': 1}, 'book <unk>': {'<unk>': 1}, '<unk> <unk>': {'was': 1}, '<unk> was': {'<unk>': 1}, 'was <unk>': {'and': 1}, '<unk> and': {'<unk>': 1}, 'and <unk>': {'</s>': 1}, 'the cat': {'sleeps': 1}, 'cat sleeps': {'.': 1}, 'sleeps .': {'</s>': 1}, '<s> and': {'the': 1}, 'and the': {'man': 1}, 'the man': {'<unk>': 1}, 'man <unk>': {'!': 1}, '<unk> !': {'</s>': 1}}}

    model.ngrams = trigrams.copy()

    try: 
        model.surprisal(('the', 'cat', 'is'), {})
    except NotImplementedError:
        sys.stderr.write('You need to implement the surprisal function\n')
        sys.exit(1)



    assert round(model.surprisal(('the', 'cat', 'is'), {}), 5) == 33.21928
    assert round(model.surprisal(('the', 'cat', 'is'), {'k': 1}), 5) == 11.24020
    assert round(model.surprisal(('the', 'cat', 'was'), {'lambdas': [0.2, 0.2,
                                                                    0.8]}), 5) == 4.84549
