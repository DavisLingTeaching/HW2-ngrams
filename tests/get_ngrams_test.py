import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ngram import NGram 

def testGetNGrams():

    model = NGram(vocab_file='vocab.txt')

    try: 
        model.get_ngrams(' ', 3)
    except NotImplementedError:
        sys.stderr.write('You need to implement the get_ngrams function\n')
        sys.exit(1)

    data = "the kjdflkajs of the book's plot was weird and gothic"
    assert model.get_ngrams(data, 3) == [('<s>', '<s>', 'the'), ('<s>', 'the',
                                                                 '<unk>'),
                                         ('the', '<unk>', 'of'), ('<unk>', 'of',
                                                                  'the'), ('of',
                                                                           'the',
                                                                           'book'),
                                         ('the', 'book', '<unk>'), ('book',
                                                                    '<unk>',
                                                                    '<unk>'),
                                         ('<unk>', '<unk>', 'was'), ('<unk>',
                                                                     'was',
                                                                     '<unk>'),
                                         ('was', '<unk>', 'and'), ('<unk>',
                                                                   'and',
                                                                   '<unk>'),
                                         ('and', '<unk>', '</s>')] 

    data = "the cat sleeps. And the man snores!"
    assert model.get_ngrams(data, 2) == [('<s>', 'the'), ('the', 'cat'), ('cat',
                                                                          'sleeps'),
                                         ('sleeps', '.'), ('.', '</s>'), ('<s>',
                                                                          'and'),
                                         ('and', 'the'), ('the', 'man'), ('man',
                                                                          '<unk>'),
                                         ('<unk>', '!'), ('!', '</s>')]
