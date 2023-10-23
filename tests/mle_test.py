import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ngram import NGram 

def compareGrams(data: str, 
                 goldNGrams: dict, studentNGrams: dict) -> None:

    print(f"Testing mle on '{data}'")
    for gram in goldNGrams:
        assert gram in studentNGrams, f"Missing {gram}"
        if gram == '1gram':
            for target in goldNGrams[gram]:
                assert target in studentNGrams[gram], f"Missing {target} from "\
                                    f"{gram}"
                count = goldNGrams[gram][target]
                assert count == studentNGrams[gram][target], f"Count is off "\
                                f"for {target}. Should be {count}"

        else:
            for context in goldNGrams[gram]:
                assert context in studentNGrams[gram], f"Missing {context} from "\
                                    f"{gram}"
                for target in goldNGrams[gram][context]:
                    assert target in studentNGrams[gram][context], f"Missing "\
                            "{target} from {gram} with context {context}"
                    count = goldNGrams[gram][context][target]
                    assert count == studentNGrams[gram][context][target], \
                            f"Count is off for {context} {target}. Should be {count}"

def testMLE():

    model = NGram(ngram_size = 3, vocab_file='vocab.txt')

    try: 
        model.mle(' ', {})
    except NotImplementedError:
        sys.stderr.write('You need to implement the mle function\n')
        sys.exit(1)

    data = "the kjdflkajs of the book's plot was weird and gothic"

    studentNgrams_1 = model.mle(data, {})
    print(studentNgrams_1)

    goldNGrams_1 = {'1gram': {'the': 2, '<unk>': 5, 'of': 1, 'book': 1, 'was': 1, 'and': 1, '</s>': 1}, '2gram': {'<s>': {'the': 1}, 'the': {'<unk>': 1, 'book': 1}, '<unk>': {'of': 1, '<unk>': 1, 'was': 1, 'and': 1, '</s>': 1}, 'of': {'the': 1}, 'book': {'<unk>': 1}, 'was': {'<unk>': 1}, 'and': {'<unk>': 1}}, '3gram': {'<s> <s>': {'the': 1}, '<s> the': {'<unk>': 1}, 'the <unk>': {'of': 1}, '<unk> of': {'the': 1}, 'of the': {'book': 1}, 'the book': {'<unk>': 1}, 'book <unk>': {'<unk>': 1}, '<unk> <unk>': {'was': 1}, '<unk> was': {'<unk>': 1}, 'was <unk>': {'and': 1}, '<unk> and': {'<unk>': 1}, 'and <unk>': {'</s>': 1}}}

    compareGrams(data, goldNGrams_1, studentNgrams_1)

    data = "the cat sleeps. And the man snores!"

    studentNgrams_2 = model.mle(data, goldNGrams_1)

    goldNGrams_2 = {'1gram': {'the': 4, '<unk>': 6, 'of': 1, 'book': 1, 'was': 1, 'and': 2, '</s>': 3, 'cat': 1, 'sleeps': 1, '.': 1, 'man': 1, '!': 1}, '2gram': {'<s>': {'the': 2, 'and': 1}, 'the': {'<unk>': 1, 'book': 1, 'cat': 1, 'man': 1}, '<unk>': {'of': 1, '<unk>': 1, 'was': 1, 'and': 1, '</s>': 1, '!': 1}, 'of': {'the': 1}, 'book': {'<unk>': 1}, 'was': {'<unk>': 1}, 'and': {'<unk>': 1, 'the': 1}, 'cat': {'sleeps': 1}, 'sleeps': {'.': 1}, '.': {'</s>': 1}, 'man': {'<unk>': 1}, '!': {'</s>': 1}}, '3gram': {'<s> <s>': {'the': 2, 'and': 1}, '<s> the': {'<unk>': 1, 'cat': 1}, 'the <unk>': {'of': 1}, '<unk> of': {'the': 1}, 'of the': {'book': 1}, 'the book': {'<unk>': 1}, 'book <unk>': {'<unk>': 1}, '<unk> <unk>': {'was': 1}, '<unk> was': {'<unk>': 1}, 'was <unk>': {'and': 1}, '<unk> and': {'<unk>': 1}, 'and <unk>': {'</s>': 1}, 'the cat': {'sleeps': 1}, 'cat sleeps': {'.': 1}, 'sleeps .': {'</s>': 1}, '<s> and': {'the': 1}, 'and the': {'man': 1}, 'the man': {'<unk>': 1}, 'man <unk>': {'!': 1}, '<unk> !': {'</s>': 1}}}

    compareGrams(data, goldNGrams_2, studentNgrams_2)

