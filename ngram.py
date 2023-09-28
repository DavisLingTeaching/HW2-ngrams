import sys
import random
import math
from nltk import sent_tokenize, word_tokenize
import glob
from typing import Union

class NGram:
    """
    A class for n-gram language modeling.  

        # Load vocab
        self.vocab_file = vocab_file
        self.vocab  = self.load_vocab(self.vocab_file)

        # Set special tokens
        self.bos = bos_token
        self.eos = eos_token
        self.unk = unk_token

        # Set up ngrams dictionary for later
        self.ngrams = {}

    Attributes:
        ngram_size (int): Size of ngrams (e.g., 3 for trigrams)
        vocab_file (str): Filename for vocab (plain-txt file with one word per
                            line)
        vocab (set): Set containing words in the vocab
        bos (str): Token to mark beginning of sentence
        eos (str): Token to mark ending of sentence
        unk (str): Token to mark out-of-vocabulary words 
        ngrams (dict): Dictionary containing ngrams. Details on its structure is
                        given in mle(...)
    """

    def __init__(self, 
                ngram_size = 3, 
                vocab_file = 'vocab.txt', 
                bos_token = '<s>', 
                eos_token = '</s>', 
                unk_token = '<unk>'):

        # Set maximum ngram size
        self.ngram_size = ngram_size

        # Load vocab
        self.vocab_file = vocab_file
        self.vocab  = self.load_vocab(self.vocab_file)
        # Set special tokens
        self.bos = bos_token
        self.eos = eos_token
        self.unk = unk_token

        # Set up ngrams dictionary for later
        self.ngrams = {}

    @classmethod
    def load_pretrained(cls, fname:str) -> object:
        """ Creates an instance of NGram from a pre-trained model.
        Pretrained models are in plain-txt format. The first line must be the
        ngram size, the next lines the bos, eos, unk, and vocab filename
        respectively. After that the lines have the ngram followed by its count.

        Args:
            fname (str): Filename of pretrained model 

        Returns:
            NGram: An instance of the NGram class

        For example, a file named 'model.txt' containing
        3
        <s>
        </s>
        <unk>
        vocab.txt
        the 20
        ...
        the cat 10
        ...
        the cat runs 1
        ...

        Would yield a trigram model when loaded as follows
        >>> ngram = NGram.load_pretrained('model.txt')
        """

        # Skip __init__
        obj = cls.__new__(cls)
        super(NGram, obj).__init__()

        cls.ngrams = {}
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                # Capture those special attributes
                if idx == 0:
                    obj.ngram_size = int(line)
                    continue
                if idx == 1:
                    obj.bos = line
                    continue
                if idx == 2:
                    obj.eos = line
                    continue
                if idx == 3:
                    obj.unk = line
                    continue
                if idx == 4:
                    obj.vocab = obj.load_vocab(line.strip())
                    continue

                # Get ngram and count and add to obj
                line = line.split()
                gram = f"{len(line)-1}gram"
                if gram not in obj.ngrams:
                    obj.ngrams[gram] = {}
                if gram == '1gram':
                    obj.ngrams[gram][line[0]] = float(line[1])
                else:
                    count = line[-1]
                    target = line[-2]
                    context = ' '.join(line[:-2])
                    if context not in obj.ngrams[gram]:
                        obj.ngrams[gram][context] = {}
                    obj.ngrams[gram][context][target] = float(count)
        return obj

    def save(self, fname:str) -> None:
        """ Saves the ngram model to fname as a plain-txt file.

        Args:
            fname (str): Filename to use for saving

        For example, if ngram_size == 2, bos == <s>, eos == </s>, 
        unk == <unk>, vocab_file == 'vocab.txt', and
        ngrams = {'the': 100, ..., 'the cat': 50}, the resultant 
        file would be
        2
        <s>
        </s>
        <unk>
        vocab.txt
        the 100
        ...
        the cat 50
        """
        with open(fname, 'w') as f:
            f.write(f"{self.ngram_size}\n")
            f.write(f"{self.bos}\n")
            f.write(f"{self.eos}\n")
            f.write(f"{self.unk}\n")
            f.write(f"{self.vocab_file}\n")
            for gram in self.ngrams: 
                if gram == '1gram':
                    for target in self.ngrams[gram]:
                        f.write(f"{target} {self.ngrams[gram][target]}\n")
                else:
                    for context in self.ngrams[gram]:
                        for target in self.ngrams[gram][context]:
                            count = self.ngrams[gram][context][target]
                            f.write(f"{context} {target} {count}\n")

    def load_vocab(self, fname: str) -> set:
        """
        Loads plain txt file vocab and returns a set.

        Args:
            fname (str): Name of vocab file

        Returns:
            set: Vocab items as set

        For example, 
            >>> model = NGram()
            >>> model.load_vocab('cat_vocab.txt')
            {'other', 'was', 'and', ',', 'is', 'again', 'that', 'know', 'as',
             'an', 'one', 'cat', 'jumps', 'the', '.', 'unhappy', '<unk>',
             'over'}
        """
        vocab = set([])
        with open(fname, 'r') as f:
            for line in f:
                vocab.add(line.strip())
        return vocab

    #TODO: 25 points
    def get_ngrams(self, data: str, size: int) -> list[tuple[str]]:
        """
        Returns the ngrams with n=size from a string. Sentences are extracted from
        the data using sent_tokenize, words from a sentence with word_tokenize,
        out of vocabulary words determined relative to self.vocab, and the
        appropriate number of bos and eos symbols affixed to each sentence. 
        
        Note: you should lowercase data!

        Args:
            data (str): string representing lines from a file
            size (int): size of ngrams (e.g., 3grams)

        Returns:
            list[tuple[str]]: List of ngrams as tuples

        For example, 

        >>> model = NGram(vocab_file='tiny_data/cat_vocab.txt')
        >>> model.get_ngrams('cat cat always cats', 1)
        [('cat',), ('cat',), ('<unk>',), ('<unk>',), ('</s>',)]
        >>> model.get_ngrams('hi. a cat that jumps is an other cat.', 2)
        [('<s>', '<unk>'), ('<unk>', '.'), ('.', '</s>'), ('<s>', '<unk>'),
         ('<unk>', 'cat'), ('cat', 'that'), ('that', 'jumps'), ('jumps', 'is'),
         ('is', 'an'), ('an', 'other'), ('other', 'cat'), ('cat', '.'), 
         ('.', '</s>')]
        >>> model.get_ngrams('That is the sentence about a CAT again', 3)
        [('<s>', '<s>', 'that'), ('<s>', 'that', 'is'), ('that', 'is', 'the'), 
         ('is', 'the', '<unk>'), ('the', '<unk>', '<unk>'), 
         ('<unk>', '<unk>', '<unk>'), ('<unk>', '<unk>', 'cat'), 
         ('<unk>', 'cat', 'again'), ('cat', 'again', '</s>')]
        """
        raise NotImplementedError

    #TODO: 25 points
    def mle(self, data: str, ngrams: dict) -> dict:
        """Implements part of maximum likelihood estimation. In particular,
        counts of ngrams in data are returned as a dictionary. If the parameter
        ngrams is passed an existing dictionary, the counts are added to that
        dictionary. 

        The returned dictionary should contain counts of ngrams of size
        self.ngram_size and all lower order ngrams. The core structure of the
        returned dictionary is keys for each ngram (e.g., 1gram, 2gram, 3gram,
        etc.) which map to dictionaries. These sub dictionaries (for n greater
        than 1) contain keys mapping contexts to dictionaries mapping targets to
        counts. For the bigram 'the cat', the context is 'the' and the target is
        'cat'. For example, if data contains one sentence 'the cat jumps the
        cat.', and
        ngram_size == 2, the returned dictionary would be 
        {'1gram': {'the': 2, 'cat': 2, 'jumps': 1, '.': 1, '</s>': 1}, 
        '2gram': {'<s>': {'the': 1}, 'the': {'cat': 2}, 
                'cat': {'jumps': 1, '.': 1}, 
                'jumps': {'the': 1}, '.': {'</s>': 1}}}

        Args:
            data (str): String containing data to count ngrams in 
            ngrams (dict): A dictionary that can contain prior counts

        Returns:
            dict: A dictionary containing counts of ngrams

        For example, 
        >>> with open('tiny_data/cat.txt', 'r') as f:
        ...     data = f.read()
        >>> model = NGram(ngram_size=1, vocab_file='tiny_data/cat_vocab.txt')
        >>> ngrams = model.mle('cat.txt', {})
        >>> ngrams
        {'1gram': {'the': 3, 'cat': 4, 'jumps': 2, 'over': 2, 'other': 2, '.':
                   2, '</s>': 2, 'was': 1, 'unhappy': 2, ',': 2, 'and': 1, 'as':
                   1, '<unk>': 3, 'know': 1, 'an': 1, 'is': 1, 'one': 1, 'that':
                   1, 'again': 1}}
        >>> model.mle('cat.txt', ngrams)
        {'1gram': {'the': 6, 'cat': 8, 'jumps': 4, 'over': 4, 'other': 4, '.':
                   4, '</s>': 4, 'was': 2, 'unhappy': 4, ',': 4, 'and': 2, 'as':
                   2, '<unk>': 6, 'know': 2, 'an': 2, 'is': 2, 'one': 2, 'that':
                   2, 'again': 2}}
        >>> model = NGram(ngram_size=2, vocab_file='tiny_data/cat_vocab.txt')
        >>> model.mle('cat.txt', {})
        {'1gram': {'the': 3, 'cat': 4, 'jumps': 2, 'over': 2, 'other': 2, '.':
                   2, '</s>': 2, 'was': 1, 'unhappy': 2, ',': 2, 'and': 1, 'as':
                   1, '<unk>': 3, 'know': 1, 'an': 1, 'is': 1, 'one': 1, 'that':
                   1, 'again': 1}, 
         '2gram': {'<s>': {'the': 2}, 'the': {'cat':
                   1, 'other': 2}, 'cat': {'jumps': 1, '.': 1, 'was': 1, 'is':
                   1}, 'jumps': {'over': 2}, 'over': {'the': 1, '<unk>': 1},
                   'other': {'cat': 2}, '.': {'</s>': 2}, 'was': {'unhappy': 1},
                   'unhappy': {',': 1, 'cat': 1}, ',': {'and': 1, 'an': 1},
                   'and': {'as': 1}, 'as': {'<unk>': 1}, '<unk>': {'know': 1,
                   'jumps': 1, 'again': 1}, 'know': {',': 1}, 'an': {'unhappy':
                   1}, 'is': {'one': 1}, 'one': {'that': 1}, 'that': {'<unk>':
                   1}, 'again': {'.': 1}}}
        """
        raise NotImplementedError

    #TODO: 25 points
    def addK_prob(self, ngram:tuple[str], k:int) -> float:
        """ Return the probability of the ngram with add-k smoothing.
        Recall that add-k smoothing equation is given as 

        P_K(target | context) = (Count(context, target) + k) /
                                (Count(context) + k*|vocab|)

        Args:
            ngram (tuple[str]): the relevant ngram (e.g., ('the',),
                                    ('cats','are'))
            k (int): the specific k for add-k smoothing

        Returns:
            float: probability of ngram with add-k smoothing.

        For example, 
        >>> with open('tiny_data/cat.txt', 'r') as f:
        ...     data = f.read()
        >>> model = NGram(ngram_size=1, vocab_file='tiny_data/cat_vocab.txt')
        >>> model.ngrams = model.mle(data, {})
        >>> model.addK_prob(('cat',), 1)
        0.09433962264150944
        >>> 
        >>> model = NGram(ngram_size=3, vocab_file='tiny_data/cat_vocab.txt')
        >>> model.ngrams = model.mle(data, {})
        >>> model.addK_prob(('the', 'cat', 'is'), 0.5)
        0.045454545454545456
        >>> model.addK_prob(('the', 'cat', 'is'), 0)
        0.0
        """
        raise NotImplementedError

    #TODO: 25 points
    def interpolation_prob(self, ngram:tuple[str], lambdas:list[float]) -> float:
        """ Return the probability of the ngram with interpolation smoothing.
        Recall that interpolation smoothing is a weighted combination of ngram
        probabilities bottoming out in unigram model (e.g., trigram model,
        interpolates trigram probabality, bigram probability, unigram
        probability). The weighthing is determined by the lambdas (which must
        sum to one!). Note, interpolation_prob should operate on ngram input of
        ngram_size. 

        Args:
            ngram (tuple[str]): the relevant ngram (e.g., ('the',),
                                    ('cats','are'))
            lambdas (list[float]): the specific lambdas for interpolation
                                smoothing. Note that the index 0 picks out the
                                interpolation for the ngram_size (that is the
                                lambdas appy in order from largest to smallest
                                ngram)

        Returns:
            float: probability of ngram with interpolation smoothing

        For example, 
        >>> with open('tiny_data/cat.txt', 'r') as f:
        ...     data = f.read()
        >>> model = NGram(ngram_size=2, vocab_file='tiny_data/cat_vocab.txt')
        >>> model.ngrams = model.mle(data, {})
        >>> model.interpolation_prob(('the', 'cat'), [0.8, 0.2])
        0.2909090909090909
        >>> 
        >>> model = NGram(ngram_size=3, vocab_file='tiny_data/cat_vocab.txt')
        >>> model.ngrams = model.mle(data, {})
        >>> model.interpolation_prob(('the', 'cat', 'is'), [0.7, 0.2, 0.1])
        0.05303030303030303
        >>> model.interpolation_prob(('the', 'cat', 'is'), [0.7, 0.3, 0.9])
        AssertionError: [0.7, 0.3, 0.9] do not sum to 1
        >>> model.interpolation_prob(('the', 'cat'), [0.7, 0.2, 0.1])
        AssertionError: ('the', 'cat') is not a 3gram!
        """

        raise NotImplementedError

    def prob(self, ngram:tuple[str], params:dict={}) -> float:
        """Function which returns the probability of a ngram. 
        Parameters are passed to control with probability is returned. If params
        is empty (default), then the unsmoothed probability of the ngram is
        returned. If params contains a key 'k', then the add-k smoothed
        probability (with k the value of 'k') is returned. Finally, 
        if params contains a key 'lambdas', then the interpolation smoothed
        probability is returned (with lambdas the values of 'lambdas'). 

        Args:
            ngram (tuple[str]): relevant ngram (e.g., ('the',))
            params (dict): parameters for different probability estimates
                        (default: {})
        Returns:
            float: probability of the ngram

        For example, 
        >>> with open('tiny_data/cat.txt', 'r') as f:
        ...     data = f.read()
        >>> model = NGram(ngram_size=3, vocab_file='tiny_data/cat_vocab.txt')
        >>> model.ngrams = model.mle(data, {})
        >>> model.prob(('the', 'cat', 'is'), {})
        0.0
        >>> model.prob(('the', 'cat', 'is'), {'k': 3})
        0.04918032786885246
        >>> model.prob(('the', 'cat', 'is'), {'lambdas': [0.7, 0.1, 0.2]})
        0.031060606060606063
        """

        if not params:
            return self.addK_prob(ngram, 0)
        elif 'k' in params:
            return self.addK_prob(ngram, params['k'])
        elif 'lambdas' in params:
            return self.interpolation_prob(ngram, params['lambdas'])
        else:
            sys.stderr.write(f"Unrecognized params: {params}\n")
            sys.exit(1)

    #TODO: 10 points
    def surprisal(self, ngram:tuple[str], params:dict={}, 
                   epsilon:float=0.0000000001) -> float:
        """Function which returns the surprisal of a ngram. 
        Parameters are passed to control with probability is used. If params
        is empty (default), then the unsmoothed probability of the ngram is
        calculated. If params contains a key 'k', then the add-k smoothed
        probability (with k the value of 'k') is calculated. Finally, 
        if params contains a key 'lambdas', then the interpolation smoothed
        probability is calculated (with lambdas the values of 'lambdas'). 

        Surprisal is the negative log base 2 of the probability of the ngram.

        Args:
            ngram (tuple[str]): relevant ngram (e.g., ('the',))
            params (dict): parameters for different probability estimates
                        (default {})
            epsilon (float): if the probability is 0, use epsilon to avoid erorr
                            with log2 (default 0.0000000001)

        Returns:
            float: surprisal of the ngram

        For example, 
        >>> with open('tiny_data/cat.txt', 'r') as f:
        ...     data = f.read()
        >>> model = NGram(ngram_size=3, vocab_file='tiny_data/cat_vocab.txt')
        >>> model.ngrams = model.mle(data, {})
        >>> model.surprisal(('the', 'cat', 'is'), {})
        33.219280948873624
        >>> model.surprisal(('the', 'cat', 'is'), {'k': 3})
        4.34577483684173
        >>> model.surprisal(('the', 'cat', 'is'), {'lambdas': [0.7, 0.1, 0.2]})
        5.008770209627732
        """
        raise NotImplementedError

    #TODO: 10 points
    def perplexity(self, data:str, params:dict={}) -> float:
        """ Returns the perplexity of data using probabilies (and smoothing
        params). See self.prob(...) for information about params.

        Args:
            data (str): data to calculate perplexity of
            params (dict): smoothing parameters (default {})
        
        Returns:
            float: perplexity

        For example, 
        >>> with open('tiny_data/cat.txt', 'r') as f:
        ...     data = f.read()
        >>> model = NGram(ngram_size=3, vocab_file='tiny_data/cat_vocab.txt')
        >>> model.ngrams = model.mle(data, {})
        >>> model.perplexity('the dog jumps sometimes over the cat.', {'k':1})
        17.107422701729504
        """
        raise NotImplementedError

    #TODO: 10 points
    def entropy(self, context:tuple[str], params:dict={}, 
                   epsilon:float=0.0000000001) -> float:
        """ Returns the entropy, recall entropy is about information gained by
        potential next words (hence the use of context in the function
        definition). 

        Args:
            context (tuple[str]): context for entropy calculation
            params (dict): parameters for smoothing (see self.prob(...))
            epsilon (float): if the probability is 0, use epsilon to avoid erorr
                            with log2 (default 0.0000000001)

        Returns:
            float: Entropy

        For example, 
        >>> with open('tiny_data/cat.txt', 'r') as f:
        ...     data = f.read()
        >>> model = NGram(ngram_size=3, vocab_file='tiny_data/cat_vocab.txt')
        >>> model.ngrams = model.mle(data, {})
        >>> model.entropy(('the', 'dog'), {'k':1})
        4.1058316901429945
        >>> model.entropy((), {'k':1})
        4.131283219768024
        """
        raise NotImplementedError

    #TODO: 10 points
    def byWordMetrics(self, data:str, params:dict={}) -> None:
        """Prints the by-word information theoretic measures using smoothed
        probabilities. Note: You should use self.ngram_size to generate ngrams
        (that is, we are using the maximum context window)!

        Args:
            data (str): data to calculate by-word information theoretic measures
            params (dict): smoothing parameters (see self.prob(...))

        For example, 
        >>> with open('tiny_data/cat.txt', 'r') as f:
        ...     data = f.read()
        >>> model = NGram(ngram_size=3, vocab_file='tiny_data/cat_vocab.txt')
        >>> model.ngrams = model.mle(data, {})
        >>> model.byWordMetrics('the dog jumps over the cat', {'k': 1})
        word	surprisal			    entropy				entropy-reduction
        the	    2.8744691179161412		4.04059893132817		0
        <unk>	4.459431618637297		4.074911999608328		0
        jumps	4.321928094887363		4.1058316901429945		0
        over	3.3923174227787602		4.08792135502739		0.017910335115604248
        the	    3.4594316186372973		4.074911999608328		0.013009355419062452
        cat	    4.392317422778761		4.08792135502739		0
        </s>	4.392317422778761		4.08792135502739		0.0
        """
        raise NotImplementedError

    def train(self, directory:str) -> None:
        """Trains a ngram model by applying mle to all the txt files in a
        directory

        Args:
            directory (str): path to relevant directory

        For example, 
        >>> ngram = NGram(ngram_size=2, vocab_file='./tiny_data/cat_vocab.txt')
        >>> ngram.train('./tiny_data')
        >>> ngram.ngrams
        {'1gram': {'the': 4, 'cat': 5, 'jumps': 3, 'over': 3, '.': 3, '</s>': 4,
                   'other': 3, 'was': 2, 'unhappy': 3, ',': 3, 'and': 2, 
                   'as': 2, '<unk>': 6, 'know': 2, 'an': 2, 'is': 2, 'one': 2, 
                   'that': 2, 'again': 2}, 
         '2gram': {'<s>': {'the': 3, 'other': 1}, 'the': {'cat': 2, 'other': 2}, 
                   'cat': {'jumps': 2, '.': 1, 'was': 1, 'is': 1}, 
                   'jumps': {'over': 3}, 'over': {'.': 1, 'the': 1, '<unk>': 1}, 
                   '.': {'</s>': 3}, 'other': {'was': 1, 'cat': 2}, 
                   'was': {'unhappy': 2}, 'unhappy': {',': 2, 'cat': 1}, 
                   ',': {'and': 2, 'an': 1}, 'and': {'as': 2}, 
                   'as': {'<unk>': 2}, 
                   '<unk>': {'<unk>': 2, 'know': 2, 'jumps': 1, 'again': 1}, 
                   'know': {'an': 1, ',': 1}, 'an': {'is': 1, 'unhappy': 1}, 
                   'is': {'one': 2}, 'one': {'that': 2}, 
                   'that': {'again': 1, '<unk>': 1}, 
                   'again': {'</s>': 1, '.': 1}}}
        """

        fnames = glob.glob(f'{directory}/*.txt')
        ngrams = {}
        for fname in fnames:
            with open(fname, 'r') as f:
                data = f.read().replace('\n', ' ')
                ngrams = self.mle(data, ngrams)
        self.ngrams = ngrams

    def test(self, fname:str, evalType:str='ppl', 
             params:dict={}) -> Union[None, float]:
        """Evaluates a ngram language model on data found in the file name fname
        using smoothed probabilities (determined by params). 

        Args:
            fname (str): name of file to test model on 
            evalType (str): either 'ppl' (should calculate perplexity) or
                            'byWordMetrics' (should use self.byWordMetrics(...))
            params (dict): smoothing params (see self.prob(...); default {})

        Returns:
            float | None: Either return perplexity of data in file (float), or
                            prints by-word metrics (None)

        For example, 
        >>> ngram = NGram(ngram_size=3, vocab_file='./tiny_data/cat_vocab.txt')
        >>> ngram.train('./tiny_data')
        >>> ngram.test('./test_data/test.txt', 'ppl', {'k': 40})
        19.81665873730638
        >>> ngram.test('./test_data/test.txt', 'byWordMetrics', {'k': 40})
        ...
        the	    4.325530331567558		4.106150813038105		0.0
        <unk>	4.321928094887363		4.1058316901429945		0.0003191228951102687
        and	    4.321928094887363		4.1058316901429945		0.0
        <unk>	4.321928094887363		4.1058316901429945		0.0
        <unk>	4.321928094887363		4.1058316901429945		0.0
        <unk>	4.289906421836837		4.106150813038105		0
        and	    4.325530331567558		4.106150813038105		0.0
        <unk>	4.321928094887363		4.1058316901429945		0.0003191228951102687
        .	    4.321928094887363		4.1058316901429945		0.0
        </s>	4.321928094887363		4.1058316901429945		0.0
        """
        with open(fname, 'r') as f:
            data = f.read().replace('\n', ' ')
        if evalType == 'ppl':
            return self.perplexity(data, params)
        elif evalType == 'byWordMetrics':
            self.byWordMetrics(data, params)
            return None
        else:
            sys.stderr.write(f"evalType: {evalType} not recognized\n")
            sys.exit(1)

    #TODO: 25 points
    def generate(self, n:int) -> list[str]:
        """ Generates n sentences using your ngram model. 
        Generating a sentence begins by seeding the start of a sentence and then
        iteratively adding to the sentence until a eos token is generated, at
        which point the sentence is done. 

        Note: You should generate using self.ngrams, which will only include
        counts for your observations (that is, you will not generate
        hallucinations in this assignment). 

        The following demonstration may help you in sampling words:
            >>> import random
            >>> population = ['the', 'cat', 'sleeps', '<unk>', '</s>']
            >>> weights = [100, 50, 10, 100, 100]
            >>> random.choices(population, weights)[0]
            'sleeps'
            >>> random.choices(population, weights)[0]
            'the'

        Args:
            n (int): number of sentences to generate

        Returns:
            list[str]: a list of n sentences

        For example, 
        >>> ngram = NGram(ngram_size=2, vocab_file='./tiny_data/cat_vocab.txt')
        >>> ngram.train('./tiny_data')
        >>> ngram.generate(5)
        ['<s> <s> the other cat . </s>', 
         '<s> <s> other was unhappy , and as <unk> <unk> know an is one that
         again </s>', '<s> <s> the other cat . </s>', '<s> <s> the other cat was
         unhappy , and as <unk> <unk> <unk> know , an unhappy cat is one that
         again </s>', '<s> <s> other was unhappy , and as <unk> know an is one
         that again </s>']
        """
        raise NotImplementedError

if __name__ == '__main__':

    ngram = NGram()
