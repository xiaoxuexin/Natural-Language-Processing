import logging
import gensim
from gensim import corpora, models, similarities
from gensim.test.utils import common_texts
import numpy as np
from gensim.corpora.dictionary import Dictionary
import os
import argparse
from glob import glob
from collections import defaultdict
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from gensim.models import ldamodel
from nltk.tokenize import TreebankWordTokenizer
kTOKENIZER = TreebankWordTokenizer()

def tokenize_file(filename):
    contents = open(filename).read()
    for ii in kTOKENIZER.tokenize(contents):
        yield ii

# print(common_texts)
# print(Dictionary(common_texts))

class VocabBuilder:
    """
    Creates a vocabulary after scanning a corpus.
    """

    def __init__(self, lang="english", min_length=3, cut_first=100):
        """
        Set the minimum length of words and which stopword list (by language) to
        use.
        """
        self._counts = FreqDist()
        self._stop = set(stopwords.words(lang))
        self._min_length = min_length
        self._cut_first = cut_first

        print(("Using stopwords: %s ... " % " ".join(list(self._stop)[:10])))

    def scan(self, words):
        """
        Add a list of words as observed.
        """

        for ii in [x.lower() for x in words if x.lower() not in self._stop \
                       and len(x) >= self._min_length]:
            self._counts[ii] += 1

    def get_vocab(self, words):
        return [x.lower() for x in words if x.lower() not in self._stop \
                       and len(x) >= self._min_length]

    def vocab(self, size=5000):
        """
        Return a list of the top words sorted by frequency.
        """
        if len(self._counts) > self._cut_first + size:
            return list(self._counts.keys())[self._cut_first:(size + self._cut_first)]
        else:
            return list(self._counts.keys())[:size]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--doc_dir", help="Where we read the source documents",
                           type=str, default=".", required=False)
    argparser.add_argument("--language", help="The language we use",
                           type=str, default="english", required=False)
    argparser.add_argument("--output", help="Where we write results",
                           type=str, default="result", required=False)
    argparser.add_argument("--vocab_size", help="Size of vocabulary",
                           type=int, default=1000, required=False)
    argparser.add_argument("--num_topics", help="Number of topics",
                           type=int, default=5, required=False)
    argparser.add_argument("--num_iterations", help="Number of iterations",
                           type=int, default=1000, required=False)
    args = argparser.parse_args()

    vocab_scanner = VocabBuilder(args.language)

    # Create a list of the files
    search_path = "%s/*.txt" % args.doc_dir
    files = glob(search_path)
    assert len(files) > 0, "Did not find any input files in %s" % search_path
    l = []
    # Create the vocabulary
    for ii in files:
        # print(ii)
        # vocab_scanner.scan(tokenize_file(ii))
        vocab = vocab_scanner.get_vocab(tokenize_file(ii))
        l.append(vocab)
    # print(l)
    my_diction = corpora.Dictionary(l)
    my_corpus = [my_diction.doc2bow(text) for text in l]
    # Gensim package
    lda = ldamodel.LdaModel(my_corpus, num_topics=10, id2word=my_diction)
    a = lda.print_topics(num_topics=5, num_words=50)
    mytopicfile = open("gensim.txt", 'w')
    for i in range(len(a)):
        mytopicfile.write(str(a[i]))
        mytopicfile.write('\n')
    mytopicfile.close()

    # mallet package
    os.environ.update({'MALLET_HOME': r'‎/Users/xinxiaoxue/mallet-2.0.8/'})
    mallet_path = '‎/Users/xinxiaoxue/mallet-2.0.8/bin/mallet'  # update this path
    model = gensim.models.wrappers.ldamallet.LdaMallet(mallet_path, corpus=my_corpus, num_topics=5, id2word=my_diction, iterations=1000)
    b = lda.print_topics(num_topics=5, num_words=50)
    my_topicfile = open("mallet.txt", 'w')
    for i in range(len(b)):
        my_topicfile.write(str(b[i]))
        my_topicfile.write('\n')
    my_topicfile.close()