import logging
import itertools
import sys
import pickle
import lda
import numpy as np
import gensim

from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS

#DEFINITIONS
def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def extract_reviews(dump_file):
    f = open(dump_file, 'rb')
    revs=[]
    while 1:
        try:
            rev=pickle.load(f)
            print( rev )
            revs.append(rev)
        except EOFError:
            break
    return revs

#Do we need to make a new filter?
#def filter_revs(rev):

def iter_rev(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    print ("inside iter_rev")
    revs=[]
    revs= extract_reviews(dump_file);
    print ("Reviews: %s"%revs)
    for text in revs:
        #print "Unfiltered: %s"%text
        text = filter_wiki(text)
        #print "filtered: %s"%text
        tokens = tokenize(text)
        print (tokens)
        if len(tokens) < 30 :#or any(title.startswith(ns + ':') for ns in ignore_namespaces):
           continue  # ignore short articles and various meta-articles
        yield tokens


class RevWikiCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).

        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        self.titles = []
        for tokens in itertools.islice(iter_rev(self.dump_file), self.clip_docs):
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs

#CODE
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

dumpFilePath='./data/MonAmiGabiTraining.pkl';
# only use simplewiki in this tutorial (fewer documents)
# the full wiki dump is exactly the same format, but larger
#stream = iter_wiki('./data/simplewiki-20140623-pages-articles.xml.bz2')

doc_stream = (tokens for tokens in iter_rev(dumpFilePath))

id2word_wiki = gensim.corpora.Dictionary(doc_stream);
print(id2word_wiki)
# ignore words that appear in less than 20 documents or more than 10% documents
id2word_wiki.filter_extremes(no_below=20, no_above=0.1)
f = open('id2word', 'wb');
pickle.dump(id2word_wiki, f)
f.close()
print(id2word_wiki)

# create a stream of bag-of-words vectors
rev_wiki_corpus = RevWikiCorpus(dumpFilePath, id2word_wiki)

f = open('revWikiObjects', 'wb');
pickle.dump(rev_wiki_corpus, f)
f.close()

gensim.corpora.MmCorpus.serialize('./data/reviews_wiki_bow.mm', rev_wiki_corpus)
mm_corpus=rev_wiki_corpus

clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)
print ('ClippedCorpus')
print (clipped_corpus)

lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=10, id2word=id2word_wiki, passes=4)

gensim.corpora.MmCorpus.serialize('./data/reviews_wiki_lda.mm', lda_model[mm_corpus])