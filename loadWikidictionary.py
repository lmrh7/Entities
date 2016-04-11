import logging
import itertools
import sys
import pickle
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

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens


class WikiCorpus(object):
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
        for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs

#CODE
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

dumpFilePath='./data/enwiki-latest-pages-articles.xml.bz2';
# only use simplewiki in this tutorial (fewer documents)
# the full wiki dump is exactly the same format, but larger
#stream = iter_wiki('./data/simplewiki-20140623-pages-articles.xml.bz2')
doc_stream = (tokens for _, tokens in iter_wiki(dumpFilePath))

id2word_wiki = gensim.corpora.Dictionary(doc_stream);
print(id2word_wiki)
# ignore words that appear in less than 20 documents or more than 10% documents
id2word_wiki.filter_extremes(no_below=20, no_above=0.1)
f = open('id2word', 'wb');
pickle.dump(id2word_wiki, f)
f.close()
print(id2word_wiki)

# create a stream of bag-of-words vectors
wiki_corpus = WikiCorpus(dumpFilePath, id2word_wiki)
vector = next(iter(wiki_corpus))
print(vector)  # print the first vector in the stream
f = open('wikiObjects', 'wb');
pickle.dump(wiki_corpus, f)
f.close()

# what is the most common word in that first article?
most_index, most_count = max(vector, key=lambda (word_index, count): count)
print(id2word_wiki[most_index], most_count)

gensim.corpora.MmCorpus.serialize('./data/wiki_bow.mm', wiki_corpus)