import pandas as pd
import pickle
import re
import time
import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities

#dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
f=open('./data/wiki_bow.mm','rb');
n=f.next();
while():
    print n
    n=f.next()

f.close();
mm_corpus = corpora.MmCorpus('./data/reviews_wiki_bow.mm')
print(mm_corpus)
clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)
print 'ClippedCorpus'
print clipped_corpus
# ClippedCorpus new in gensim 0.10.1
# copy&paste it from https://github.com/piskvorky/gensim/blob/0.10.1/gensim/utils.py#L467 if necessary (or upgrade your gensim)

print 'id2word_wiki: '
id2word_wiki = pickle.load(r'/data/id2word')
print id2word_wiki

lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=10, id2word=id2word_wiki, passes=4)

gensim.corpora.MmCorpus.serialize('./data/reviews_wiki_lda.mm', lda_model[mm_corpus])

_ = lda_model.print_topics(-1)  # print a few most important words for each LDA topic