from __future__ import print_function
import pickle
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from gensim.utils import simple_preprocess
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

class MyCorpus(object):
     def __iter__(self):
         for line in open('./data/MonAmiGabiTraining.pkl','rb'):
             # assume there's one document per line, tokens separated by whitespace
             yield dictionary.doc2bow(line.lower().split())

training=extract_reviews('./data/MonAmiGabiTesting.pkl');

documents=training;

print("Texts before STOPWORDS: ",documents)

texts=[]
texts = [[word for word in simple_preprocess(document) if word not in STOPWORDS] for document in documents]
#texts = [[word for word in documents.lower().split() if word not in STOPWORDS]]

print("Texts after STOPWORDS: ",texts)
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
       frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts];

from pprint import pprint   # pretty-printer
#pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('./data/MonAmiGabiTrain.dict') # store the dictionary, for future reference
print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('./data/MonAmiGabiTrain.mm', corpus) # store to disk, for later use
#print(corpus)

#print("\n Build DTM")
#tf = CountVectorizer(stop_words='english')

#print("\n Fit DTM")
#tfs1 = tf.fit_transform(token_dict.values())

# set the number of topics to look for
num = 3
print("Creating tfidf")
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
#print(tfidf)

corpus_tfidf = tfidf[corpus]
#for doc in corpus_tfidf:
#    print(doc)

print("Creating LSI") #Can be used online
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

print("Creating LDA") #Not really good online unless topics don't change too much
model = gensim.models.LdaModel(corpus,id2word=dictionary,num_topics=num)
model.print_topics(num)