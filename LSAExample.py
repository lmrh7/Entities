import pandas as pd
import pickle
import re
import time

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
corpus = corpora.MmCorpus('/tmp/deerwester.mm')
print(corpus)

photos = pd.read_pickle(r'C:\Users\LauraM\Desktop\mon_ami_gabi_photos.pkl')
reviews = pd.read_pickle(r'C:\Users\LauraM\Desktop\mon_ami_gabi_reviews.pkl')
restaurantName="mon ami gabi"

#docs=photos['caption']
#resultsFile=("%s_capTags"%restaurantName)

docs=reviews['text']
resultsFile='%s_revEntities'%restaurantName
maxItems=6000

docsEntities=[]
counter=0
totalDocs=len(docs)
#sum=0
#avglen=0
#for text in docs:
#    counter=counter+1
#    sum=sum+len(text)

#avglen=sum/counter
#print ('Average length: %f')%avglen
#661 avg length for mon ami gabi reviews
#29 avg length for mon ami gabi captions
# 76.262 s time spent processing for mon ami gabi captions
#736.241 s time spent processing for mon ami gabi captions

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())


f = open(resultsFile, 'wb')
startTime=time.time()
counter=0
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in docs]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
       frequency[token] += 1

entities = [[token for token in text if frequency[token] > 1]
          for text in texts]

from pprint import pprint   # pretty-printer
pprint(entities)
pickle.dump(entities, f)
print '%s/%d : %s'%(id,totalDocs,entities)


f.close()
endTime=time.time()
elapsed=endTime-startTime
print "Time spent: %f"%elapsed
#f=open('', 'rb')
#pickle.load(f)
