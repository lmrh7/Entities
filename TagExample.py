import pandas as pd
import pickle
import re
import time

from textblob import TextBlob
from textblob import Word
from textblob import WordList

photos = pd.read_pickle(r'C:\Users\LauraM\Desktop\mon_ami_gabi_photos.pkl')
reviews = pd.read_pickle(r'C:\Users\LauraM\Desktop\mon_ami_gabi_reviews.pkl')
restaurantName="mon ami gabi"

#docs=photos['caption']
#resultsFile=("%s_capTags"%restaurantName)

docs=reviews['text']
resultsFile='%s_revTags'%restaurantName
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

f = open(resultsFile, 'wb')
startTime=time.time()
counter=0
for text in docs:
    if(counter<maxItems):
        counter=counter+1
        textTB=TextBlob(text)
        textTB.correct()
        #remove numerical and symbol chars
        print('%d : Processed Text: %s')%(counter,textTB)
        #print textTB.sentiment
        id='%d'%counter
        entities=[id]
        for phrase in textTB.noun_phrases:
            print phrase
            #Maybe remove the phrases containing the name of the restaurant?
            phrase.singularize()
            phrase.lemmatize()
            print ('Processed Word: %s'%phrase)
            entities.append(phrase)
        if(len(entities)>1):
            docsEntities.append(entities)
            pickle.dump(entities, f)
            print '%s/%d : %s'%(id,totalDocs,entities)


f.close()
endTime=time.time()
elapsed=endTime-startTime
print "Time spent: %f"%elapsed
#f=open('', 'rb')
#pickle.load(f)