import pandas as pd

from textblob import TextBlob
from textblob import Word

mon_ami_photos = pd.read_pickle(r'C:\Users\LauraM\Desktop\mon_ami_gabi_photos.pkl')
mon_ami_reviews = pd.read_pickle(r'C:\Users\LauraM\Desktop\mon_ami_gabi_reviews.pkl')
#print(mon_ami_photos['caption'])
#print(mon_ami_reviews['text'])

reviewsEntities=[]
counter=0
totalRev=len(mon_ami_reviews['text'])
for review in mon_ami_reviews['text']:
    print '%d : %s'%(len(reviewsEntities),review)
    textTB=TextBlob(review);
    textTB.correct();
    print textTB.sentiment
    entities=[]
    for word in textTB.noun_phrases:
        w=Word(word)
        w.singularize();
        w.lemmatize()
        entities.append(w)
        print(w)
    reviewsEntities.append(entities)
    print '%d/%d : %s'%(len(reviewsEntities),totalRev,entities)

