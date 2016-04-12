import pandas as pd
import pickle

reviews = pd.read_pickle(r'C:\Users\LauraM\Desktop\mon_ami_gabi_reviews.pkl')
restaurantName="MonAmiGabi"
docs=reviews['text']
trainingFile=open("%sTraining.pkl"%restaurantName,'wb')
testingFile=open("%sTesting.pkl"%restaurantName,'wb')
percentageTraining=0.2;
counter=0;
totalRevs = len(reviews);
maxRevs=percentageTraining*totalRevs;
print "%d Total Reviews"%totalRevs
for review in reviews['text']:
    counter=counter+1
    print review
    if(counter<maxRevs):
        print "%f%% Training:"%(counter/totalRevs*100)
        pickle.dump(review,trainingFile);
    else:
        if(counter==maxRevs):
            trainingFile.close();
            print "Opening reviewsTesting"

        print "%f%% Testing:"%(counter/totalRevs*100)
        pickle.dump(review,testingFile)

testingFile.close();