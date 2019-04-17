
#Reading the dataset
import pandas as pd
train=pd.read_csv('In-Class-Project-Dataset/Tweets-train.csv')

#Selecting only sentiment and tweet column from the entire data set
train=train[['airline_sentiment','text']]

#Observe randomly generated 10 tweets for each sentiment with respect to the following:
#Text contains references with ‘@’
#Text contains links (http , https )
#Text contains punctuations
#Text contains Emoticons 

#See some positive sentiments
for each in train[train['airline_sentiment']=="positive"].sample(10,random_state=10)['text']:
    print (each)
    
#See some negative sentiments
for each in train[train['airline_sentiment']=="negative"].sample(10,random_state=10)['text']:
    print (each) 

#See some neutral sentiments
for each in train[train['airline_sentiment']=="neutral"].sample(10,random_state=10)['text']:
    print (each) 

# ## Observations are as follows-
# 

# #### 1.  Data contains words starting with '@'

# #### 2.  Data contains words having '#' 

# #### 3.  Data contains links 'https:...."

# #### 4. Data contains emoticons and punctuations such as ' , . ; ❤️✨ ! etc etc
    
#The next step can be to clean the data and remove such things as they are not going to help in classifer model.

#@ mentions
import re
print (train.text[5]) 
print (re.sub(r'@+','',train.text[5]))

#Links 
print (train.text[10])
print (re.sub('http?://[A-Za-z0-9./]+','',train.text[10]))

# selects only aplhabets numbers so that punctuations and emoticons are removed.
print (train.text[22])
print (re.sub("[^a-zA-Z0-9]", " ",train.text[22]))
print (train.text[5977])
print (re.sub("[^a-zA-Z0-9]", " ",train.text[5977]))

#We can prepare a function to clean all the above observed tokens from the tweet text.
#Save changes in a new column 
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
def tweet_cleaner(text):
    text=re.sub(r'@+','',text)
    text=re.sub('http?://[A-Za-z0-9./]+','',text)
    text=re.sub("[^a-zA-Z]", " ",text)
    lower_case = text.lower()
    words = tokenizer.tokenize(lower_case)
    return (" ".join(words)).strip()
train['Cleaned-Text']=pd.Series(list(map(lambda x:tweet_cleaner(x),train['text'])))

#See some words in All sentiments
from collections import Counter
for group_name,subset in train.groupby('airline_sentiment'):
    sentimentData=subset['Cleaned-Text']
    words=[]
    
    i = 0
    for each in sentimentData:
        i+=1
        words.extend(each.split(" "))
        
    print (group_name)
    print (Counter(words).most_common(5))

#We observe that most of the frequencies are of stopwords , so let's remove them 
from PreProcess import RemoveStopWords 
train['Clean-Text-StopWords-Removed']=pd.Series((map(lambda x:RemoveStopWords(x),train['Cleaned-Text'])))

#Again let's see the counts of most common words after removing stopwords.
for group_name,subset in train.groupby('airline_sentiment'):
    sentimentData=subset['Clean-Text-StopWords-Removed']
    words=[]
    for each in sentimentData:
        words.extend(each.split(" "))

    print (group_name)
    print (Counter(words).most_common(5))
 

#Remove below words from all the tweets.
#americanair, united, delta, southwestair, jetblue, virginamerica, usairways, flight, plane
#Save changes in a new column and list down most common 15 words.
def RemoveExplicitlyMentionedWords(string,listofWordsToRemove):
    listOfAllWords=string.split(" ")
    listOfWords= [x for x in listOfAllWords if x not in listofWordsToRemove]
    return (" ".join(listOfWords)).strip()    
list_of_words_to_remove=['americanair','united','delta','southwestair','jetblue','virginamerica','usairways','flight','plane']
train['Final-Wrangled-Text']=pd.Series(list(map(lambda x:RemoveExplicitlyMentionedWords(x,list_of_words_to_remove),train['Clean-Text-StopWords-Removed'])))

#Count Again words in All sentiments
from collections import Counter
for group_name,subset in train.groupby('airline_sentiment'):
    sentimentData=subset['Final-Wrangled-Text']
    words=[]
    for each in sentimentData:
        words.extend(each.split(" "))
    print (group_name)
    print (Counter(words).most_common(15))


#Encode Sentiments using Label Encoder
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
train['SentimentLabel']=l.fit_transform(train['airline_sentiment'])
train.head(4)

#Here we observe,
#0->neutral
#1->positive
#2->negative

#Vectorize the Text Column (You can choose any vectorizer of your choice)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x_train=vectorizer.fit_transform(train['Final-Wrangled-Text'])

#Prepare a multiclass Classification model using any classification algorithm and create a model 
y_train=train['SentimentLabel']

# Preparing Model Using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
nb = MultinomialNB()
nb.fit(x_train,y_train)

#Read the test data and carry our data cleaning, encoding and vectorising operations on the test data
test=pd.read_csv('In-Class-Project-Dataset/Tweets-test.csv')
test=test[['airline_sentiment','text']]
test['Cleaned-Text']=pd.Series(map(lambda x:tweet_cleaner(x),test['text']))
test['Clean-Text-StopWords-Removed']=pd.Series(map(lambda x:RemoveStopWords(x),test['Cleaned-Text']))
test['Final-Wrangled-Text']=pd.Series(map(lambda x:RemoveExplicitlyMentionedWords(x,list_of_words_to_remove),test['Clean-Text-StopWords-Removed']))
x_test=vectorizer.transform(test['Final-Wrangled-Text'])

#Encoding label for test data as well
test['SentimentLabel']=l.transform(test['airline_sentiment'])
y_test=test['SentimentLabel']

#Predict the sentiments for test data
y_pred=nb.predict(x_test)

def GetOrignalSentiment(val):
    if val==0:
        return 'negative'
    elif val==1:
        return 'neutral'
    else:
        return 'positive'
    
Result=test[['text','airline_sentiment']]
Result['Predicted_sentiment']=pd.Series(map(lambda x:GetOrignalSentiment(x),y_pred))
Result.head(3)

#Print and explain the Confusion Matrix 
print ("Confusion Matrix:\n\n",metrics.confusion_matrix(Result['airline_sentiment'],Result['Predicted_sentiment'],labels=['negative','neutral','positive']))

#Explaning Confusion Matrix Elements
for i,x in Result.groupby(['airline_sentiment','Predicted_sentiment']):
    print ("Actual "+ i[0]+ " Predicted "+i[1]+ ":", len(x))

#Compute Accuracy of your model
ActualNegativePrdictedNegative=2396
ActualNeutralPrdictedNeutral=333
ActualPositivePrdictedPositive=365
TotalCorrect=ActualNegativePrdictedNegative+ActualNeutralPrdictedNeutral+ActualPositivePrdictedPositive
print ("Accuracy=",TotalCorrect*100.0/len(test) ,"%")
