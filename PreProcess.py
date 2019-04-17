
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = stopwords.words("english")
lmt = WordNetLemmatizer()
fdist = FreqDist()

def Tokenize(text):
    return word_tokenize(text)

def RemoveStopWords(text):
    post_rm_stop_words = []
    tokens = text.split()
    for token in tokens:
        if token in stop_words:
            fdist[token] += 1
        else:
            post_rm_stop_words.append(token)
    return ' '.join(post_rm_stop_words)

def Lemmatize(words):
    #string_pos_tag = nltk.pos_tag([string])
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(lmt.lemmatize(word))
        
    return lemmatized_words