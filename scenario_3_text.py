#!/usr/bin/env python
# coding: utf-8

# ## Scenario II - All NLP Preprocessing Steps 

# In[1]:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup 
import re
from gensim.parsing.preprocessing import remove_stopwords
import nltk

import unidecode
from num2words import num2words
from autocorrect import Speller


# In[2]:


df = pd.read_csv("base_clean")


# In[3]:


X = df.text
y = df.label


# In[ ]:

def remove_html_tags(text):
        unicode = unidecode.unidecode(text)
        stop = remove_stopwords(unicode)
        twitter = re.sub(pattern=r'@[A-Za-z0-9]+',repl=' ', string=stop)
        html = re.sub(pattern=r'<.*?>', repl=' ', string=twitter)
        urls = re.sub(pattern=r'https?://\S+|www\.\S+', repl=' ', string=html)
        special_ch = re.sub(pattern='[^a-zA-Z]',repl=' ',string=urls)
        special_cha= re.sub(pattern='\[[^]]*\]', repl=' ', string=special_ch)
        lower = special_cha.lower()
        return lower

def spell_autocorrect(text):
        correct_spell_words = []
        spell_corrector = Speller(lang='en')
        for word in word_tokenize(text):
            correct_word = spell_corrector(word)
            correct_spell_words.append(correct_word)
        correct_spelling = correct_spell_words
        return correct_spelling

def lemmatization(tokens):
        lemma = WordNetLemmatizer()
        for index in range(len(tokens)):
            lemma_word = lemma.lemmatize(tokens[index])
            tokens[index] = lemma_word
        return ' '.join(tokens)




# In[ ]:






# In[ ]:


## def num_to_words(word):
##         new_word = []
##         for word in word_tokenize(word):
##             if word.isdigit():
##                 correct_word = num2words(word)
##                 new_word.append(correct_word)
##             else:
##                 new_word.append(word.lower())
##         corret_word = ' '.join(new_word)
##         return corret_word
## X = X.apply(num_to_words)
## X


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(  
    X, y,
    test_size= 0.30,
    random_state= 50)


# In[ ]:


tfidf = TfidfVectorizer(lowercase=False)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)


# In[ ]:


count_vectorizer =  CountVectorizer(lowercase=False)
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)


# #### Training the Algorithm

# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


## Fits and train the NB algorithm with the tfidf vectorized training dataset
## Apply the algorithm to the tfidf test dataset

nb_tfidf = MultinomialNB()
nb_tfidf.fit(tfidf_train, y_train)
pred_tfidf = nb_tfidf.predict(tfidf_test)


# In[ ]:


## Fits and train the NB algorithm with the count vectorized training dataset
## Apply the algorithm to the count test dataset

nb_count = MultinomialNB()
nb_count.fit(count_train, y_train)
pred_count = nb_count.predict(count_test)


# #### Models evaluation

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[ ]:


print('Accuracy tfidf: ', metrics.accuracy_score(y_test, pred_tfidf))
print('Accuracy count: ', metrics.accuracy_score(y_test, pred_count))


# #### K-fold

# In[ ]:


from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score


# In[ ]:


X_kfold_tfidf = tfidf.transform(df.text)


# In[ ]:


X_kfold_count = count_vectorizer.transform(df.text)


# In[ ]:


kfold_5_tfidf = cross_val_score(nb_tfidf , X_kfold_tfidf, y, cv = 5)
kfold_10_tfidf = cross_val_score(nb_tfidf , X_kfold_tfidf, y, cv = 10)
kfold_5_count = cross_val_score(nb_count , X_kfold_count, y, cv = 5)
kfold_10_count = cross_val_score(nb_count , X_kfold_count, y, cv = 10)


# In[ ]:


print('Accuracy tfidf: ', metrics.accuracy_score(y_test, pred_tfidf))
print('Accuracy count: ', metrics.accuracy_score(y_test, pred_count))
print("Avg accuracy kfold 5 tfidf: {}".format(kfold_5_tfidf.mean()))
print("Avg accuracy kfold 10 tfidf: {}".format(kfold_10_tfidf.mean()))
print("Avg accuracy kfold 5 Count: {}".format(kfold_5_count.mean()))
print("Avg accuracy kfold 10 Count: {}".format(kfold_10_count.mean()))

