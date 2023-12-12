import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
from sklearn.metrics import accuracy_score

url = 'https://raw.githubusercontent.com/meghjoshii/NSDC_DataScienceProjects_SentimentAnalysis/main/IMDB%20Dataset.csv'

data= pd.read_csv(url)
data

# TODO: Print the first 5 rows of the data using head function of pandas
#Head Function
data.head()

data.describe()

sentiment_counts=data['sentiment'].value_counts()
print(sentiment_counts)

import nltk
nltk.download('punkt')
data['review']=data['review'].apply(word_tokenize)

data['review'][1]

data['review']=data['review'].apply(lambda x: [item for item in x if item.isalpha()])

print(' '.join(data['review'][1]))

data['review']=data['review'].apply(lambda x: [item.lower() for item in x])

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))

data['review']=data['review'].apply(lambda x: [item for item in x if item not in stop_words])

from nltk.stem import PorterStemmer
ps=PorterStemmer()
data['review']=data['review'].apply(lambda x: [ps.stem(item) for item in x ])

#train reviews
data['review'] = data['review'].apply(lambda x: " ".join(x))
train_reviews = data.review[:40000]

#test reviews
test_reviews = data.review[40000:]

#TODO: train sentiments
train_sentiment = data.sentiment[:40000]
test_sentiment = data.sentiment[40000:]

#Count vectorizer for bag of words
cv = CountVectorizer(min_df=0, max_df=1, binary = False, ngram_range = (1,3))


#transformed train reviews
cv_train_reviews = cv.fit_transform(train_reviews)

#transformed test reviews
cv_test_reviews = cv.transform(test_reviews)

#labeling the sentient data
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

# transformed sentiment data
lb_train_sentiments = lb.fit_transform(train_sentiment)

#TODO: transformed test sentiment data (similar to count vectorizer, transform test reviews, name it lb_test_sentiments)
lb_test_binary = lb.fit_transform(test_sentiment)

# training the model
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

# fitting the model
mnb_bow = mnb.fit(cv_train_reviews, lb_train_sentiments)

#Predicting the model for bag of words
mnb_bow_predict = mnb.predict(cv_test_reviews)

#Accuracy score for bag of words
mnb_bow_score = accuracy_score(lb_test_binary, mnb_bow_predict)
print("Accuracy :", mnb_bow_score)
