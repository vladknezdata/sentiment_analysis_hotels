import nltk
nltk.download("vader_lexicon")
nltk.download("stopwords")

from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from collections import Counter
import re
import math
import html
import sklearn
import sklearn.metrics as metrics
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint




#%matplotlib inline

import pymongo
from pymongo import MongoClient
import pprint

client = MongoClient('mongodb+srv://vlad:starReviews1@reviews-4f7lv.mongodb.net/test?retryWrites=true&w=majority')
db = client.reviews
reviews = db.newyork
hotelDf = pd.DataFrame(list(reviews.find()))
print(hotelDf)
print()

# Read in from pandas
#hotelDf = pd.read_csv('JBNYC11.csv')
# hotelDf.columns=['id','filePath','hotelName','review','ratingScore','groundTruth']
#hotelDf.columns=['id', 'filePath','hotelName','review','ratingScore','groundTruth']
hotelDf.columns = ['_id', 'filePath', 'groundTruth', 'hotelName', 'ratingScore', 'review']
df_filter = pd.DataFrame(list(reviews.find()))
#df_filter = pd.read_csv('JBNYC11.csv')
# hotelDf.columns=['id','filePath','hotelName','review','ratingScore','groundTruth']
#df_filter.columns=['filePath','hotelName','review','ratingScore','groundTruth']
df_filter.columns = ['_id', 'filePath', 'groundTruth', 'hotelName', 'ratingScore', 'review']

df_filter.drop(columns=['_id'])
hotelDf.drop(columns=['_id'])

df_filter = df_filter.groupby(['hotelName'])['groundTruth'].agg(['count'])
df_filter = df_filter.reset_index()
df_filter = df_filter.loc[df_filter['count']>50]

hotelDf = pd.merge(hotelDf, df_filter, how='inner', indicator=True)

import re
df2=hotelDf['hotelName'].unique()

hotelStr=[]
for i in df2:
    hotelStr.append(re.sub(r'[.!,;?]', ' ', i).split())

flat_list = [item for sublist in hotelStr for item in sublist]

# for i in hotelDf['hotelName'].unique():
for i in flat_list:
    hotelDf['review'] = hotelDf["review"].str.replace(i, " ", case = False)


# There are unparsed html tags in the hotelnames. We can change the html tags to ascii equivalents by using the following code.
for i in range(len(hotelDf)):
    hotelname = hotelDf.at[i, 'hotelName']
    hotelname = hotelname.encode("utf-8")
    hotelname = hotelname.decode("ascii", "ignore")
#     hotelname = hotelname.decode("ascii", "namereplace")
    hotelname = html.unescape(hotelname)
    hotelDf.at[i, 'hotelName'] = hotelname

# specialChar = ['u0430','u043d','u043e']
hotelDf['review'] = hotelDf["review"].str.replace("\\u2019", "'", case = False)
hotelDf['review'] = hotelDf["review"].str.replace("u2019", "'", case = False)
hotelDf['review'] = hotelDf["review"].str.replace(" ei ", "", case = False)
hotelDf['review'] = hotelDf["review"].str.replace("bo red", "bored", case = False)

# hotelDf['review'] = hotelDf["review"].str.replace("tribeca", "'", case = False)
# hotelDf['review'] = hotelDf["review"].str.replace("  u2019", "'", case = False)
# hotelDf['review'] = hotelDf['review'].str.replace(j,"")
# hotelDf
locationName = ['tribeca','Tribeca','Times Square', 'Timessquare','timessquare','times square',
                'grand central', 'Grand Central','chelsea','Chelsea','hudson','river','Hudson', 'River', 'Empire', 'empire', 'central park', 'Central park', 'Central Park', 'building', 'Building']

for z in locationName:
    hotelDf['review'] = hotelDf["review"].str.replace(z,"", case = False)

specialChar = '\\u041e\\u0447\\u0435\\u043d\\u044c \\u0441\\u043e\\u0432\\u0440\\u0435\\u043c\\u0435\\u043d\\u043d\\u044b\\u0439, \\u043d\\u043e\\u0432\\u044b\\u0439, \\u0441\\u043a\\u043e\\u0440\\u043e\\u0441\\u0442\\u043d\\u044b\\u0435 \\u043b\\u0438\\u0444\\u0442\\u044b, \\u043e\\u0431\\u0437\\u043e\\u0440\\u043d\\u0430\\u044f \\u043f\\u043b\\u043e\\u0449\\u0430\\u0434\\u043a\\u0430 \\u043d\\u0430\\u0432\\u0435\\u0440\\u0445\\u0443 \\u0437\\u0434\\u0430\\u043d\\u0438\\u044f, \\u043e\\u0442\\u043b\\u0438\\u0447\\u043d\\u044b\\u0435 \\u0432\\u0438\\u0434\\u044b \\u0438\\u0437 \\u043e\\u043a\\u043d\\u0430 \\u043d\\u043e\\u043c\\u0435\\u0440\\u0430, \\u0440\\u0430\\u0441\\u043f\\u043e\\u043b\\u043e\\u0436\\u0435\\u043d\\u0438\\u0435 \\u0438\\u0434\\u0435\\u0430\\u043b\\u044c\\u043d\\u043e\\u0435, \\u043f\\u0440\\u0438\\u0432\\u0435\\u0442\\u043b\\u0438\\u0432\\u044b\\u0439 \\u0438 \\u0432\\u0435\\u0436\\u043b\\u0438\\u0432\\u044b\\u0439 \\u043f\\u0435\\u0440\\u0441\\u043e\\u043d\\u0430\\u043b, \\u0443\\u0434\\u043e\\u0431\\u043d\\u0430\\u044f \\u043f\\u0440\\u043e\\u0441\\u0442\\u043e\\u0440\\u043d\\u0430\\u044f \\u043a\\u0440\\u043e\\u0432\\u0430\\u0442\\u044c, \\u043f\\u0440\\u0435\\u043a\\u0440\\u0430\\u0441\\u043d\\u0430\\u044f \\u0437\\u0432\\u0443\\u043a\\u043e\\u0438\\u0437\\u043e\\u043b\\u044f\\u0446\\u0438\\u044f, \\u0432 \\u043d\\u043e\\u043c\\u0435\\u0440\\u0435 \\u043c\\u0438\\u043a\\u0440\\u043e\\u0432\\u043e\\u043b\\u043d\\u043e\\u0432\\u043a\\u0430, \\u043a\\u043e\\u0444\\u0435-\\u043c\\u0430\\u0448\\u0438\\u043d\\u0430,'.split("\\")
for j in specialChar:
    hotelDf['review'] = hotelDf['review'].str.replace(j,"")

# Initialize  the sentiment Analyzer
sid = SentimentIntensityAnalyzer()

vaderScores = []
#Assign Vader score to individual review using Vader compound score
for rownum, review in enumerate(hotelDf['review']):
    scores = sid.polarity_scores(review)
    vaderScores.append(scores['compound'])
    if (rownum % 1000 == 0):
            print("processed %d reviews" % (rownum+1))

hotelDf = hotelDf.assign(vaderScore = vaderScores)

ratingByHotel = hotelDf.groupby(['hotelName']).mean()['ratingScore'].reset_index()
vaderByHotel = hotelDf.groupby(['hotelName']).mean()['vaderScore'].reset_index()

#RATINGS
ratingByHotel = ratingByHotel.sort_values('ratingScore', ascending=False)

# VADER
vaderByHotel = vaderByHotel.sort_values('vaderScore', ascending=False)


#1. Word Frequency: While not as effective, still running the code!
def get_topk_ngram(df, ngram_range=(1,1), k=None, stopwords=True, with_count=False):
    '''
    Extract the most frequently occurred words in countvector
    '''
    if stopwords:
        temp = []
        for name in hotelDf.hotelName.unique():
            for token in name.split():
                if len(token) > 1:
                    temp.append(token)
        my_stop_words = ENGLISH_STOP_WORDS.union(temp)
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=my_stop_words, max_features=500)
        
    else:
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=None, max_features=k)
        
    countvector = vectorizer.fit_transform(df['review'])

    # Get topk occurred ngrams
    topk_words = []
    sortedindices = countvector.toarray().sum(axis=0).argsort()[::-1][:k]
    counts = countvector.toarray().sum(axis=0)
    
    for i in sortedindices:
        word = vectorizer.get_feature_names()[i]
        
        if with_count:
            count = counts[i]
            topk_words.append((word, count))
        else:
            topk_words.append(word)
            
    return topk_words


topkTotal = get_topk_ngram(hotelDf, k=500)
topkTotal_bigram = get_topk_ngram(hotelDf, ngram_range=(2,2), k=500)
topkPos = get_topk_ngram(hotelDf.loc[hotelDf['groundTruth']=='positive'], ngram_range=(1,1), k=10, with_count=True)
topkNeg = get_topk_ngram(hotelDf.loc[hotelDf['groundTruth']=='negative'], ngram_range=(1,1), k=10, with_count=True)
topkPos_bigram = get_topk_ngram(hotelDf.loc[hotelDf['groundTruth']=='positive'], ngram_range=(2,2), k=10, with_count=True)
topkNeg_bigram = get_topk_ngram(hotelDf.loc[hotelDf['groundTruth']=='negative'], ngram_range=(2,2), k=10, with_count=True)

print("The most frequently occured top 10 words in positive reviews")
#pprint(pd.DataFrame(topkPos, columns=['Word', 'Count']))

print("\nThe most frequently occured top 10 words in negative reviews")
#pprint(pd.DataFrame(topkNeg, columns=['Word', 'Count']))

print("\nThe most frequently occured top 10 bigrams in positive reviews")
#pprint(pd.DataFrame(topkPos_bigram, columns=['Word', 'Count']))

print("\nThe most frequently occured top 10 bigrams in negative reviews")
#pprint(pd.DataFrame(topkNeg_bigram, columns=['Word', 'Count']))

#2. Mutual Information


# positive = 1 / negative = 0
gtScore = []
for i in range(len(hotelDf)):
    if hotelDf['groundTruth'][i] == 'positive':
        gtScore.append(1)
    else:
        gtScore.append(0)

# let's calculate Mutual Information for unigrams and bigrams
vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english', max_features=500)
countvector = vectorizer.fit_transform(hotelDf['review'])
densevector = np.array(countvector.todense())
    
# miScore_unigram = pd.DataFrame(data = {'word': vectorizer.get_feature_names(),
#              'MI Score': [mutual_info_score(gtScore, densevector[:,i].squeeze()) for i in range(500)]})

miScore_unigram = pd.DataFrame(data =
                            {'MI Score': [mutual_info_score(gtScore, densevector[:,i].squeeze()) for i in range(500)]}
                            , index = vectorizer.get_feature_names())

# Bigram version
vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english', max_features=500)
countvector = vectorizer.fit_transform(hotelDf['review'])
densevector = np.array(countvector.todense())
miScore_bigram = pd.DataFrame(data =
                    {'MI Score': [mutual_info_score(gtScore, densevector[:,i].squeeze()) for i in range(500)]},
                    index = vectorizer.get_feature_names())


miScore_unigram.sort_values('MI Score', inplace=True, ascending=False)

miScore_bigram.sort_values('MI Score', inplace=True, ascending=False)

#3. Pointwise Mutual Information

def getPMI_ngram(df, gt, ngram_range=(1,1), max_features=500):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features)
    countvector = vectorizer.fit_transform(hotelDf['review'])
    densevector = np.array(countvector.todense())
    
    px = sum(df['groundTruth'] == gt) / len(df)
    pmis = []
    
    for i in range(max_features):
        py = sum(densevector[:,i] == 1) / len(df)
        pxy = len(df[(df['groundTruth'] == gt) & (densevector[:,i] == 1)]) / len(df)
        
        if pxy == 0:
            pmi = math.log10((pxy + 0.0001) / (px * py))
        else:
            pmi = math.log10(pxy / (px * py))
            
        pmis.append(pmi)
        
    pmis = pd.DataFrame(data = {'pmi' + gt: pmis}, index = vectorizer.get_feature_names())
    return pmis.sort_values('pmi' + gt, ascending=False)


pmiPos_unigram = getPMI_ngram(hotelDf, 'positive')
pmiNeg_unigram = getPMI_ngram(hotelDf, 'negative')
pmiPos_bigram = getPMI_ngram(hotelDf, 'positive', ngram_range=(2,2))
pmiNeg_bigram = getPMI_ngram(hotelDf, 'negative', ngram_range=(2,2))

print('PMI for positive reviews - Bigram')
print('PMI for positive reviews - Bigram')
print('PMI for negative reviews - Unigram')
print('PMI for negative reviews - Bigram')


pmiPos_unigram.head(10).plot.bar(rot=40, color='b', fontsize=7,
                                title='Top 10 words in Positive Reviews based on PMI scores')
plt.savefig("top10PosUni")
pmiNeg_unigram.head(10).plot.bar(rot=40, color='r', fontsize=7,
                                title='Top 10 words in Negative Reviews based on PMI scores')
plt.savefig("top10NegUni")

pmiPos_bigram.head(10).plot.bar(rot=40, color='b', fontsize=7,
                                title='Top 10 words in Positive Reviews based on PMI scores')
plt.savefig("top10PosBi")
pmiNeg_bigram.head(10).plot.bar(rot=40, color='r', fontsize=7,
                                title='Top 10 words in Negative Reviews based on PMI scores')
plt.savefig("top10NegBi")

"""
plt.xlabel('Rating Score')
hotelDf['ratingScore'].plot(kind='hist', title='Histogram - Rating Scores',
                            bins=np.arange(1,7)-0.5)


plt.xlabel('Vader Sentiment Score')
hotelDf['vaderScore'].plot(kind='hist', title='Histogram - Vader Scores', 
                        xticks=[-1.0, -0.5, 0.0, 0.5, 1.0])


x = [hotelDf['ratingScore'].as_matrix() / 5]
y = [(hotelDf['vaderScore'].as_matrix() + 1 )/ 2]
bins = np.linspace(0, 1, 100)
plt.hist(x, bins, label='Rescaled True Ratings')
plt.hist(y, bins, label='Rescaled Vader Scores')
plt.title('Histogram - Rescaled')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend(loc='upper left')

#Boxplot

#Plot top 5 side-by-side boxplot for top 5 ground truth rated hotel
tp5gthotel = ratingByHotel.sort_values('ratingScore', ascending=False).head(5).hotelName.as_matrix()

tempdf = hotelDf[(hotelDf.hotelName == tp5gthotel[0]) | (hotelDf.hotelName == tp5gthotel[1]) | 
        (hotelDf.hotelName == tp5gthotel[2]) | (hotelDf.hotelName == tp5gthotel[3]) | 
        (hotelDf.hotelName == tp5gthotel[4])]

g = sns.factorplot(kind='box',        # Boxplot
            y='ratingScore',       # Y-axis - values for boxplot
            x='hotelName',        # X-axis - first factor
            data=tempdf,        # Dataframe 
            size=6,            # Figure size (x100px)      
            aspect=1.5,        # Width = size * aspect 
            legend_out=False)  # Make legend inside the plot

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=30) # set new labels

g = sns.factorplot(kind='box',        # Boxplot
            y='vaderScore',       # Y-axis - values for boxplot
            x='hotelName',        # X-axis - first factor
            data=tempdf,        # Dataframe 
            size=6,            # Figure size (x100px)      
            aspect=1.5,        # Width = size * aspect 
            legend_out=False)  # Make legend inside the plot

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticklabels(labels, rotation=30) # set new labels

y = hotelDf['ratingScore'].as_matrix()
x = hotelDf['vaderScore'].as_matrix()
plt.title('Vader score vs. True Ratings')
plt.xlabel('Vader Scores')
plt.ylabel('True Ratings')
plt.xticks([-1, -0.5, 0, 0.5, 1])
plt.yticks([1,2,3,4,5])
plt.plot(x, y, "o", ms=3, color='b')
"""