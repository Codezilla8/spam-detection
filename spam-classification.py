#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv("/home/junix/sms-spam-classifier/spam.csv", encoding="latin-1")
print(df)


# In[ ]:


df.shape


# In[ ]:


# 1. cleeaning the data
# 2. EDA
# 3. Text Preprocessing
# 4. Model Building
# 5. Model Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


# In[ ]:


df.info()


# In[ ]:


df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
df.sample(8)


# In[ ]:


# df["target"] = df["target"].map({"ham": "0", "spam": "1"})
# df.sample(10)


# In[ ]:


# df["target"] = (df["target"] == "ham").astype(int)
# df.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["target"] = encoder.fit_transform(df["target"])
df.head()


# In[ ]:


#checking for missing and duplicate values
print(df.isnull().sum())


# In[ ]:


print(df.duplicated().sum())


# In[ ]:


df = df.drop_duplicates(keep="first")
print(df.duplicated().sum())


# In[ ]:


df.shape


# In[ ]:


df.value_counts()


# In[ ]:


df['target'].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
#autopct is used to show the percentage of each class in the pie chart and %0.2f is used to format the percentage to 2 decimal places


# In[ ]:


#notice rhat the data is imbalanced!


# In[ ]:


pip install nltk


# In[ ]:


import nltk
nltk.download("punkt")


# In[ ]:


#adding a new column for the number of characters in the text message
df["num_characters"] = df["text"].apply(len)
df.head()


# In[ ]:


df["num_words"] = df["text"].apply(lambda x: len(nltk.word_tokenize(x)))
df["num_sentences"] = df["text"].apply(lambda x: len(nltk.sent_tokenize(x)))
df.head()   


# In[ ]:


df.info()


# In[ ]:


df[['num_characters', 'num_words', 'num_sentences']].describe()


# In[ ]:


df[df['target'] == '1'][['num_characters', 'num_words', 'num_sentences']].describe()


# In[ ]:


sns.histplot(df[df['target'] == 0]['num_characters'], color = 'blue', label='Ham')
sns.histplot(df[df['target'] == 1]['num_characters'], color = 'red', label='Spam')
plt.legend()


# In[ ]:


sns.pairplot(df, hue='target')


# In[ ]:


sns.heatmap(df.corr(numeric_only=True), annot=True)


# In[ ]:


#text preprocessing


# In[ ]:


def text_preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(lemmatizer.lemmatize(i))

    return y


# In[ ]:


from nltk.corpus import stopwords
import string


# In[ ]:


text_preprocessing("????????????????????If you're visiting this page, you're likely here because you're")


# In[ ]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[ ]:


df['transformed_text'] = df['text'].apply(text_preprocessing)


# In[ ]:


df.sample(7)


# In[ ]:


from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')


# In[ ]:


pip install wordcloud


# In[ ]:


spam_text = df.loc[df['target'] == 1, 'transformed_text'].fillna('').astype(str).str.cat(sep=' ')
spam_wc = wc.generate(spam_text)


# In[ ]:


plt.imshow(spam_wc)


# In[ ]:


ham_text = df.loc[df['target'] == 0, 'transformed_text'].fillna('').astype(str).str.cat(sep=' ')
ham_wc = wc.generate(ham_text)


# In[ ]:


plt.imshow(ham_wc)


# In[ ]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg:
        spam_corpus.append(word)


# In[ ]:


len(spam_corpus)


# In[ ]:


from collections import Counter
counter_df = pd.DataFrame(Counter(spam_corpus).most_common(30), columns=['word', 'count'])
sns.barplot(data=counter_df, x='word', y='count')
plt.xticks(rotation='vertical')


# In[ ]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg:
        ham_corpus.append(word)


# In[ ]:


len(ham_corpus)


# In[ ]:


from collections import Counter
counter_df = pd.DataFrame(Counter(ham_corpus).most_common(30), columns=['word', 'count'])
sns.barplot(data=counter_df, x='word', y='count')
plt.xticks(rotation='vertical')


# In[ ]:


#Model Building


# In[ ]:


#1. Naive Bayes


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[ ]:


X = cv.fit_transform(df['transformed_text']).toarray()


# In[ ]:




