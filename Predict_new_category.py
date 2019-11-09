# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 00:26:19 2019

@author: ASH
"""

# Importing Libraries
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset1 = pd.read_excel("Data_Train.xlsx")
dataset2 = pd.read_excel("Data_Test.xlsx")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

processed_story = []
for i in range(len(dataset1)):
    story = re.sub('@[\w]*', ' ', dataset1['STORY'][i])
    story = re.sub('^a-zA-Z#', ' ', story)
    story = re.sub(r'[^\w\s]', ' ', story)
    #story = re.sub("^\d+\s|\s\d+\s|\s\d+$", ' ', story)
    story = re.sub(' [\d]*', ' ', story)
    story = story.lower()
    story = story.split()
    story = [ps.stem(token) for token in story if not token in stopwords.words('english')]
    story = ' '.join(story)
    processed_story.append(story)

import language_check
tool = language_check.LanguageTool('en-US')

#error_free_processed_story = []
#for i in range(len(dataset)):
#    matches = tool.check(processed_story[i])
#    temp = language_check.correct(processed_story[i], matches)
#    error_free_processed_story.append(temp)

error_free_processed_story = [language_check.correct(processed_story[i], tool.check(processed_story[i])) for i in range(len(dataset1))]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)

X1 = cv.fit_transform(processed_story).toarray()
X2 = cv.fit_transform(error_free_processed_story).toarray()
y = dataset1['SECTION'].values

print(cv.get_feature_names())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()

n_b.fit(X_train, y_train)
n_b.score(X_train, y_train)

n_b.fit(X_test, y_test)
n_b.score(X_test, y_test)

n_b.fit(X1, y)
n_b.score(X1, y)

n_b.fit(X2, y)
n_b.score(X2, y)

processed_story2 = []
for i in range(len(dataset2)):
    story = re.sub('@[\w]*', ' ', dataset2['STORY'][i])
    story = re.sub('^a-zA-Z#', ' ', story)
    story = re.sub(r'[^\w\s]', ' ', story)
    #story = re.sub("^\d+\s|\s\d+\s|\s\d+$", ' ', story)
    story = re.sub(' [\d]*', ' ', story)
    story = story.lower()
    story = story.split()
    story = [ps.stem(token) for token in story if not token in stopwords.words('english')]
    story = ' '.join(story)
    processed_story2.append(story)

#error_free_processed_story2 = []
#for i in range(len(dataset2)):
#    matches = tool.check(processed_story2[i])
#    temp = language_check.correct(processed_story2[i], matches)
#    temp = temp.lower()
#    error_free_processed_story2.append(temp)

error_free_processed_story2 = [language_check.correct(processed_story2[i], tool.check(processed_story2[i])).lower() for i in range(len(dataset2))]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)

X3 = cv.fit_transform(processed_story2).toarray()
X4 = cv.fit_transform(error_free_processed_story2).toarray()

y_pred1 = n_b.predict(X3)
y_pred2 = n_b.predict(X4)

#error_free_processed_story3 = []
#dataset3 = list(dataset2.values.flatten())
#dataset3 = [list(l) for l in zip(*dataset2.values)]
#for i in range(len(dataset2)):
#    matches = tool.check(dataset3[i])
#    temp = language_check.correct(dataset3[i], matches)
#    temp = temp.lower()
#    error_free_processed_story3.append(temp)

df1 = pd.DataFrame(processed_story2)
df2 = pd.DataFrame(y_pred1)

df = pd.concat([df1, df2], axis = 1, ignore_index = True)

import xlsxwriter

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()