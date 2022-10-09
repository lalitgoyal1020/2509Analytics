# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:42:55 2022

@author: vikas
"""


#grouping


import pandas as pd

rno = pd.Series(range(1,100001))

name = pd.Series(['Student'+str(i) for i in range(1,100001)])
name

import numpy as np
gender = np.random.choice(['M','F'], size=100000)
gender

city = np.random.choice(['Delhi', 'Mumbai', 'Chennai', ' Chandigarh'], size=100000)

courses = np.random.choice(['BBA', 'MBA', 'BTECH', 'MTECH'], size=100000)
courses

marks1 = np.random.randint(0,101, size=100000)

marks2 = np.random.randint(0,101, size=100000)


df = pd.DataFrame({'rno':rno, 'name':name, 'gender':gender, 'courses':courses, 'city':city,
                    'marks1':marks1, 'marks2':marks2})

df.describe()

df.dtypes
df.count()


df.head(5)

#Groupby


# size() - Aggregate Function

df.groupby(['gender']).size()

df.groupby(['courses']).size()

df.groupby(['gender', 'courses']).size()

df.groupby([ 'courses', 'gender']).size()

df.head(3)

df.groupby([ 'courses', 'gender', 'city']).size()

df.groupby([ 'courses', 'city']).count()

df.groupby([ 'gender']).aggregate({'marks1':np.mean})

df.groupby([ 'gender']).aggregate({'marks1':[np.mean, np.max, np.std, np.min]})

res = df.groupby([ 'gender']).aggregate({'marks1':[np.mean, np.max, np.std, np.min], 
                                   'marks2':[np.mean, np.max, np.std, np.min]})

df.columns
res.to_csv("groupbyRes.csv")

df[[ 'courses', 'city', 'rno']].groupby([ 'courses', 'city']).count()



#Pivot Table
df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                         "small", "large", "small", "small",
                         "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
df

table = pd.pivot_table(df, values=['D','E'], index=['A'],
                    columns=['C','B'], aggfunc=np.sum)

table


table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
                    aggfunc={'D': np.mean,
                             'E': [min, max, np.mean]})

table



df.to_excel('data.xlsx')

writer
with pd.ExcelWriter('data1.xlsx') as writer:
    df.to_excel(writer, sheet_name='Data')
    table.to_excel(writer, sheet_name='Pivot')
    


df1 = pd.read_excel('data1.xlsx', sheet_name = 'Pivot')
df1

df1 = pd.read_excel('data1.xlsx', sheet_name = 'Data')
df1



#JSON

import json

df1 = pd.DataFrame([["a", "b"], ["c", "d"]], columns=['A','B'],
                   index=['C','D'])
df1


df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                         "small", "large", "small", "small",
                         "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
df


df1 = pd.DataFrame([["a", "b"], ["c", "d"]], columns=['A','B'],
                   index=['C','D'])
df1

df1.to_json( orient = 'split', index = False)
df1.to_json( orient = 'split', index = True)

df1.to_json()
df1.to_json('file1.json', orient = 'split', index = 'false')

df2 = pd.read_json('file1.json', orient ='split')
df2



#Denco Case Study

import pandas as pd

df = pd.read_csv('20denco.csv')
df.dtypes
df['partnum'] = df['partnum'].astype('object')
df.dtypes
df.describe()
df.count()
np.sum(df.isnull())


#most loyal cust, most frequent visitor

df.columns
df.groupby(['custname']).size()
df.groupby(['custname']).size().sort_values(ascending=False)
df.groupby(['custname']).size().sort_values(ascending=False).head(5)

df.groupby(['custname']).aggregate({'revenue':sum}).sort_values(ascending=False, by='revenue').head(5)

df.columns

df.groupby(['partnum']).aggregate({'revenue':sum}).sort_values(ascending=False, by='revenue').head(5)

df.groupby(['partnum']).aggregate({'margin':sum}).sort_values(ascending=False, by='margin').head(5)



#RFM Analysis

import pandas as pd

data = pd.read_csv('RFM//OnlineRetail.csv', parse_dates=['InvoiceDate'],
                   encoding= 'unicode_escape')


data.columns
data.dtypes
data
pd.set_option('display.max_columns', None)
data

data= data[pd.notnull(data['CustomerID'])]
data

data.Country.value_counts().plot(kind='bar')


data=data[['CustomerID','InvoiceDate','InvoiceNo','Quantity','UnitPrice']]

data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

data.head(5)

data= data.drop(['Quantity','UnitPrice'], axis=1)

data.head(5)

data = data[~(data['InvoiceNo'].str.contains('InvoiceNo')=='C') | ~(data['InvoiceNo'].str.contains('InvoiceNo')=='c')]

data


data.dtypes
data['CustomerID'] = data['CustomerID'].astype('int').astype('object')
data.dtypes

#RFM

data.groupby('InvoiceNo').size()

data.groupby('CustomerID').size()

# Recency

import datetime as dt

PRESENT = dt.datetime(2011,12,10)
data.columns

rec = lambda date: (PRESENT - date.max()).days
fre = lambda num: len(num)
mon = lambda price: price.sum()
rfm = data.groupby('CustomerID').agg({'InvoiceDate': rec,
                                'InvoiceNo':fre,
                                'TotalPrice':mon})
rfm
rfm.columns=['Recency', 'Frequency','Monetory']
rfm.head()

rfm['rq'] = pd.qcut(rfm['Recency'],3, ['1','2','3'] )
rfm[['Recency','rq']]

rfm['fq'] = pd.qcut(rfm['Frequency'],3, ['3','2','1'] )
rfm[['Frequency','fq']]

rfm['mq'] = pd.qcut(rfm['Monetory'],3, ['3','2','1'] )
rfm[['Monetory','mq']]


rfm.head()

rfmAnalysis = rfm['rq'].astype(str) + rfm['fq'].astype(str) + rfm['mq'].astype(str)

rfmAnalysis.sort_values()

rfmAnalysis[~(rfmAnalysis=='333')]






























