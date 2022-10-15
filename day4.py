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

data.groupby(['CustomerID']).aggregate({'InvoiceDate':max})

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

rfmAnalysis[(rfmAnalysis=='331')]



#Matplotlib

import matplotlib.pyplot as plt

Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
Unemployment_Rate1 = [1.8,2,8,4.2,6.0,7,3.5,5.2,7.5,5.3]
Year
Unemployment_Rate


plt.plot(Year,Unemployment_Rate, color= 'g', marker='o', label='UR1')
plt.plot(Year,Unemployment_Rate1, color = 'r', marker='<', label='UR2')
plt.title('Year Vs UR', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('UR', fontsize=14)
plt.grid()
plt.legend()
plt.show()




import pandas as pd
Data = {'Year': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010], 'Unemployment_Rate': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]}
Data  
df = pd.DataFrame(Data,columns=['Year','Unemployment_Rate'])
df  
plt.plot(df['Year'],df['Unemployment_Rate'], color= 'g', marker='o', label='UR1')
plt.title('Year Vs UR', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('UR', fontsize=14)
plt.grid()
plt.legend()
plt.show()



Country = ['USA','Canada','Germany','UK','France']
GDP_Per_Capita = [45000,42000,52000,49000,47000]

plt.bar(Country, GDP_Per_Capita)
plt.title('Country Vs GDP Per Capita')
plt.xlabel('Country')
plt.ylabel('GDP Per Capita')
plt.grid()
plt.show()


New_Colors = ['green','blue','purple','brown','teal']
plt.bar(Country, GDP_Per_Capita, color=New_Colors)
plt.title('Country Vs GDP Per Capita', fontsize=14)
plt.xlabel('Country', fontsize=14)
plt.ylabel('GDP Per Capita', fontsize=14)
plt.grid(True)
plt.show();


nc = ['#1f4ea5', '#8ddca4', '#63326e', '#291711']
plt.bar(Country, GDP_Per_Capita, color=nc)
plt.title('Country Vs GDP Per Capita', fontsize=14)
plt.xlabel('Country', fontsize=14)
plt.ylabel('GDP Per Capita', fontsize=14)
plt.grid(True)
plt.show();


#Scatter

import numpy as np
x = list(range(1,11))
y = np.random.randint(10,20, size=10)

x
y

plt.scatter(x,y)


from pydataset import data
mtcars = data('mtcars')
mtcars.columns

mt = mtcars[['mpg','hp', 'disp', 'gear']]
mt

plt.scatter(mt['hp'], mt['mpg'], s=mt['disp'])

plt.scatter(mt['hp'], mt['mpg'], s=mt['disp'], c = mt['gear'])

mtcars.head(1)


import matplotlib.pyplot as plt
import seaborn as sns

sns.set() #default settings
tips_df = sns.load_dataset('tips')
tips_df
tips_df.columns
tips_df.total_bill
total_bill = tips_df.total_bill.to_numpy()
tip = tips_df.tip.to_numpy()
tip
plt.scatter(total_bill, tip)
plt.show();

plt.scatter(total_bill, tip)
plt.title(label='Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show();






sc = plt.scatter(mt['hp'], mt['mpg'], sizes=mt['disp'])
handles, labels = sc.legend_elements(prop='sizes')
plt.legend(handles, labels)
plt.show();




mtcars
scatter = plt.scatter('wt', 'mpg', s='hp', data=mtcars)
handles, labels = scatter.legend_elements(prop='sizes')
plt.xlabel('wt')
plt.ylabel('mpg')
plt.legend(handles, labels)
plt.show();



import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows = 2)

ax[0].scatter(mtcars['wt'],mtcars['disp'])
ax[1].scatter(mtcars['wt'],mtcars['mpg'])



fig, ax = plt.subplots(nrows = 2)

ax[0].scatter(mtcars['wt'],mtcars['disp'])
ax[1].plot(mtcars['wt'],mtcars['mpg'])



fig, ax = plt.subplots(ncols=2)
ax[0].scatter(mtcars['wt'],mtcars['disp'])
ax[1].scatter(mtcars['wt'],mtcars['mpg'])


fig, ax = plt.subplots(nrows=2,ncols=2)
ax.shape
ax[0,0].scatter(mtcars['wt'],mtcars['disp'])
ax[0,1].scatter(mtcars['wt'],mtcars['mpg'])
ax[1,0].scatter(mtcars['wt'],mtcars['disp'])
ax[1,1].scatter(mtcars['wt'],mtcars['mpg'])



import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



# Histogram


import pandas as pd
import numpy as mp
import matplotlib.pyplot as plt

data1 = np.random.normal(100, 10, 100000)

data1

np.max(data1)
np.min(data1)

np.mean(data1)
np.median(data1)
np.std(data1)

l1 = [9,3,2,4,6,5,1,8,7]

np.mean(l1)
np.median(l1)

plt.hist(data1)



data2 = np.random.randint(50,150, 100000)
plt.hist(data2)


data3 = np.random.binomial(2, .8, 100)
plt.hist(data3)

'''
data3 = np.random.multinomial(2, [0.3, 0.7])
plt.hist(data3)
'''


import matplotlib.pyplot as plt
import numpy as np

data = np.random.randint(1,100, size=1000)
plt.boxplot(data)
np.mean(data), np.median(data), np.std(data), min(data), max(data), 
plt.hist(data)


data = np.random.normal(50,30, size=1000)
plt.boxplot(data)
data
np.mean(data), np.median(data), np.std(data), min(data), max(data), 
plt.hist(data)




import matplotlib.pyplot as plt
labels = ['Male',  'Female']
percentages = [60, 40]
explode=(0.15,0)
#

color_palette_list = ['#f600cc', '#ADD8E6', '#63D1F4', '#0EBFE9', '#C1F0F6', '#0099CC']

fig, ax = plt.subplots(dpi=300)
ax.pie(percentages, explode=explode, labels=labels, colors= color_palette_list, autopct='%0.2f%%',  shadow=True, startangle=90,  pctdistance=1.2, labeldistance=1.6)
ax.axis('equal')
ax.set_title("Distribution of Gender in Class", y=1)
ax.legend(frameon=False, bbox_to_anchor=(0.2,0.8))
plt.show()



#Regression

import numpy as np
import matplotlib.pyplot as plt

y = np.random.normal(1, 11, size=10)
y
x = np.arange(1,11)
x
x.shape

x = x.reshape((-1,1))
x
x.shape
plt.scatter(x,y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x,y)

slope = model.coef_
intercept = model.intercept_
intercept

ypred  = model.predict(x)

plt.scatter(x,y)
plt.scatter(x,ypred)
plt.plot(x,ypred)

r2 = model.score(x,y)
r2




model = LinearRegression()
model.fit(x[:-2],y[:-2])

slope = model.coef_
intercept = model.intercept_
intercept

ypred  = model.predict(x[:-2])

plt.scatter(x[:-2],y[:-2])
plt.scatter(x[:-2],ypred)
plt.plot(x[:-2],ypred)

r2 = model.score(x[:-2],y[:-2])
r2



#Case Housing

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('24_LR//Housing.csv')
df =df.dropna()
df.columns

x = df['area'].values.reshape((-1,1))
y = df['price'].values
x.shape
y.shape

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)

r2 = model.score(x,y)
r2

ypred = model.predict(x)

plt.scatter(x,y)
plt.scatter(x,ypred)

r = np.linspace(min(y), max(y), 10)
r

plt.boxplot(y)
plt.yticks(r)

val = 0.87*10000000

len(y)
len(y[y<val])

plt.boxplot(y[y<val])



# Outlier Removed Case Housing

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('24_LR//Housing.csv')
df =df.dropna()
df.columns
val = 0.87*10000000
df = df[df['price']<val]

x = df['area'].values.reshape((-1,1))
y = df['price'].values

plt.boxplot(y)


x.shape
y.shape

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)

r2 = model.score(x,y)
r2

ypred = model.predict(x)

plt.scatter(x,y)
plt.scatter(x,ypred)




# Data Split

from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x,y, test_size=0.3)
x.shape
xtr.shape, ytr.shape, xte.shape, yte.shape

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtr,ytr)

r2 = model.score(xte,yte)
r2

ypred = model.predict(xte)

plt.scatter(xtr,ytr)
plt.scatter(xte,yte)
plt.scatter(xte,ypred)


#Case Real estate.csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('24_LR//Real estate.csv')
df =df.dropna()
df.columns

x = df.drop(['n', 'Y house price of unit area'], axis=1).values
y = df['Y house price of unit area'].values
x.shape
y.shape

from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x,y, test_size=0.3)
x.shape
xtr.shape, ytr.shape, xte.shape, yte.shape

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtr,ytr)

r2 = model.score(xte,yte)
r2

ypred = model.predict(xte)
ypred


df.columns


xval = ['X1 transaction date', 'X2 house age',
       'X3 distance to the nearest MRT station',
       'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']

import seaborn as sns
for xv in xval:
    sns.residplot(data=df, x=xv, y='Y house price of unit area')
    plt.show()



for xv in xval:
    sns.scatterplot(data=df, x=xv, y='Y house price of unit area')
    plt.show()


for xv in xval:
    model = LinearRegression()
    model.fit(df[xv].values.reshape((-1,1)), df['Y house price of unit area'].values)
    print(model.score(df[xv].values.reshape((-1,1)), df['Y house price of unit area'].values))




#Feature Selection through PCA

from sklearn.decomposition import PCA

x = df.drop(['n', 'Y house price of unit area'], axis=1).values
y = df['Y house price of unit area'].values
x.shape
y.shape

pca = PCA(4)
'''
pca.fit(x, y)
pca.explained_variance_
x = pca.transform(x)
pca.explained_variance_
'''
xp = pca.fit_transform(x,y)
xp

from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(xp,y, test_size=0.3)
xp.shape
xtr.shape, ytr.shape, xte.shape, yte.shape

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtr,ytr)

r2 = model.score(xte,yte)
r2



# Case MTcars

from pydataset import data

mt = data('mtcars')

mt.columns

mt.head(2)


y = mt['mpg'].values
x = mt.drop('mpg', axis=1).values


xval = mt.drop('mpg',axis=1).columns


import seaborn as sns
for xv in xval:
    sns.residplot(data=mt, x=xv, y='mpg')
    plt.show()


for xv in xval:
    sns.scatterplot(data=mt, x=xv, y='mpg')
    plt.show()


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtr,ytr)

r2 = model.score(x,y)
r2

y
xpred = np.array([  10.   , 321.   , 305.   ,   2.54 ,   3.07 ,  12.6  ,   1.   ,
   1.   ,   7.   ,   6.   ]).reshape((1,-1))

x.shape
xpred.shape

yp = model.predict(xpred)
yp


# DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtr,ytr)
r2 = model.score(x,y)
r2

#Randomforest Regressor


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(xtr,ytr)
r2 = model.score(x,y)
r2



# Classification
#Logistic Regression

X= list(range(-6,7))
X

y = []

import math

for x in X:
    y.append((1)/(1 + math.exp(-x)))
y


import matplotlib.pyplot as plt

plt.scatter(X,y)






import numpy as np
x= np.array(list(range(-6,7,1))).reshape((-1,1))
x
x.shape

y = np.array([0,0,0,0,1,1,0,0,1,1,1,1,1])
y.shape

plt.scatter(x,y)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x,y)

yp = model.predict(x)

plt.scatter(x,y)
plt.scatter(x,yp)



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x,y)
yp = model.predict(x)

plt.scatter(x,y)
plt.scatter(x,yp)

ypp = model.predict_proba(x)
ypp.shape


plt.scatter(x,y)
plt.scatter(x,ypp[:,1])
plt.scatter(x,yp)


#Case Titanic

'''
y  = [1,1,1,0,1,0,1,1,1,0]
yp = [1,1,1,0,1,0,1,1,1,0]


TP = 1 - 1 Accepted
TN = 0 - 0 Accepted
FP = 0 - 1 Error
FN = 1 - 0 Error

y  = [1,0,1,0,1,0,1,0,1,0]
yp = [1,0,0,1,1,0,1,0,1,0]
from sklearn.metrics import classification_report

print(classification_report(y,yp))


y  = [1,0,1,1,1,1,1,1,1,0]
yp = [1,1,1,1,1,1,1,1,1,1]

print(classification_report(y,yp))

y  = [0,0,0,0,0,0,0,0,0,0]
yp = [1,1,1,1,1,1,1,1,1,1]
print(classification_report(y,yp))

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('25_LogR/titanic.csv')
df.columns

x = df.drop('survived', axis=1).values
y = df['survived'].values

from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x,y, test_size=0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(xtr, ytr)
yp = model.predict(xte)
yp

from sklearn.metrics import classification_report

print(classification_report(yte,yp))



#HR analytics Case

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('25_LogR/HRNum.csv')
df.columns

x = df.drop('Attrition', axis=1).values
y = df['Attrition'].values

from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x,y, test_size=0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(xtr, ytr)
yp = model.predict(xte)
yp

from sklearn.metrics import classification_report

print(classification_report(yte,yp))



#Class Imbalancing

df['Attrition'].value_counts()


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
xs, ys = oversample.fit_resample(x, y)

pd.Series(ys).value_counts()


from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(xs,ys, test_size=0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(xtr, ytr)
yp = model.predict(xte)
yp

from sklearn.metrics import classification_report

print(classification_report(yte,yp))
















































