# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 19:34:25 2022

@author: vikas
"""


#Case Study Loan Data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('25_LogR\\loan_data.csv')
df.columns
df.dtypes
df['purpose'].value_counts()


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['purpose'] = le.fit_transform(df['purpose'].values)
df.dtypes
df['purpose'].value_counts()


x = df.drop('not.fully.paid', axis=1).values
y = df['not.fully.paid'].values



from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))



from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))





from imblearn.over_sampling import SMOTE
sm = SMOTE()
xf, yf = sm.fit_resample(x,y)

pd.Series(yf).value_counts()



from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
xs = ss.fit_transform(xf)
ys = yf


from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(xs, ys, test_size=0.2)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))



from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
model = QuadraticDiscriminantAnalysis()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(accuracy_score(yte, yp))

print(precision_score(yte, yp))

print(recall_score(yte, yp))

print(f1_score(yte, yp))



from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
scores = cross_val_score(model, xs, ys, cv=10)
scores
sd = np.std(scores)
sd


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
scores = cross_val_score(model, xs, ys, cv=5)
scores
sd = np.std(scores)
sd




# Water Dataset
#Case Study Loan Data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('DrinkingWater_Final_Dataset.csv')
df.columns
df.dtypes

df= df.drop('id', axis=1)
df['Potability'].value_counts()

x = df.drop('Potability', axis=1).values
y = df['Potability'].values


from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, xtr, ytr, cv=10)
scores
sd = np.std(scores)
sd


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
scores = cross_val_score(model, xtr, ytr, cv=10)
scores
sd = np.std(scores)
sd





from imblearn.over_sampling import SMOTE
sm = SMOTE()
xf, yf = sm.fit_resample(x,y)

pd.Series(yf).value_counts()



from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
xs = ss.fit_transform(xf)
ys = yf


from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(xs, ys, test_size=0.2)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))



from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
model = QuadraticDiscriminantAnalysis()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(xtr, ytr)
yp = model.predict(xte)

from sklearn.metrics import classification_report
print(classification_report(yte, yp))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(accuracy_score(yte, yp))

print(precision_score(yte, yp))

print(recall_score(yte, yp))

print(f1_score(yte, yp))



from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
scores = cross_val_score(model, xs, ys, cv=10)
scores
sd = np.std(scores)
sd


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
scores = cross_val_score(model, xs, ys, cv=5)
scores
sd = np.std(scores)
sd





# Unsupervised Learning
#clustering

import matplotlib.pyplot as plt
import pandas as pd

data = {'x1': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'x2': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }
  
df = pd.DataFrame(data,columns=['x1','x2'])
print (df)

plt.scatter(df['x1'], df['x2'])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df)

centroids = kmeans.cluster_centers_
centroids

kmeans.labels_


plt.scatter(df['x1'], df['x2'], c=kmeans.labels_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(df)

centroids = kmeans.cluster_centers_
centroids

kmeans.labels_


plt.scatter(df['x1'], df['x2'], c=kmeans.labels_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")


# Elbow or Knee Method


sse=[]

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
sse



plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow




from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

centroids = kmeans.cluster_centers_
centroids

kmeans.labels_


plt.scatter(df['x1'], df['x2'], c=kmeans.labels_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")


df['label'] = kmeans.labels_
df

xs = df.drop('label', axis=1).values
ys = df['label'].values

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
scores = cross_val_score(model, xs, ys, cv=10)
scores
sd = np.std(scores)
sd



#Case Study Drinking Water

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('DrinkingWater_Clustering.csv')

df.columns
df.dtypes



x = df.values

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x = ss.fit_transform(x)



from sklearn.decomposition import PCA
pca = PCA(4)
x = pca.fit_transform(x)

from sklearn.cluster import KMeans

sse=[]
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    sse.append(kmeans.inertia_)
sse


plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)

df['label'] = kmeans.labels_

df.to_csv('labeledDWSSPCA.csv')


#Case Food Odering NonNumeric

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('28_Clustering//FoodOrder.csv')

df.columns
df.dtypes

df1 = pd.read_csv('28_Clustering//FoodOrder.csv')

from sklearn import preprocessing
col = df.select_dtypes(include=['object']).columns

for c in col:
    le = preprocessing.LabelEncoder()
    df[c] = le.fit_transform(df[c].values)

df.dtypes
df.columns

x = df.drop(['Cust_Id'], axis=1).values


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x = ss.fit_transform(x)

from sklearn.cluster import KMeans
sse=[]
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    sse.append(kmeans.inertia_)
sse


plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

df
df1['label'] = kmeans.labels_

df1.to_csv('FoodOrderCluster.csv')

df1[df1['label']==0]


l1 = [1,2,3,4]
l2 = l1
l1.pop()
l2.append(10)

l3 = l1.copy()
l1.append(22)


# Apriori

from efficient_apriori import apriori

transactions = [('eggs', 'bacon', 'soup', 'milk'), 
                ('eggs', 'bacon', 'apple', 'milk'), 
                ('soup', 'bacon', 'banana')]

itemsets, rules = apriori(transactions)

rules

for r in rules:
    print (r)



import numpy as np
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

transactions = [['milk', 'water'], ['milk', 'bread'],
                ['milk','bread','water']]
transactions


te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
te_ary

df = pd.DataFrame(te_ary, columns=te.columns_)
df

frequent_itemsets = apriori(df, min_support=0.0000001, use_colnames = True)
frequent_itemsets

pd.set_option('display.max_columns',None)

res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.000001)

res.columns

res = res[['antecedents', 'consequents','support', 'confidence', 'lift']]
res


l1 = [[1,3,2,4,5], [4,36,3], [1], [5,4,32,7,5,43,2]]

l1


#Case Study Store

import numpy as np
import pandas as pd
from efficient_apriori import apriori

store_data = pd.read_csv('29_Apriori\\store_data1.csv', header=None)
store_data.head()


records = []

for i in range(0, len(store_data)):

    print(i)    
    '''
    l = lambda x: x!='nan'
    
    records.append(list(filter(l ,store_data.values[i,:])))
    '''
    '''
    row=[]
    for j in range(0, 20):
        if str(store_data.values[i,j]) != 'nan']:
        row.append(str(store_data.values[i,j]))
    records.append(row)
    
    '''
    
    records.append([str(store_data.values[i,j]) for j in range(0, 20) if str(store_data.values[i,j]) != 'nan'])

records



itemsets, rules = apriori(records, min_support=0.01, min_confidence=0.01)

for r in rules:
    print (r)


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
te_ary

df = pd.DataFrame(te_ary, columns=te.columns_)
df

frequent_itemsets = apriori(df, min_support=0.05, use_colnames = True)
frequent_itemsets

pd.set_option('display.max_columns',None)

res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.000001)

res.columns

res = res[['antecedents', 'consequents','support', 'confidence', 'lift']]
res


#Online Store Case Study

import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



df = pd.read_csv('29_Apriori\\online_store.csv')

df['Description'] = df['Description'].str.strip()


df.dropna(axis=0, subset=['InvoiceNo'], inplace=True) #remove NA invoice nos

df.dtypes

df['InvoiceNo'] = df['InvoiceNo'].astype('str')


df = df[~df['InvoiceNo'].str.contains('C')]
df


basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
basket.head()


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

basket_sets.head()

basket_sets.columns

basket_sets.drop('POSTAGE', inplace=True, axis=1) #remove POSTAGE column not an item

frequent_itemsets = apriori(basket_sets, min_support=0.03,  use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="lift")
rules[['antecedents','consequents', 'support', 'confidence', 'lift']].to_csv('data.csv')

#%%
rules[ (rules['confidence'] >= 0.8) ]

basket['ALARM CLOCK BAKELIKE GREEN'].sum()

basket['ALARM CLOCK BAKELIKE RED'].sum()






































