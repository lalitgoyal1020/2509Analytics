# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 19:33:55 2022

@author: vikas
"""

#!pip install numpy

import numpy as np

a = np.random.randint(100)
a

b = np.random.randint(10, 100)
b

n1 = np.random.randint(0, 10, size=10)
n1

n1.shape

#Numpy array is Homogeneous

n2 = np.random.randint(0, 10, size=(5,4))
n2

n2.shape

n3 = np.random.randint(0,10, size=(3,4,5))
n3

# Indexing

n1 = np.random.randint(10,20, size=10)
n1

n1[0]
n1[1]
n1[0:4]
n1[:4]
n1
n1[4:]
n1[-1]
n1[-2]
n1[-3:]
n1[-5:-2]


n2 = np.random.randint(5, 15, size=(4,5))

n2
n2[0]
n2[0:2]
n2[0][0]
n2[0,0]
n2[0,0:2]

n2[0][-2:]
n2


n2[1]
n2[1][1:3]

n2
n2[1:3,1:3]

n2
n2[2:, -2:]
n2
n2[:,-1]


n3 = np.random.randint(10, 30, size=(3,4,5))

n3


n3[0]
n3[1]
n3[2]

n3[2,-1,-2:]



nr1 = np.arange(10)
nr1

nr1 =np.arange(10,20)
nr1

nr2 = np.arange(10, 40, 5)
nr2

nr2.shape

nr2 = np.arange(10, 30)
nr2
nr2.shape

nr3 = nr2.reshape((5,4))
nr3

nr4 = nr2.reshape(20)
nr4


l1 = [20,30,40]
l2 = [40,50,30]

l3 = l1+l2
l3

np1 = np.array(l1)
np2 = np.array(l2)

np1+np2
np1*np2

np.multiply(np1,np2)
np.sum(np1)


np1 = np.random.randint(10,20, size=(5,6))
np1

npbase = np.zeros((5,6))
npbase


np1 + npbase


type((5,6))

npb1 = np.ones((5,6))
npb1

np1*npb1

npb2 = np.eye(5,6)

npb2

npb2*np1


'''
np3 = np.empty((5,6))
np3
'''

np.linspace(0,10, num=5)


np5 = np.random.randint(10, 30, size=(6,8))
np5

np.sum(np5)

np.sum(np5, axis=0)

np.sum(np5, axis=1)

np.mean(np5)

np.mean(np5, axis=0)

np.mean(np5, axis=1)

#aggregation

np.median(np5)

np.std(np5)

np.var(np5)



np1 = np.random.randint(10,20,size=(5,4))
np1

np2 = np.random.randint(30,50,size=(5,4))
np2

np1
np2
np.concatenate ((np1,np2), axis=0)

np3 = np.concatenate ((np1,np2), axis=1)


np.floor([1.2,1.6])
np.ceil([1.2,1.6])
np.trunc([1.2,1.6])
np.round([1.2,1.6])


np.floor([-1.2,-1.6])
np.ceil([-1.2,-1.6])
np.trunc([-1.2,-1.6])
np.round([-1.2,-1.6])

np.round(1.337764,4)


np3.shape

np3 = np.random.randint(10,20, size=(10,6))
np4 = np.split(np3, 2)
np4

np4 = np.split(np3, 5)
np4

'''
np4 = np.split(np3, [5,1])
np4
np4[2].shape
'''



np3
np3>10
np3
np.all(np3>2)
np.any(np3>10)

np3.shape
np3>10
np.sum(np3>18)

np.sum(np3>18, axis=1)
np.sum(np3>18, axis=0)

np.sum( (np3 > 10) & (np3 < 15))


np4 = np.random.randint(10,20, size=(4,6))
np4.shape
np5 = np4.reshape((1,-1))

np5

np5.shape

np5 = np4.reshape((-1,1))
np5.shape

np5

np4.shape

np4
np5 = np.transpose(np4)
np5


np.sort(np5)


#Pandas


'''
import pandas as pd
import glob

files = glob.glob('F:\\ML-Lab\\2509Analytics\\excel\\*.*')
files

df1 = pd.read_csv(files[0])
for f in files[1:]:
    df = pd.read_csv(f)
    df1 = pd.concat([df,df1])

df1.to_csv('F:\\ML-Lab\\2509Analytics\\excel\\final.csv')
'''

import pandas as pd

#!pip install pydataset

from pydataset import data
data('')
mt = data('mtcars')
mt
type(mt)
mt
mt.to_csv('F:\\ML-Lab\\2509Analytics\\mtcars.csv')


import pandas as pd

df = pd.read_csv('mtcars.csv')

df.columns
df.head(3)
df.head()
df.tail()
df.tail(3)
df.dtypes
df.describe()


# Pandas Series

list(range(11,21))

s = pd.Series(range(11,21))
s


s = pd.Series([22,33,55,66,77])
s


s = pd.Series([22,33,55,66,77], index=[3,5,7,6,8])
s


s = pd.Series([22,33,55,66,77], index=[3,5,7,6,7])
s
s[7]


s = pd.Series([22,33,55,66,77], index=['a','b','c','a','b'])
s['a']
s.loc['a']
s
s.iloc[0]
s.iloc[3]
s.iloc[4]


s.values
s.index

s>55

s[s>55]

s[(s>30) & (s<56)]
s

s['a']=99
s



import pandas as pd

course = pd.Series(['BTech', 'MTech', 'MBA', 'BBA'])
course
strength = pd.Series([100,130,200,150])
strength
fees = pd.Series([10, 15, 13, 20])
fees

d1 = {'Course':course, 'Strength':strength, 'Fees':fees}
d1

df1 = pd.DataFrame(d1)
df1

df1 = pd.DataFrame({'Course':course, 'Strength':strength, 'Fees':fees})
df1


df1
df1.columns
df1.index

df1.index = df1['Course']
df1

df1 = df1.drop(['Course'], axis=1)
df1

df1 = df1.drop(['Course'], axis='columns')
df1

type(df1['Strength'])

type(df1[['Strength','Fees']])

df1.values

df1.columns

df1.index


df1['Strength']>140
df1
df1[df1['Strength']>140]

df1.describe()

df1.count()


df1.index=='BBA'

df1[df1.index=='BBA']




import pandas as pd
import numpy as np

placed = pd.Series([None,np.nan, 100, None])
placed

df1 = pd.DataFrame({'Course':course, 'Strength':strength, 'Fees':fees})
df1
df1['Placed'] = placed
df1

df1.dropna(axis=0)
df1.dropna(axis='rows')

df1.dropna(axis=1)
df1.dropna(axis='columns')

df1.count()


df1.sum()
df1.max()
df1.min()

df1.isnull()

type(df1.isnull())

np.sum(df1.isnull().values, axis=0)

import pandas as pd
pd4 = pd.DataFrame([['dhiraj', 50, 'M', 10000, None], ['Vikas', None, None, None, None], ['kanika', 28, None, 5000, None], ['tanvi', 20, 'F', None, None], ['poonam',45,'F',None,None],['upen',None,'M',None, None]])
pd4
pd4.dropna(axis=0)
pd4
pd4.dropna(axis=1)

pd4
pd4.dropna(axis=1, how='all')
pd4
pd4.dropna(axis=1, how='any')

pd4
pd4.dropna(axis=1, thresh=3)



pd4.fillna(0)
pd4
pd4.dtypes
pd4.fillna('ABC')
pd4.fillna('ABC').dtypes



import pandas as pd

df =pd.read_csv('AirPassengers.csv')
df

df.plot()

df.fillna(0).plot()


df.fillna(method='ffill').plot()

df.fillna(method='bfill').plot()


df1 = df[1:10]

df1

df1 = df1.fillna(method='ffill')
df1

df1.fillna(method='bfill')




grades1 = {'subject1': ['A1','B1','A2','A3'],'subject2': ['A2','A1','B2','B3']   }
grades1
df1 = pd.DataFrame(grades1)
df1

grades2 = {'subject2': ['A1','B1','A2','A3'],'subject4': ['A2','A1','B2','B3']}
df2 = pd.DataFrame(grades2)
df2


pd.concat([df1,df2])



teacher1 = {'rno': [1,2,3,4],'subject2': ['A2','A1','B2','B3']   }
df1 = pd.DataFrame(teacher1)
df1

teacher2 = {'rno': [1,2,3,4],'subject4': ['A2','A1','B2','B3']}
df2 = pd.DataFrame(teacher2)
df2

df1
df2
pd.concat([df1,df2])
pd.concat([df1,df2], axis=0)

pd.concat([df1,df2], axis=1)


import pandas

rno = pd.Series(range(1,11))

name = []
for i in range(1,11):
    name.append('Student'+str(i))
name
['Student'+str(i) for i in range(1,11)]

name = pd.Series(['Student'+str(i) for i in range(1,11)])
name

import numpy as np
gender = np.random.choice(['M','F'], size=10)
gender

courses = np.random.choice(['BBA', 'MBA', 'BTECH', 'MTECH'], size=10)
courses

marks1 = np.random.randint(0,101, size=10)

marks2 = np.random.randint(0,101, size=10)


pd6 = pd.DataFrame({'rno':rno, 'name':name, 'gender':gender, 'courses':courses,
                    'marks1':marks1, 'marks2':marks2})

pd6



import pandas
rno = pd.Series(range(1,11))
name = pd.Series(['Student'+str(i) for i in range(1,11)])
import numpy as np
gender = np.random.choice(['M','F'], size=10)
courses = np.random.choice(['BBA', 'MBA', 'BTECH', 'MTECH'], size=10)
marks1 = np.random.randint(0,101, size=10)
marks2 = np.random.randint(0,101, size=10)


sinfo = pd.DataFrame({'rno':rno, 'name':name, 'gender':gender, 'courses':courses})
sinfo
rno1 = pd.Series(range(6,16))
einfo = pd.DataFrame({'rno':rno1,'marks1':marks1, 'marks2':marks2})
einfo

pd8 = pd.merge(sinfo, einfo, how='inner')
pd8

pd9 = pd.merge(sinfo, einfo, how='outer')
pd9

sinfo
einfo

pd10 = pd.merge(sinfo, einfo, how='left')
pd10

pd11 = pd.merge(sinfo, einfo, how='right')
pd11

einfo = pd.DataFrame({'rno1':rno1,'marks1':marks1, 'marks2':marks2})
einfo

pd12 = pd.merge(sinfo, einfo, how='inner', left_on='rno', right_on='rno1')
pd12




#grouping


import pandas

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



#Groupby
df.groupby(['courses']).size()







































































































































