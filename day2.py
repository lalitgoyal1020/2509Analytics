# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 19:33:14 2022

@author: vikas
"""

# Extended Data Structs

# List, Tuple, Set and Dictionary, 
# Frozen Set, Enumerate

# Hetrogeneous, Mutable, Ordered, Indexed


#List

l1 = []

l1 = [1,2,4,5,3,9,8]

print(l1)
#Mixed Order

#Hetrogeneous

l2 = [2, 3.55, True, "Vivek"]
l2

# Indexed

l2
#Indexes always start with 0 uto n-1

l2[0]
l2[1]
l2[2]
l2[3]
l2[4] #IndexError: list index out of range


for i in l2:
    print(i)
    

r1 = range(11)
l3 = list(r1)
l3

r2 = range(5, 21)
l4 = list(r2)
l4

r3 = range(0, 101, 5)
l5 = list(r3)
print(l5)


l2
l2.append('Vikas')
l2

l2.remove(2)
l2

l2.remove(2) #ValueError: list.remove(x): x not in list

l2
l2.pop()

l2.pop(2)
l2


l2.clear()

l1.clear()

del l1
del l2


l4 = [3,2,5,4,3,2,1]
l4.count(2)


l2 = [2, 3.55, True, "Vivek"]

l1 = l2

l1

l1.append("ABC")

l3 = l2.copy()
l3.append("xyz")



# Set

s1 = {1}

#Ordered Set Values
s2 = {4,1,8,7,9,4,3}

# Not consider Duplicate Values
s3 = {4,1,9,3,9,4,3}


# Not Indexed
s2[3] #TypeError: 'set' object is not subscriptable


print(s2)



for i in s2:
    print(i)


s3
s3.add(6)
s3


s3.update([5,1,8,7,9,4])
s3

s3.remove(8)
s3

s3.remove(8) #KeyError: 8
s3

s3.discard(7)
s3

s3.discard(7)
s3


s3.pop()

s3.clear()
del s3



teamA = {'India', 'Australia','Pakistan', 'England'}
teamB = {'Bangladesh', 'New Zealand', 'West Indies', 'India'}
teamA
teamB

teamA.union(teamB)

+teamA.intersection(teamB)

teamA.difference(teamB)

c1 = set(range(1,11))
c1

e1 = set(range(6,15))
e1

c1.difference(e1)



#Dictionary

d1 = {}

#Key Value pair
d2 = {"rno":1, "name":"VK", "Class":"Analytics"}
d2

#Not indexed
d2[1] #KeyError: 1

#Access
d2["rno"]
d2['name']


car = { 'brand':'Honda', 'model': 'Jazz', 'year' : 2017}
car

car.get('brand')

car.keys()

car.values()

car.items()

for i,j in car.items():
    print(i, j)
    

#Mutable

car['brand'] = "Hundayi"
car

car['color'] = "Red"

car

car.pop('color')
car

car.popitem()

car.clear()

del car




l1 = [1,2,3]

l2 = [(1,2), (3,4), (5,6)]

l3 = [[1,2], [3,4], [5,6]]

l4 = [{1,2}, {3,4}, {5,6}]


for i in l4:
    print(i)


for i in l4:
    for x in i:
        print(x)

for i,j in l4:
    print(i,j)



d4 = car.items()

list(d4)


for i in d4:
    print(i)


for i,j in d4:
    print(i,j)


# Tuple

t1 = ()

#Not Ordered
t2 = (4,2,7,6,9)
t2

#Indexed

t2[3]
t2[4]


# Not Mutable

t2[3] = 22 #TypeError: 'tuple' object does not support item assignment

t3 = (5,4,2,5,7,6,9,1,2,4,6)
t3.count(6)



# Condition and Iterative

# Conditions

'''
Other programming Languages
if (condition)
{
    statement1
    statement2 
    .....
    .....
}
'''

a=10
b=20

if (a<b):
    print("a is lesser")
    

a=30
b=20

if (a<b):
    print("a is lesser")
    
    

a<b

if (a<b):
    print("a is lesser")
    print("b is greater")
    

if (a<b):
    print("a is lesser")
print("b is greater")



if (True):
    print("a is lesser")


if (False):
    print("a is lesser")

    
#If Else


a=20
b=30
if (a<b):
    print("a is lesser")
else:
    print("b is lesser")



a=40
b=30
if (a<b):
    print("a is lesser")
else:
    print("b is lesser")





a=40
b=30
if (a<b):
    print("a is lesser")
    print("b is greater")
else:
    print("b is lesser")
    print("a is greater")


if (a!=b):
    print("a is not equal to b")


# if elseif

'''
marks           grades
<50             F
<60 and >=50    D
<70 and >=60    C
<80 and >=70    B
<90 and >=80    A
>=90            O
'''

marks = 45
marks < 50

marks = 55
marks<60 and marks>=50

marks=95
if marks < 50:
    print("F")
elif (marks<60 and marks >=50):
    print("D")
elif (marks<70 and marks >=60):
    print("C")
elif (marks<80 and marks >=70):
    print("B")
elif (marks<90 and marks >=80):
    print("A")
else:
    print("O")

# Nested IF
'''
marks           grades
<50             F
<60 and >=50    D
<70 and >=60    C
<80 and >=70    B
<90 and >=80    A
>=90            O
'''

marks = 75

if marks<50:
    print("F")
else:
    if marks<60:
        print("D")
    else:
        if(marks<70):
            print("C")
        else:
            if(marks<80):
                print("B")
            else:
                if(marks<90):
                    print("A")
                else:
                    print('O')


#Iterative Statements or Loop Structs

l1 = list(range(1,1000000))
l1

for i in l1:
    print(i)
    
    
#Print table of 2
'''
2 * 1 = 2
2 * 2 = 4

2 * 10 = 20
'''


for i in range(1, 11):
    print("2","*",i,"=",2*i)

for i in range(1, 11):
    print(f"2 * {i} = {2*i}")


j=2
for i in range(1, 11):
    print(f"{j} * {i} = {j*i}")



for j in range(2,6):
    for i in range(1, 11):
        print(f"{j} * {i} = {j*i}")

list(range(10,1))

for j in range(1,6):
    for i in range(1,j+1):
        print('*', end='')
    print()

for j in range(1,6):
    for i in range(1,j+1):
        print(i, end='')
    print()

for j in range(1,6):
    for i in range(1,j+1):
        print(j, end='')
    print()



'''
While

while (condition):
    statement 1
    statement 2
    statement n
'''


i = 1
while(i<11):
    print(i)
    i = i +1

i=1
while(True):
    print(i)
    i = i +1

i=1
while(False):
    print(i)
    i = i +1


j = 2
i = 1
while(i<11):
   print(f"{j} * {i} = {j*i}") 
   i=i+1



j=2
while(j<6):
    i = 1
    while(i<11):
       print(f"{j} * {i} = {j*i}") 
       i=i+1
    j=j+1



'''
n=5

f = 5*4*3*2

'''

f=1
n=5
while(n>1):
   f=f*n
   n=n-1
print(f)


# break and Continue

l1 = ['UAE', "SA", "AUS", "USA", "IND", "NEP", "BAN"]

for i in l1:
    if(i == 'IND'):
        print(i, "Found")
    else:
        print(i, "Not Found")
        
for i in l1:
    print(i)
    


for i in l1:
    if(i == 'IND'):
        print(i, "Found")
    else:
        print(i, "Not Found")
print("Loop completed")


for i in l1:
    if(i == 'IND'):
        print(i, "Found")
        break
    else:
        print(i, "Not Found")
print("Loop completed")


#Continue


for i in l1:
    break
    print(i, "Executed")

for i in l1:
    if(i == 'IND'):
        print(i, "Found")
        break
    print(i, "Executed")
print("Loop completed")


for i in l1:
    if(i == 'IND'):
        print(i, "Found")
    print(i, "Executed")
print("Loop completed")


for i in l1:
    if(i == 'IND'):
        print(i, "Found")
        continue
    print(i, "Executed")
print("Loop completed")



#Functions
# call Anywhere
# write lenght of code Once and call with single word name
# give different define inputs
# generated different define outputs
# call same code again and again with the single word name
# take inputs called as parameters or arguments
# generate outcomes calles a return values


def fn():
    a=10
    b=20
    c=a+b
    print(c)


fn()

fn()


def fn1(a,b):
    c=a+b
    print(c)


fn1(10,20)

fn1(40,30)

fn1(20,60)

o = fn1(2,4)
print(o)

#Function Call

l1 = [4,6,3,8,7]
m = max(l1)
print(m)

#Return - Ouput of a function

def fn2(a,b):
    c=a+b
    return(c)


m = fn2(10,20)
m=m*2
print(m)

#maximum function

def maximum(l):
    m=0
    for i in l:
        if (m<i):
            m=i
    return(m)

l1 = [4,2,8,7,4]
a = maximum(l1)
print(a)



l2 = [5,6,2,8,7,4]
a = maximum(l2)
print(a)



def empdet(eno, name, email='Not Avail'):
    print(eno,name, email)
    
empdet(11, 'VK', 'vk@gmail.com')

empdet(12, 'VKay')

empdet(12)



# Lambda Functions
#Single line functions

def fn4(s):
    return(s.upper())

fn4('abc')


def sq(x):
    return(x**2)

sq(4)

sq(9)


sqr = lambda x: x**2

sqr(4)

sqr(9)


lm1 = lambda x,y : x*y
lm1(10,20)


# map, filter


l1 = [2,3,4,5,6]
l2 = []

for i in l1:
    l2.append(i**2)

l2


lm2 = lambda x: x**2

l3 = list(map(lm2, l1))
l3


l2 = ['abc', 'def', 'efg']

def fn4(s):
    return(s.upper())

l3 = list(map(fn4, l2))
l3



#filter

l1 = [2,3,4,5,6]

l2 = []

for i in l1:
    if (i%2==0):
        l2.append(i)

l2


even = lambda x: x%2==0

even(10)


l3 = list(filter(even, l1))
l3



l3 = list(map(even, l1))
l3

































































    

































































    
    















































'''
Stud 
rno name
1 A
2 B
3 C

exam
rno marks
1 22
3 34


s1 = 1,2,3
s2 = 1,3

s1 in s2
'''

















































