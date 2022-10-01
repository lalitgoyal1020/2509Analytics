# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 20:38:59 2022

@author: vikas
"""
'''
Interpreter Python, R
Machine Codes conversion of Line by Line in one go-> 
Binary 0/1 -> Execution of Machine Code -> Conversion of Machine Output ot Human Readable->
Ouput in Human Readable Format


Compiler  C++, Java  -> Machine Codes conversion of Full File in one go-> 
Binary 0/1 -> Execution of Machine Code -> Conversion of Machine Output ot Human Readable->
Ouput in Human Readable Format 


CDE - Command based development Env

IDE -  Interactive Development Environments

'''


print('Python')
print('Analytics')


print?

#Function may or may not have parameters
print("Python")

print("Python","Analytics", 'Finanace', 'Business')

print("Python","Analytics", 'Finanace', 'Business', sep='-')


print("Python")
print("Analytics")
print('Finanace')
print('Business')


'''
Multi Line Comment
special Characters 
\n  for new line
\t for new tab
'''
# Comment Single Line

print("Python", end='-')
print("Analytics")
print('Finanace',end='///%')
print('Business')
help(print)


# Open Source
# Source codes avaialable to all

#!pip install pandas


import pandas1 #ModuleNotFoundError: No module named 'pandas1'

import pandas

pandas.__version__

import pandas as pd
pd.__version__


print("Hello World")

#String Variable
country = "India"

# Integer Variable
a = 10

#Float Variable
b = 20.3

'''
C++ steps for variable declaration

int a;
a=10
'''

print("India")

print("country")
print(country)

country = input()

#Inputs given either at Before Execution and After Execution

#country = input("Enter your Country")


lc = 'India'
wc ='USA'

print(lc, wc)

print("I live in India and I work in USA")

# Handling the print statments
print(f"I live in {lc} and I work in {wc}")

print("I live in {1} and I work in {0}".format(lc,wc))

print("I live in",lc,"and I work in",wc)

print("I live in",lc,"and I work in",wc, sep ='-')


a = 10

print(f"a is {a}")

b = "a is"
a=10
b =20
c= a+b


a =" a is "
b = "20"
c =a+b
print(c)

a = 'a is '
b = 20
c = a+b

#Type Casting or Type Conversion

age = 35
"35 years"
age = str(age) + " years"
print(age)

#Int to str
a = 20
c = str(a)

#Int to float
a = 20
c = float(a)

#Float to str
a = 31.4
c = str(a)

#Float to int
a =31.4
c = int(a)

#str to int
a = "12"
c = int(a)

#str to float
a="10.2"
c =float(a)

# Char to ASCII Decimal Value

x = "Z"
a = ord(x)
a

x = "%"
a = ord(x)
a

x = "g"
a = ord(x)
a
    
a = 65
x = chr(a)
x

a = 108
x = chr(a)
x


# Password
# A-Z

pwd = "AABB"
pwd = "1AA#BBzz"

for p in pwd:
    c = ord(p)
    if not(c>=65 and c<= 90):
        print("Not correct Password")
        break

# Airth Operators
a = 5

# Binary Airth Operators
a + 1
a-1
a/2
a*2

a =5
b= 6

a+b
a-b
a/b
a*b

a=2
b=3
a**b
a**(1/b)

a = 10
b = 2

a/b
a%b

a = 11
b = 2

a/b
a%b


x = 131221
x%2

if (x%2 == 0):
    print("Value is Even")
else:
    print("Value is Odd")


# Assignment operator "="

a = 20

#Boolean Variables
# 0 or 1, true or false, yes or no
t = True
f = False

t

# Comparasion Operators 

a = 10
b = 20
a == b
# a "is equal to" b

a = 20
b = 20
a == b

a = 20
b = 10

a<b
a>b
a<=b
a>=b
a!=b


# Boolean Operations
'''
AND
A B X
F F F
F T F
T F F
T T T
'''

c = 99
x = c>=65
x
y = c<=90
y

x and y
c>=65 and c<= 90

'''
OR
A B X
F F F
F T T
T F T
T T T
'''

c = 99
x = c>=65
x
y = c<=90
y

x or y
c>=65 or c<= 90


c = 60
x = c>=65
x
y = c<=90
y

x or y
c>=65 or c<= 90

'''
NOT
A X
F T
T F
'''

not(True)
not (False)


#String Handling

var = "python"
var.capitalize()
var = var.upper()
var
var.lower()


var = " I work in JAVA"
var = var.replace("JAVA", "PYTHON")
var


var = "Python Programming"
var = var.split(" ")

var

var[0]
var[1]


var = "     vikas khullar    "
var = var.strip()
var

# Basic Data Types - Int, String, Float, Boolean

# Extended Datatype - List, Set, Dictionary, Tuple

#Properties - Mutable, Indexed, Ordered, Key-Value Paired, Hetrogeneous, etc


# List

L1 = []

l2 = [7,2,3,6,8,9]

print(l2)

#Indexed
l2[0]
l2[1]
l2[2]
l2[3]
l2[4]
l2[5]
l2[6] # IndexError: list index out of range

l2

#Looping or Iterative Structure

for i in l2:
    print(i) #Intendation

'''
for()
{
print()
}

'''

# Mutable or Changable
l2
l2[0]
l2[0] = 30
l2

# Hetrogeneous
type(l2[0])
type(l2[1])

# Hetrogeneous
l3 = [5, 4.6, "Python", True]
l3

l3










































































































































































