# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:47:18 2022

@author: vikas
"""

'''
Compiler  C++, Java  -> Machine Codes conversion of Full File in one go-> 
Binary 0/1 -> Execution of Machine Code -> Conversion of Machine Output ot Human Readable->
Ouput in Human Readable Format 


Interpreter Python, R
Machine Codes conversion of Line by Line in one go-> 
Binary 0/1 -> Execution of Machine Code -> Conversion of Machine Output ot Human Readable->
Ouput in Human Readable Format

CDE - Command based development Env
IDE -  Interactive Development Environments
'''



print("Python")
print("Analytics")


print?


print("Python", "Anlytics", "Finance")

print("Python", "Anlytics", "Finance", sep='-')


print("Python")
print("Analytics")
print("Finance")

print?


'''
standard 
\n next line, 
\t tab, 
\b backspace
'''

#standard \n next line, \t tab, \b backspace

print("Python", end='\t')
print("Analytics", end='-')
print("Finance")


!pip install pandas

import pandas

pandas.__version__

import pandas as pd

pd.__version__



#String

country = "India"
a =10
b =30.4
c =True


print("country")

print(country)
print(a)
print(b)
print(c)

lc ="India"
wc ='USA'

print("I live in India and I work in USA")

print("I live in", lc," and I work in", wc, sep='-')

print(f"I live in {lc} and I work in {wc}")

print(f"I live in {lc} and I work in {wc}", sep='-')

print("I live in {0} and I work in {1}".format(lc, wc))

print("I live in {1} and I work in {0}".format(lc, wc))

print("I live in {0} and I work in {1}".format(wc, lc))



a=10
b=20
c=a+b
print(c)


a="10"
b="20"
c=a+b
print(c)


st = "Your marks are "
marks = 80

result = st + str(marks)
result




a=10
b="20"

c = a+int(b)
c


# Type Casting or Type conversion

#int to str
a =10
b = str(a)
print(b)

#int to float
a =10
b = float(a)
print(b)


#float to int
a =10.8
b = int(a)
print(b)


#float to sting
a =10.8
b = str(a)
print(b)


#str to int
a ="10"
b = int(a)
print(b)


#str to float
a ="10"
b = float(a)
print(b)



# Char to ASCII

c = "A"
o = ord(c)
print(o)


c = "i"
o = ord(c)
print(o)


c = "!"
o = ord(c)
print(o)


c = " "
o = ord(c)
print(o)


# ASCII to Char

c = 65
o = chr(c)
print(o)

c = 102
o = chr(c)
print(o)

c = 25
o = chr(c)
print(o)




pas1 = "ABC"
pas1 = "21AB"
pas1 = "%%AV"


for i in pas1:
    if ord(i)<65 or ord(i)>90:
        print("Not Correct")
        break



#Airthmetic Operators

a =50
b=20

c =a+b
print(c)

c =a-b
print(c)

c =a*b
print(c)

c =a/b
print(c)

c =a%b
print(c)


a = 1002
a%2 == 0

if(a%2==0):
    print("Even")
else:
    print("Odd")


a =5
b=(6)
c = a**b
c


#Boolean Operators
#o or 1 , True or False, #Yes or No

t = True 
f = False


#Comparison Operators


a = 10
b = 20

# = Assignment Operator
# == Comparison Operator to check "Is Equal To"

a == b

a=10
b=10
a==b

a=10
b=10
a!=b

a=10
b=20
a!=b


a=10
b=20
a<b

a>b

a<=b
a>=b


#Boolean Logics

'''
AND
X Y O
0 0 0
0 1 0
1 0 0
1 1 1

OR
X Y O
0 0 0
0 1 1
1 0 1
1 1 1

NOT
X O
0 1
1 0
'''

t
f

t and t
t and f

t or f
t or t


# String Handling

var = "python"
var = var.capitalize()

var = var.upper()
var

var = var.lower()
var

var = "  I work in JAVA"
var = var.replace("JAVA", "PYTHON")
var


var = '    vikaskhullar     '
print(var)
var = var.strip()
var


var = "Vikas Khullar"
fname, lname = var.split(' ')

fname
lname
















































































































