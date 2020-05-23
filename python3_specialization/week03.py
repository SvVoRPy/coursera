# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:19:52 2020

@author: svollnhals

Python 3 Specialization on Coursera
Course 4 Python Classes and Inheritance
Week 03 stuff
"""

# equality test
assert 'No' == 'Yes'

assert 'Yes' == 'Yes'

# type test
assert type(9//5) == int

assert type(9.0//5) == int

# Check that all elements of a list have the same type
lst = ['a', 'b', 'c']

# start with first element and check if all consecutive elements are equal type
first_type = type(lst[0])
for item in lst:
    assert type(item) == first_type

lst2 = ['a', 'b', 'c', 17]
first_type = type(lst2[0])
for item in lst2:
    assert type(item) == first_type
    
# check conditional if list has less than n elements
lst = ['a', 'b', 'c']

assert len(lst) < 10

### Testing Conditionals

x = 4
y = 4
if x < y:
    z = x
else:
    if x > y:
        z = y
    else:
        ## x must be equal to y
        assert x==y
        z = 0
print(z)      

### Testing Loops

nums = [1, 5, 8]
accum = 0
for w in nums:
    accum = accum + w
assert accum == 14

nums = []
accum = 0
for w in nums:
    accum = accum + w
assert accum == None
# -> expectation for edge case set to None, but 0 set
# Fixed:
nums = []
if len(nums) == 0:
   accum = None
else:
   accum = 0
   for w in nums:
       accum = accum + w
assert accum == None

### Writing Test Cases for Functions

# The class Student is supposed to accept two arguments in its constructor:
# A name string
# An optional integer representing the number of years the student has been 
# at Michigan (default:1)
# Every student has three instance variables:
# self.name (set to the name provided)
# self.years_UM (set to the number of years the student has been at Michigan)
# self.knowledge (initialized to 0)
# There are three methods:
# .study() should increase self.knowledge by 1 and return None
# .getKnowledge() should return the value of self.knowledge
# .year_at_umich() should return the value of self.years_UM
# There are one or more errors in the class. Use this space to write test cases 
# to determine what errors there are. You will be using this information to 
# answer the next set of multiple choice questions.

class Student:
    def __init__(self, name, years_UM = 1):
        self.name = name
        self.years_UM = years_UM
        self.knowledge = 0
        
    def study(self):
        self.knowledge += 1
        return None
    
    def getKnowledge(self):
        return self.knowledge
    
    def year_at_umich(self):
        return self.years_UM

Student('Sven').year_at_umich()
Student('Sven', 2).year_at_umich()

test_case = Student('Sven')
assert test_case.knowledge == 0
test_case.study()
assert test_case.knowledge == 1


### Exception Handling Flow-of-control
# general exception
try:
    items = ['a', 'b']
    third = items[2]
    print("This won't print")
except Exception:
    print("got an error")

print("continuing")

# Only index exception error
try:
    items = ['a', 'b']
    third = items[2]
    print("This won't print")
except IndexError:
    print("error 1")

print("continuing")

try:
    x = 5
    y = x/0
    print("This won't print, either")
except IndexError:
    print("error 2")

print("continuing again")

# print the exact error catched in the exception stage
try:
    items = ['a', 'b']
    third = items[2]
    print("This won't print")
except Exception as e:
    print("got an error")
    print(e)

print("continuing")

# test case 1

students = [('Timmy', 95, 'Will pass'), ('Martha', 70), ('Betty', 82, 'Will pass'), ('Stewart', 50, 'Will not pass'), ('Ashley', 68), ('Natalie', 99, 'Will pass'), ('Archie', 71), ('Carl', 45, 'Will not pass')]

passing = {'Will pass': 0, 'Will not pass': 0}
for tup in students:
    try:
        if tup[2] == 'Will pass':
            passing['Will pass'] += 1
        elif tup[2] == 'Will not pass':
            passing['Will not pass'] += 1
    except Exception as error_inst:
        print('got an error for tuple {}: {}'.format(tup, error_inst))

# test case 2
        
nums = [5, 9, '4', 3, 2, 1, 6, 5, '7', 4, 3, 2, 6, 7, 8, '0', 3, 4, 0, 6, 5, 
        '3', 5, 6, 7, 8, '3', '1', 5, 6, 7, 9, 3, 2, 5, 6, '9', 2, 3, 4, 5, 1]
plus_four = []
for num in nums:
    try:
        plus_four.append(num+4)
    except:
        plus_four.append('Error')
print(plus_four)

# Exception Types
# except Exception: -> Any kind of exception
# split up as e.g.:
# a) except ZeroDivisionError:-> Only division by 0 exception handled
# b) except IndexError: -> Only index out of bounds error
# c) except NameError: -> Only if object referenced does not exist
# d) except Exception: -> handles everything else

# overview: https://docs.python.org/3/library/exceptions.html

# The code below takes the list of country, country, and searches to see if it 
# is in the dictionary gold which shows some countries who won gold during the 
# Olympics. However, this code currently does not work. Correctly add 
# try/except clause in the code so that it will correctly populate the list, 
# country_gold, with either the number of golds won or the string 
# “Did not get gold”.

gold = {"US":46, "Fiji":1, "Great Britain":27, "Cuba":5, "Thailand":2, 
        "China":26, "France":10}
country = ["Fiji", "Chile", "Mexico", "France", "Norway", "US"]
country_gold = []

for x in country:
    try:
        country_gold.append(gold[x])
    except:
        country_gold.append("Did not get gold")
        
# Provided is a buggy for loop that tries to accumulate some values out of 
# some dictionaries. Insert a try/except so that the code passes.
        
di = [{"Puppies": 17, 'Kittens': 9, "Birds": 23, 'Fish': 90, "Hamsters": 49}, 
      {"Puppies": 23, "Birds": 29, "Fish": 20, "Mice": 20, "Snakes": 7}, 
      {"Fish": 203, "Hamsters": 93, "Snakes": 25, "Kittens": 89}, 
      {"Birds": 20, "Puppies": 90, "Snakes": 21, "Fish": 10, "Kittens": 67}]
total = 0
for diction in di:
    try:
        total = total + diction['Puppies']
    except:
        print("No puppies, no action!")

print("Total number of puppies:", total)

# The list, numb, contains integers. Write code that populates the list 
# remainder with the remainder of 36 divided by each number in numb. 
# For example, the first element should be 0, because 36/6 has no remainder. 
# If there is an error, have the string “Error” appear in the remainder.

numb = [6, 0, 36, 8, 2, 36, 0, 12, 60, 0, 45, 0, 3, 23]

remainder = []

for i in numb:
    try:
        remainder.append(36 % i)
    except:
        remainder.append('Error')
        
print(remainder)

# Provided is buggy code, insert a try/except so that the code passes.
lst = [2, 4, 10, 42, 12, 0, 4, 7, 21, 4, 83, 8, 5, 6, 8, 234, 5, 6, 523, 42, 
       34, 0, 234, 1, 435, 465, 56, 7, 3, 43, 23]

lst_three = []

for num in lst:
    print(num)
    try:
        if 3 % num == 0:
            lst_three.append(num)
    except ZeroDivisionError as e:
        print("Ecountered: {}".format(e))
        
# Write code so that the buggy code provided works using a try/except. 
# When the codes does not work in the try, have it append to the list attempt 
# the string “Error”.
        
full_lst = ["ab", 'cde', 'fgh', 'i', 'jkml', 'nop', 'qr', 's', 'tv', 
            'wxy', 'z']

attempt = []

for elem in full_lst:
    try:
        attempt.append(elem[1])
    except IndexError as e:
        print("Encountered for element {}: {}".format(elem, e))
        attempt.append('Error')

# The following code tries to append the third element of each list in conts 
# to the new list third_countries. Currently, the code does not work. 
# Add a try/except clause so the code runs without errors, and the string 
# ‘Continent does not have 3 countries’ is appended to countries instead of 
# producing an error.
        
conts = [['Spain', 'France', 'Greece', 'Portugal', 'Romania', 'Germany'], 
         ['USA', 'Mexico', 'Canada'], 
         ['Japan', 'China', 'Korea', 'Vietnam', 'Cambodia'], 
         ['Argentina', 'Chile', 'Brazil', 'Ecuador', 'Uruguay', 'Venezuela'], 
         ['Australia'], 
         ['Zimbabwe', 'Morocco', 'Kenya', 'Ethiopa', 'South Africa'], 
         ['Antarctica']]

third_countries = []

for c in conts:
    try:
        third_countries.append(c[2])
    except IndexError as e:
        print("Continent {} does not have 3 countries, error: {}".format(c, e))
        third_countries.append('Continent does not have 3 countries')
        
# The buggy code below prints out the value of the sport in the list sport. 
# Use try/except so that the code will run properly. 
# If the sport is not in the dictionary, ppl_play, 
# add it in with the value of 1.
        
sport = ["hockey", "basketball", "soccer", "tennis", "football", "baseball"]

ppl_play = {"hockey":4, "soccer": 10, "football": 15, "tennis": 8}

for x in sport:
    try:
        print(ppl_play[x])
    except KeyError as e:
        print("{} not in dictionary".format(x))
        ppl_play[x] = 1
        
# Provided is a buggy for loop that tries to accumulate some values out of some 
# dictionaries. Insert a try/except so that the code passes. If the key is not
# there, initialize it in the dictionary and set the value to zero.
        
di = [{"Puppies": 17, 'Kittens': 9, "Birds": 23, 'Fish': 90, "Hamsters": 49}, 
      {"Puppies": 23, "Birds": 29, "Fish": 20, "Mice": 20, "Snakes": 7}, 
      {"Fish": 203, "Hamsters": 93, "Snakes": 25, "Kittens": 89}, 
      {"Birds": 20, "Puppies": 90, "Snakes": 21, "Fish": 10, "Kittens": 67}]
total = 0
for diction in di:
    try:
        total = total + diction['Puppies']
    except KeyError:
        print("'Puppies' not a key in dictionary {}, adding it..."
              .format(diction))
        diction['Puppies'] = 0

print("Total number of puppies:", total)
        













