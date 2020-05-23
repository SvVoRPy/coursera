# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:19:52 2020

@author: svollnhals

Python 3 Specialization on Coursera
Course 4 Python Classes and Inheritance
Week 1 stuff
"""

# 20.5 Adding Other Methods to a Class
# __init__ is the constructor of the class
class Animal:
    """ Create a class called Animal that accepts two numbers as inputs and assigns 
        them respectively to two instance variables: arms and legs. Create an 
        instance method called limbs that, when called, returns the total number 
        of limbs the animal has. To the variable name spider, assign an instance 
        of Animal that has 4 arms and 4 legs. Call the limbs method on the spider 
        instance and save the result to the variable name spidlimbs. """
    
    def __init__(self, int1, int2):
        
        self.arms = int1
        self.legs = int2
        
    def limbs(self):
        return self.arms + self.legs
    
spider = Animal(4, 4)
spidlimbs = spider.limbs()

print(spidlimbs)

# 20.6 Objects as Arguments and Parameters

import math

class Point:
    """ Point class for representing and manipulating x,y coordinates. """

    def __init__(self, initX, initY):

        self.x = initX
        self.y = initY

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distanceFromOrigin(self):
        return ((self.x ** 2) + (self.y ** 2)) ** 0.5

# call distance on initialized instances of class Point
def distance(point1, point2):
    xdiff = point2.getX()-point1.getX()
    ydiff = point2.getY()-point1.getY()

    dist = math.sqrt(xdiff**2 + ydiff**2)
    return dist

p = Point(4,3)
q = Point(0,0)
print(distance(p, q))

### Creating Instances from Data

cityNames = ['Detroit', 'Ann Arbor', 'Pittsburgh', 'Mars', 'New York']
populations = [680250, 117070, 304391, 1683, 8406000]
states = ['MI', 'MI', 'PA', 'PA', 'NY']

city_tuples = list(zip(cityNames, populations, states))
print(city_tuples)

# define this into a class
class City:
    def __init__(self, n, p, s):
        self.name = n
        self.population = p
        self.state = s
    def __str__(self):
        return '{}, {} (pop: {})'.format(self.name, self.state, self.population)
    
cities = []
for city_tup in city_tuples:
    name, pop, state = city_tup
    print("{} {} {}".format(name, pop, state))
    city = City(name, pop, state) # instance of the City class
    cities.append(city)
print(cities)

cities = [City(n, p, s) for (n, p, s) in city_tuples]
print(cities)

cities = [City(*t) for t in city_tuples]
print(cities)

# str on class: determines what gets represented as a string/print out when 
# calling print on the class
class Point:
    """ Point class for representing and manipulating x,y coordinates. """

    def __init__(self, initX, initY):

        self.x = initX
        self.y = initY

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distanceFromOrigin(self):
        return ((self.x ** 2) + (self.y ** 2)) ** 0.5

    def __str__(self):
        return "x = {}, y = {}".format(self.x, self.y)

p = Point(7,6)
print(p)

class Cereal:
    
    def __init__(self, str1, str2, int1):
        self.name = str1
        self.brand = str2
        self.fiber = int1
        
    def __str__(self):
        return ("{} cereal is produced by {} and has {} grams of fiber in every serving!"
                .format(self.name, self.brand, self.fiber))
    
c1 = Cereal("Corn Flakes", "Kellogg's", 2)
c2 = Cereal("Honey Nut Cheerios", "General Mills", 3)

print(c1)
print(c2)

### special dunderscore
class Point:
    """ Point class for representing and manipulating x,y coordinates. 
        Includes the special dunderscores add and sub """

    def __init__(self, initX, initY):
        self.x = initX
        self.y = initY
        
    def __str__(self):
        return "Point ({}, {})".format(self.x, self.y)
    
    def __add__(self, otherPoint):
        return Point(self.x + otherPoint.x,
                     self.y + otherPoint.y)
        
    def __sub__(self, otherPoint):
        return Point(self.x - otherPoint.x,
                     self.y - otherPoint.y)
        
p1 = Point(-5, 10)
p2 = Point(15, 20)

print(p1)
print(p2)
print(p1 + p2)
print(p1 - p2)
    
### Instances as Return Values

class Point:
    """ Point class for representing and manipulating x,y coordinates. 
        Includes a function that returns a new point that is halfway
        between p and q. """

    def __init__(self, initX, initY):
        self.x = initX
        self.y = initY

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distanceFromOrigin(self):
        return ((self.x ** 2) + (self.y ** 2)) ** 0.5
    
    def halfway(self, target):
        mx = (self.x + target.x) / 2
        my = (self.y + target.y) / 2
        return Point(mx, my)

    def __str__(self):
        return "x = {}, y = {}".format(self.x, self.y)

p = Point(3, 4)
q = Point(5, 12)

mid = p.halfway(q)
# -> mid is a new output of class Point!
print(mid) 
print(mid.x)
print(mid.y)

### Sorting list of instances
class Fruit():
    """ Takes as an input the name and price of a fruit. """
    def __init__(self, name, price):
        self.name = name
        self.price = price
        
    def sort_priority(self):
        return self.price
        
L = [Fruit('Cherry', 10), Fruit('Apple', 5), Fruit('Blueberry', 20)]

print("-----sorted by price, referencing a class method-----")
for f in sorted(L, key = Fruit.sort_priority):
    print(f.name)

print("---- one more way to do the same thing-----")
for f in sorted(L, key = lambda x: x.sort_priority()):
    print(f.name)  

### Class Variables and Instance Variables
class Point:
    """ Point class for representing and manipulating x,y coordinates. 
        Includes sub classes """
        
    printed_rep = "*" # -> own class variables that belong to the class
    
    def __init__(self, initX, initY):
        self.x = initX
        self.y = initY
        
    def graph(self):
        rows = []
        size = max(int(self.x), int(self.y)) + 2
        for j in range(size - 1):
            if (j + 1) == int(self.y):
                special_row = (str((j+1) % 10) + (" "*(int(self.x) - 1)
                    + self.printed_rep))
                rows.append(special_row)
            else:
                rows.append(str((j + 1) % 10))
        rows.reverse() # higher values of y first
        x_axis = ""
        for i in range(size):
            x_axis += str(i % 10)
        rows.append(x_axis)
        
        return "\n".join(rows)
    
p1 = Point(2, 3)
p2 = Point(3, 12)
print(p1.graph()) # -> looks first in instance p1, if not there searches
                  # in rest of class
print()
print(p2.graph()) # -> 
print(p1.printed_rep)
print(p1.x)
print(p1.z) # -> runtime error as not existent
                
### General for classes
# 1) What kind of data do you want to represent with the class (songs) ?
# 2) What does an instance represent? (e.g. a song, a student)
# 3) What are the instance variables, what is unique to them? (e.g. artist,
# track length)
# 4) What methods do you want? -> e.g. ping an external API to get lyrics for 
# a song
# 5) What is a good representation/print of an instance? 
# (e.g. artist and title)

# -> It is more an art than a science!

### Testing classes
class Point:
    """ Point class for representing and manipulating x,y coordinates. """

    def __init__(self, initX, initY):

        self.x = initX
        self.y = initY

    def distanceFromOrigin(self):
        return ((self.x ** 2) + (self.y ** 2)) ** 0.5

    def move(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy


#testing class constructor (__init__ method)
p = Point(3, 4)
assert p.y == 4
assert p.x == 3

#testing the distance method
p = Point(3, 4)
assert p.distanceFromOrigin() == 5.0

#testing the move method
p = Point(3, 4)
p.move(-2, 3)
assert p.x == 1
assert p.y == 7

### Text Implementation of a Tamagotchi Game
# class Pet: Each instance of the class will be one electronic pet 
# for the user to take care of. Each instance will have a current state, 
# consisting of three instance variables:
# 1) hunger, an integer (random initial value)
# 2) boredom, an integer (random initial value)
# 3) sounds, a list of strings, each a word that the pet has been taught to say
# (copy of existing list of sounds to manipulate them)
# Method 1: clock_tick increments boredom and hunger instances
# __str__: pet's current state
# Method 2: teach() new word
# Method 3: hi() interact with the pet
# Method 4: reduce_boredom() which is invoced by Method 2 and 3
# Method 5: feed() to reduce hunger

from random import randrange

class Pet():
    boredom_decrement = 4
    hunger_decrement = 6
    boredom_threshold = 5
    hunger_threshold = 10
    sounds = ['Mrrp']
    def __init__(self, name = "Kitty"):
        self.name = name
        self.hunger = randrange(self.hunger_threshold)
        self.boredom = randrange(self.boredom_threshold)
        self.sounds = self.sounds[:]  
        # copy the class attribute, so that when we make changes to it, 
        # we won't affect the other Pets in the class

    def clock_tick(self):
        self.boredom += 1
        self.hunger += 1

    def mood(self):
        if self.hunger <= self.hunger_threshold and self.boredom 
        <= self.boredom_threshold:
            return "happy"
        elif self.hunger > self.hunger_threshold:
            return "hungry"
        else:
            return "bored"

    def __str__(self):
        state = "     I'm " + self.name + ". "
        state += " I feel " + self.mood() + ". "
        # state += ("Hunger {} Boredom {} Words {}"
        #           .format(self.hunger, self.boredom, self.sounds))
        return state

    def hi(self):
        print(self.sounds[randrange(len(self.sounds))])
        self.reduce_boredom()

    def teach(self, word):
        self.sounds.append(word)
        self.reduce_boredom()

    def feed(self):
        self.reduce_hunger()

    def reduce_hunger(self):
        self.hunger = max(0, self.hunger - self.hunger_decrement)

    def reduce_boredom(self):
        self.boredom = max(0, self.boredom - self.boredom_decrement)

### Visual implementation of the game to let someone play
        
import sys
sys.setExecutionLimit(60000)

def whichone(petlist, name):
    for pet in petlist:
        if pet.name == name:
            return pet
    return None # no pet matched

def play():
    animals = []

    option = ""
    base_prompt = """
        Quit
        Adopt <petname_with_no_spaces_please>
        Greet <petname>
        Teach <petname> <word>
        Feed <petname>

        Choice: """
    feedback = ""
    while True:
        action = input(feedback + "\n" + base_prompt)
        feedback = ""
        words = action.split()
        if len(words) > 0:
            command = words[0]
        else:
            command = None
        if command == "Quit":
            print("Exiting...")
            return
        elif command == "Adopt" and len(words) > 1:
            if whichone(animals, words[1]):
                feedback += "You already have a pet with that name\n"
            else:
                animals.append(Pet(words[1]))
        elif command == "Greet" and len(words) > 1:
            pet = whichone(animals, words[1])
            if not pet:
                feedback += "I didn't recognize that pet name. Please try again.\n"
                print()
            else:
                pet.hi()
        elif command == "Teach" and len(words) > 2:
            pet = whichone(animals, words[1])
            if not pet:
                feedback += "I didn't recognize that pet name. Please try again."
            else:
                pet.teach(words[2])
        elif command == "Feed" and len(words) > 1:
            pet = whichone(animals, words[1])
            if not pet:
                feedback += "I didn't recognize that pet name. Please try again."
            else:
                pet.feed()
        else:
            feedback+= "I didn't understand that. Please try again."

        for pet in animals:
            pet.clock_tick()
            feedback += "\n" + pet.__str__()

play()

##############################################################################
### Assessment: Week One

# Define a class called Bike that accepts a string and a float as input, and 
# assigns those inputs respectively to two instance variables, color and price.
# Assign to the variable testOne an instance of Bike whose color is blue and 
# whose price is 89.99. Assign to the variable testTwo an instance of Bike 
# whose color is purple and whose price is 25.0.

class Bike:
    """ """
    def __init__(self, str_in, float_in):
        self.color = str_in
        self.price = float_in
        
testOne = Bike('blue', 89.99)
testTwo = Bike('purple', 25.0)

# Create a class called AppleBasket whose constructor accepts two inputs: 
# a string representing a color, and a number representing a quantity of apples
# The constructor should initialize two instance variables: apple_color and 
# apple_quantity. Write a class method called increase that increases the 
# quantity by 1 each time it is invoked. You should also write a __str__ method 
# for this class that returns a string of the format: 
# "A basket of [quantity goes here] [color goes here] apples." e.g. 
# "A basket of 4 red apples." or "A basket of 50 blue apples." 
# (Writing some test code that creates instances and assigns values to 
# variables may help you solve this problem!)

class AppleBasket:
    """ """
    def __init__(self, color, n_apples):
        self.apple_color = color
        self.apple_quantity = n_apples
        
    def increase(self):
        self.apple_quantity += 1
        
    def __str__(self):
        return ("A basket of {} {} apples."
                .format(self.apple_quantity, self.apple_color))
        
# Define a class called BankAccount that accepts the name you want associated 
# with your bank account in a string, and an integer that represents the 
# amount of money in the account. The constructor should initialize two 
# instance variables from those inputs: name and amt. Add a string method so 
# that when you print an instance of BankAccount, you see 
# "Your account, [name goes here], has [start_amt goes here] dollars." 
# Create an instance of this class with "Bob" as the name and 100 as the amount
# Save this to the variable t1.
        
class BankAccount:
    """ """
    def __init__(self, name_account, amount_money):
        self.name = name_account
        self.amt = amount_money
        
    def __str__(self):
        return ("Your account, {}, has {} dollars.".format(self.name, self.amt))
    
t1 = BankAccount('Bob', 100)




    
    
    
    
    
    
    
    
    
    
