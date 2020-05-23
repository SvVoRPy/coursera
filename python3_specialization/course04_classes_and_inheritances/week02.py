# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:19:52 2020

@author: svollnhals

Python 3 Specialization on Coursera
Course 4 Python Classes and Inheritance
Week 02 stuff
"""

# 22.2 Inheriting Variables and Methods
import time
CURRENT_YEAR = int(time.strftime("%Y"))

class Person:
    def __init__(self, name, year_born):
        self.name = name
        self.year_born = year_born
        
    def getAge(self):
        return CURRENT_YEAR - self.year_born
    
    def __str__(self):
        return '{} ({})'.format(self.name, self.getAge())
    
alice = Person('Alice Smith', 1990)
print(alice)

# modified to a studen class with another knowledge instance
class Student:
    def __init__(self, name, year_born):
        self.name = name
        self.year_born = year_born
        self.knowledge = 0
        
    def study(self):
        self.knowledge += 1
        
    def getAge(self):
        return CURRENT_YEAR - self.year_born
    
    def __str__(self):
        return '{} ({})'.format(self.name, self.getAge())
    
alice = Student('Alice Smith', 1990)
print(alice)    
print(alice.knowledge) 

# Easier with inheritance!!!
# -> conceptual: Every student is a person and can be inheritad from Person
class Student(Person):
    def __init__(self, name, year_born):
        # call the instructor from the Person class
        Person.__init__(self, name, year_born)
        self.knowledge = 0
        
    def study(self):
        self.knowledge += 1

alice = Student('Alice Smith', 1990)
alice.study()
print(alice.knowledge)   
print(alice) # -. able to call getAge and __str__ via Person in Student!

### Overriding Methods
# general: a sub-class gets inherited from a super-class, instance gets created
# of sub-class
# Ordering to search for called method: 
# 1) look in instance
# 2) look in sub-class (class of instance)
# 3) look in super-class of sub-class

# When should you inherit? Only(!) if sub-class has everything that also 
# super-class has plus something more or modified

class Book():
    def __init__(self, title, author):
        self.title = title
        self.author = author
        
    def __str__(self):
        return '"{}" by {}'.format(self.title, self.author)
    
myBook = Book('The Odyssey', 'Homer')
print(myBook)

# now: paper and e-books as sub classes of a book
class PaperBook(Book):
    def __init__(self, title, author, numPages):
        Book.__init__(self, title, author)
        self.numPages = numPages
        
class EBook(Book):
    def __init__(self, title, author, size):
        Book.__init__(self, title, author)
        self.size = size
        
myEBook = EBook('The Odyssey', 'Homer', 2)
myPaperBook = PaperBook('The Odyssey', 'Homer', 500)
print(myEBook.size)
print(myPaperBook.numPages)

# -> EBook and PaperBook have everthing that super-class has, plus one more
# characteristic (size or numPages)

### When you don't want to inherit!
# library that contains books
class Library:
    def __init__(self):
        self.books = []
    def addBook(self, book):
        self.books.append(book)
    def getNumBooks(self):
        return len(self.books)
    
aadl = Library()
myEBook = EBook('The Odyssey', 'Homer', 2)
myPaperBook = PaperBook('The Odyssey', 'Homer', 500)
aadl.addBook(myEBook)
aadl.addBook(myPaperBook)

print(aadl.getNumBooks())
# -> library is not a super-class for the books, as it has different charact.

### Invoking the Parent Class Method
from random import randrange

# Here's the original Pet class, which involves a method feed
# every Pet has its own way to say thank you if you feed him...
# so the method has to be overwritten for every Pet class (see below)
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
        self.sounds = self.sounds[:]  # copy the class attribute, so that when we make changes to it, we won't affect the other Pets in the class

    def clock_tick(self):
        self.boredom += 1
        self.hunger += 1

    def mood(self):
        if self.hunger <= self.hunger_threshold and self.boredom <= self.boredom_threshold:
            return "happy"
        elif self.hunger > self.hunger_threshold:
            return "hungry"
        else:
            return "bored"

    def __str__(self):
        state = "     I'm " + self.name + ". "
        state += " I feel " + self.mood() + ". "
        # state += "Hunger %d Boredom %d Words %s" % (self.hunger, self.boredom, self.sounds)
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

# Dog is a sub-class of Pet that modifies the feed
class Dog(Pet):
    def feed(self):
        print('Arf! Thanks')
        # call feed from super-class!
        Pet.feed(self)
        
d1 = Dog('YoYoBelly')
d1.feed()

class Bird(Pet):
    sounds = ["chirp"]
    def __init__(self, name = "Kitty", chirp_number = 2):
        Pet.__init__(self, name) # call the parent class's constructor
        # basically, call the SUPER -- the parent version -- of the constructor
        # with all the parameters it needs.
        self.chirp_number = chirp_number
        
    def hi(self):
        for i in range(self.chirp_number):
            print(self.sounds[randrange(len(self.sounds))])
        self.reduce_boredom()

b1 = Bird('tweety', 5)
b1.teach("Polly wanna cracker")
b1.hi()
print(b1)

#####################################################################
### Assignments Week 02

class Pokemon(object):
    attack = 12
    defense = 10
    health = 15
    p_type = "Normal"

    def __init__(self, name, level = 5):
        self.name = name
        self.level = level

    def train(self):
        self.update()
        self.attack_up()
        self.defense_up()
        self.health_up()
        self.level = self.level + 1
        if self.level%self.evolve == 0:
            return self.level, "Evolved!"
        else:
            return self.level

    def attack_up(self):
        self.attack = self.attack + self.attack_boost
        return self.attack

    def defense_up(self):
        self.defense = self.defense + self.defense_boost
        return self.defense

    def health_up(self):
        self.health = self.health + self.health_boost
        return self.health

    def update(self):
        self.health_boost = 5
        self.attack_boost = 3
        self.defense_boost = 2
        self.evolve = 10

    def __str__(self):
        self.update()
        return "Pokemon name: {}, Type: {}, Level: {}".format(self.name, self.p_type, self.level)

class Grass_Pokemon(Pokemon):
    attack = 15
    defense = 14
    health = 12

    def update(self):
        self.health_boost = 6
        self.attack_boost = 2
        self.defense_boost = 3
        self.evolve = 12

    def moves(self):
        self.p_moves = ["razor leaf", "synthesis", "petal dance"]
        
    def action(self):
        return '{} knows a lot of different moves!'.format(self.name)

p1 = Grass_Pokemon('Belle')
print(p1)

# Modify the Grass_Pokemon subclass so that the attack strength for 
# Grass_Pokemon instances does not change until they reach level 10. 
# At level 10 and up, their attack strength should increase by the 
# attack_boost amount when they are trained.

# To test, create an instance of the class with the name as "Bulby". 
# Assign the instance to the variable p2. Create another instance of the 
# Grass_Pokemon class with the name set to "Pika" and assign that instance to 
# the variable p3. Then, use Grass_Pokemon methods to train the p3 
# Grass_Pokemon instance until it reaches at least level 10.

class Grass_Pokemon(Pokemon):
    attack = 15
    defense = 14
    health = 12
    p_type = "Grass"

    # only update attack by boost after level 10
    def update(self):
        self.health_boost = 6
        # gets level from super class Pokemon, as not existent in Grass_Pokemon
        if self.level >= 10:
            self.attack_boost = 2
        else:
            self.attack_boost = 0
        self.defense_boost = 3
        self.evolve = 12

    def moves(self):
        self.p_moves = ["razor leaf", "synthesis", "petal dance"]
    
    
p2 = Grass_Pokemon('Bulby')
p3 = Grass_Pokemon('Pika')

print(p2); print(p3)
# train Pika x times
n_trains = 10
for i in range(n_trains):
    attack_before = p3.attack
    p3.train()
    print('Trained {} to level {}, attack was at {}, is now at {}'
          .format(p3.name, p3.level, attack_before, p3.attack))

# Along with the Pokemon parent class, we have also provided several subclasses
# Write another method in the parent class that will be inherited by the 
# subclasses. Call it opponent. It should return which type of pokemon the 
#current type is weak and strong against, as a tuple.

# Grass is weak against Fire and strong against Water

# Ghost is weak against Dark and strong against Psychic

# Fire is weak against Water and strong against Grass

# Flying is weak against Electric and strong against Fighting

# For example, if the p_type of the subclass is 'Grass', .opponent() should 
# return the tuple ('Fire', 'Water')

class Pokemon():
    attack = 12
    defense = 10
    health = 15
    p_type = "Normal"

    def __init__(self, name,level = 5):
        self.name = name
        self.level = level
        self.weak = "Normal"
        self.strong = "Normal"

    def train(self):
        self.update()
        self.attack_up()
        self.defense_up()
        self.health_up()
        self.level = self.level + 1
        if self.level%self.evolve == 0:
            return self.level, "Evolved!"
        else:
            return self.level

    def attack_up(self):
        self.attack = self.attack + self.attack_boost
        return self.attack

    def defense_up(self):
        self.defense = self.defense + self.defense_boost
        return self.defense

    def health_up(self):
        self.health = self.health + self.health_boost
        return self.health

    def update(self):
        self.health_boost = 5
        self.attack_boost = 3
        self.defense_boost = 2
        self.evolve = 10
        
    def opponent(self):
        self.map = {'Grass': ('Fire', 'Water'), 'Ghost': ('Dark', 'Psychic'),
                    'Fire': ('Water', 'Grass'), 'Flying': ('Electric', 'Fighting')}
        return self.map[self.p_type]

    def __str__(self):
        self.update()
        return "Pokemon name: {}, Type: {}, Level: {}".format(self.name, self.p_type, self.level)

class Grass_Pokemon(Pokemon):
    attack = 15
    defense = 14
    health = 12
    p_type = "Grass"

    def update(self):
        self.health_boost = 6
        self.attack_boost = 2
        self.defense_boost = 3
        self.evolve = 12

class Ghost_Pokemon(Pokemon):
    p_type = "Ghost"

    def update(self):
        self.health_boost = 3
        self.attack_boost = 4
        self.defense_boost = 3

class Fire_Pokemon(Pokemon):
    p_type = "Fire"

class Flying_Pokemon(Pokemon):
    p_type = "Flying"

Flying_Pokemon('TestPokemon').opponent()



















    
    
    
    
    
    
    
    
    
    