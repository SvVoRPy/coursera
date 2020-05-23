# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:19:52 2020

@author: svollnhals

Python 3 Specialization on Coursera
Course 4 Python Classes and Inheritance
Week 03 Final Assignment
"""

### Wheel of Fortune Implementation
import os
os.chdir("""C:\\Users\\svollnhals\\Desktop\\Python-Specialization\\Course4_PythonClassesAndInheritance""")

# System sleep implementation
import time

for x in range(2, 6):
    print('Sleep {} seconds..'.format(x))
    time.sleep(x) # "Sleep" for x seconds
print('Done!')

# random number and letter generator

import random

rand_number = random.randint(1, 10)
print('Random number between 1 and 10: {}'.format(rand_number))

letters = [letter for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
rand_letter = random.choice(letters)
print('Random letter: {}'.format(rand_letter))

# lower upper and count for implementation
myString = 'Hello, World! 123'

print(myString.upper()) # HELLO, WORLD! 123
print(myString.lower()) # hello, world! 123
print(myString.count('l')) # 3

s = 'python is pythonic'
print(s.count('python')) # 2

### Helper Functions

import json
import random
import time

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Repeatedly asks the user for a number between min & max (inclusive)
def getNumberBetween(prompt, min, max):
    userinp = input(prompt) # ask the first time

    while True:
        try:
            n = int(userinp) # try casting to an integer
            if n < min:
                errmessage = 'Must be at least {}'.format(min)
            elif n > max:
                errmessage = 'Must be at most {}'.format(max)
            else:
                return n
        except ValueError: # The user didn't enter a number
            errmessage = '{} is not a number.'.format(userinp)

        # If we haven't gotten a number yet, add the error message
        # and ask again
        userinp = input('{}\n{}'.format(errmessage, prompt))

# Spins the wheel of fortune wheel to give a random prize
# Examples:
#    { "type": "cash", "text": "$950", "value": 950, "prize": "A trip to Ann Arbor!" },
#    { "type": "bankrupt", "text": "Bankrupt", "prize": false },
#    { "type": "loseturn", "text": "Lose a turn", "prize": false }
def spinWheel():
    with open("wheel.json", 'r') as f:
        wheel = json.loads(f.read())
        return random.choice(wheel)

# Returns a category & phrase (as a tuple) to guess
# Example:
#     ("Artist & Song", "Whitney Houston's I Will Always Love You")
def getRandomCategoryAndPhrase():
    with open("phrases.json", 'r') as f:
        phrases = json.loads(f.read())

        category = random.choice(list(phrases.keys()))
        phrase   = random.choice(phrases[category])
        return (category, phrase.upper())

# Given a phrase and a list of guessed letters, returns an obscured version
# Example:
#     guessed: ['L', 'B', 'E', 'R', 'N', 'P', 'K', 'X', 'Z']
#     phrase:  "GLACIER NATIONAL PARK"
#     returns> "_L___ER N____N_L P_RK"
def obscurePhrase(phrase, guessed):
    rv = ''
    for s in phrase:
        if (s in LETTERS) and (s not in guessed):
            rv = rv+'_'
        else:
            rv = rv+s
    return rv

# Returns a string representing the current state of the game
def showBoard(category, obscuredPhrase, guessed):
    return """
Category: {}
Phrase:   {}
Guessed:  {}""".format(category, obscuredPhrase, ', '.join(sorted(guessed)))

#### Test

category, phrase = getRandomCategoryAndPhrase()

guessed = []
for x in range(random.randint(10, 20)):
    randomLetter = random.choice(LETTERS)
    if randomLetter not in guessed:
        guessed.append(randomLetter)

print("getRandomCategoryAndPhrase()\n -> ('{}', '{}')"
      .format(category, phrase))

print("\n{}\n".format("-"*5))

print("obscurePhrase('{}', [{}])\n -> {}".format(phrase, ', '.join(["'{}'"
      .format(c) for c in guessed]), obscurePhrase(phrase, guessed)))

print("\n{}\n".format("-"*5))

obscured_phrase = obscurePhrase(phrase, guessed)
print("showBoard('{}', '{}', [{}])\n -> {}".format(phrase, obscured_phrase, ','
      .join(["'{}'".format(c) for c in guessed]), 
      showBoard(phrase, obscured_phrase, guessed)))

print("\n{}\n".format("-"*5))

num_times_to_spin = random.randint(2, 5)
print('Spinning the wheel {} times (normally this would just be done\
      once per turn)'.format(num_times_to_spin))

for x in range(num_times_to_spin):
    print("\n{}\n".format("-"*2))
    print("spinWheel()")
    print(spinWheel())


print("\n{}\n".format("-"*5))

print("In 2 seconds, will run getNumberBetween('Testing getNumberBetween().\
      Enter a number between 1 and 10', 1, 10)")

time.sleep(2)

print(getNumberBetween('Testing getNumberBetween(). Enter a number between \
                       1 and 10', 1, 10))

##############################################################################
## Define Player as Class

VOWEL_COST = 250
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
VOWELS = 'AEIOU'

# Write the WOFPlayer class definition (part A) here
class WOFPlayer:
    
    def __init__(self, name):
        self.name = name
        self.prizeMoney = 0
        self.prizes = []
        
    def addMoney(self, amt):
        self.prizeMoney += amt
        
    def goBankrupt(self):
        self.prizeMoney = 0
        
    def addPrize(self, prize):
        self.prizes.append(prize)
     
    def __str__(self):
        return '{} (${})'.format(self.name, self.prizeMoney)

# Write the WOFHumanPlayer class definition (part B) here
class WOFHumanPlayer(WOFPlayer):
           
    def getMove(self, category, obscuredPhrase, guessed):    
        
        def showBoard(category, obscuredPhrase, guessed):
            return """
            Category: {}
            Phrase:   {}
            Guessed:  {}""".format(category, obscuredPhrase, ', '.join(sorted(guessed)))       
        
        print("{} has ${}\n{}".format(self.name, self.prizeMoney,
              showBoard(category, obscuredPhrase, guessed)))
              
        userinp = input("Guess a letter, phrase, or type 'exit' or 'pass':")
           
        return userinp
            
category, phrase = getRandomCategoryAndPhrase()
guessed = []        
obscuredPhrase = obscurePhrase(phrase, guessed)
human_player = WOFHumanPlayer("Sven")
human_player.getMove(category, obscuredPhrase, guessed)

# Write the WOFComputerPlayer class definition (part C) here
class WOFComputerPlayer(WOFPlayer):
    
    VOWEL_COST = 250
    LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    VOWELS = 'AEIOU'
    SORTED_FREQUENCIES = 'ZQXJKVBPYGFWMUCLDRHSNIOATE'
    
    def __init__(self, name, difficulty):
        WOFPlayer.__init__(self, name)
        self.name = name
        self.difficulty = difficulty
        
    def smartCoinFlip(self):
        if random.randint(1, 10) > self.difficulty:
            return True
        else:
            return False
        
    def getPossibleLetters(self, guessed):
        if self.prizeMoney < self.VOWEL_COST:
            return list(filter(lambda x: x not in guessed + [char for char in self.VOWELS], self.LETTERS))
        else:
            return list(filter(lambda x: x not in guessed, self.LETTERS))
        
    def getMove(self, category, obscuredPhrase, guessed):
        rem_letters = self.getPossibleLetters(guessed)
        
        if set(rem_letters) <= set(self.VOWELS) and self.prizeMoney < self.VOWEL_COST:
            return 'pass'
        else:
            if self.smartCoinFlip() == True:
                return list(filter(lambda x: x in rem_letters, self.SORTED_FREQUENCIES))[-1]
            else:
                return random.choice(rem_letters)
        
        
guessed = ["R"]

computer_player = WOFComputerPlayer("AI", 2)
computer_player.smartCoinFlip()

computer_player.getPossibleLetters(guessed)
computer_player.prizeMoney = 300
computer_player.getPossibleLetters(guessed)

computer_player.getMove(category, obscuredPhrase, guessed)

random.choice(LETTERS)

