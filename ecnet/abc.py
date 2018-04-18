#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ecnet/abc.py
#  v.1.2.7.dev1
#  Developed in 2018 by Hernan Gelaf-Romer <hernan_gelafromer@student.uml.edu>
#
#  This program implements an artificial bee colony to tune ecnet hyperparameters
#

# 3rd party packages (open src.)
from random import randint
import numpy as np
import sys as sys

### Artificial bee colony object, which contains multiple bee objects ###
class ABC:

    def __init__(self, valueRanges, fitnessFunction=None, endValue = None, iterationAmount = None, amountOfEmployers = 50):
        if endValue == None and iterationAmount == None:
            raise ValueError("must select either an iterationAmount or and endValue")
        if fitnessFunction == None:
            raise ValueError("must pass a fitness function")
        print("***INITIALIZING***")
        self.valueRanges = valueRanges
        self.fitnessFunction = fitnessFunction
        self.employers = []
        self.bestValues = []                    # Store the values that are currently performing the best
        self.onlooker = Bee('onlooker')
        self.bestFitnessScore = None           # Store the current best Fitness Score
        self.fitnessAverage = 0
        self.endValue = endValue
        self.iterationAmount = iterationAmount
        # Initialize employer bees, assign them values/fitness scores
        for i in range(amountOfEmployers):
            sys.stdout.flush()
            sys.stdout.write("Creating bee number: %d \r" % (i + 1))
            self.employers.append(Bee('employer', generateRandomValues(self.valueRanges)))
            self.employers[i].currFitnessScore = self.fitnessFunction(self.employers[i].values)
        print("***DONE INITIALIZING***")
     
    ### Assign a new position to the given bee
    def assignNewPositions(self, firstBee):
        valueTypes = [t[0] for t in self.valueRanges]
        secondBee = randint(0, len(self.employers) -1)
        # Avoid both bees being the same
        while (secondBee == firstBee):
            secondBee = randint(0, len(self.employers) -1)
        self.onlooker.getPosition(self.employers, firstBee, secondBee, self.fitnessFunction, valueTypes)
    
    ### Collect the average fitness score across all employers
    def getFitnessAverage(self):
        self.fitnessAverage = 0
        for employer in self.employers:
            self.fitnessAverage += employer.currFitnessScore
            # While iterating through employers, look for the best fitness score/value pairing
            if self.bestFitnessScore == None or employer.currFitnessScore < self.bestFitnessScore:
                self.bestFitnessScore = employer.currFitnessScore
                self.bestValues = employer.values      
        self.fitnessAverage /= len(self.employers)
    
    ### Check if new position is better than current position held by a bee
    def checkNewPositions(self, bee):
        # Update the bee's fitness/value pair if the new location is better
        if bee.currFitnessScore  > self.fitnessAverage:
            bee.values = generateRandomValues(self.valueRanges)
            bee.currFitnessScore = self.fitnessFunction(bee.values)

    ### If termination depends on a target value, check to see if it has been reached
    def checkIfDone(self, count):
        keepGoing = True
        if self.endValue != None:
            for employer in self.employers:
                if employer.currFitnessScore <= self.endValue:
                    print("Fitness score =", employer.currFitnessScore)
                    print("Values =", employer.values)
                    keepGoing = False
        elif count >= self.iterationAmount:
            keepGoing = False
        return keepGoing
    
    ### Run the artificial bee colony
    def runABC(self):
        running = True
        count = 0

        while True:
            print("Assigning new positions")
            for i in range(len(self.employers)):
                sys.stdout.flush()
                sys.stdout.write('At bee number: %d \r' % (i+1))
                self.assignNewPositions(i)
            print("Getting fitness average")
            self.getFitnessAverage()
            print("Checking if done")
            count+=1
            running = self.checkIfDone(count)
            if running == False and self.endValue != None:
                saveScore(self.bestFitnessScore, self.bestValues)
                break
            print("Current fitness average:", self.fitnessAverage)
            print("Checking new positions, assigning random positions to bad ones")
            for employer in self.employers:
                self.checkNewPositions(employer)
            print("Best score:", self.bestFitnessScore)
            print("Best value:", self.bestValues)
            if self.iterationAmount != None:
                print("Iteration {} / {}".format(count, self.iterationAmount))
            if running == False:
                saveScore(self.bestFitnessScore, self.bestValues)
                break
            saveScore(self.bestFitnessScore, self.bestValues)

        return self.bestValues


### Bee object, employers contain value/fitness
class Bee:
    
    def __init__(self, beeType, values=[]):
        self.beeType = beeType
        # Only the employer bees should store values/fitness scores
        if beeType == "employer":               
            self.values = values            
            self.currFitnessScore = None

    ### Onlooker bee function, create a new set of positions
    def getPosition(self, beeList, firstBee, secondBee, fitnessFunction, valueTypes):
        newValues = []
        currValue = 0
        for i in range(len(valueTypes)):
            currValue = valueFunction(beeList[firstBee].values[i], beeList[secondBee].values[i])
            if valueTypes[i] == 'int':
                currValue = int(currValue)
            newValues.append(currValue)
        beeList[firstBee].getFitnessScore(newValues, fitnessFunction)

    #### Employer bee function, get fitness score for a given set of values
    def getFitnessScore(self, values, fitnessFunction):
        if self.beeType != "employer":
            raise RuntimeError("Cannot get fitness score on a non-employer bee")
        else:
            # Your fitness function must take a certain set of values that you would like to optimize
            fitnessScore = fitnessFunction(values)  
            if self.currFitnessScore == None or fitnessScore < self.currFitnessScore:
                self.value = values
                self.currFitnessScore = fitnessScore

### Private functions to be called by ABC

### Generate a random set of values given a value range
def generateRandomValues(value_ranges):
    values = []
    if value_ranges == None:
        raise RuntimeError("must set the type/range of possible values")
    else:
        # t[0] contains the type of the value, t[1] contains a tuple (min_value, max_value)
        for t in value_ranges:  
            if t[0] == 'int':
                values.append(randint(t[1][0], t[1][1]))
            elif t[0] == 'float':
                values.append(np.random.uniform(t[1][0], t[1][1]))
            else:
                raise RuntimeError("value type must be either an 'int' or a 'float'")
    return values

### Method of generating a value in between the values given
def valueFunction(a, b):  
    activationNum = np.random.uniform(-1, 1)
    return a + abs(activationNum * (a - b))

### Function for saving the scores of each iteration onto a file
def saveScore(score, values, filename = 'scores.txt'):
    f = open(filename, 'a')
    string = "Score: {} Values: {}".format(score, values)
    f.write(string)
    f.write('\n')
    f.close()
