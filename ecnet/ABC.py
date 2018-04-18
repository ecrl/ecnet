'''
Copyright 2018 Hernan Gelaf-Romer
University of Massachusetts Lowell - CS
ECNET Research Team
'''
from random import randint
import numpy as np
import sys as sys

'''
Artificial bee colony used to optimize parameters given any range of parameter values, their types, and a fitness function. The fitness
function will be passed to the constructor and called when each individual bee must generate a fitness score for its current set of 
values that it contains.
Was meant to be a wrapper class for the ecnet neural network.
More information on artificial bee colonies can be found here : https://abc.erciyes.edu.tr/
'''

class ABC:

    def __init__(self, valueRanges, fitnessFunction=None, endValue = None, iterationAmount = None, amountOfEmployers = 50):

        
        '''
        Initialize the program by assigning 50 worker bees, one scout, and one onlooker bee which will then be called upon by the program.
        
        Pass an end value, which is the target fitness score you would like to achieve before the program terminates. 
        
        
        '''
        
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
        
        for i in range(amountOfEmployers):
            sys.stdout.flush()
            sys.stdout.write("Creating bee number: %d \r" % (i + 1))
            self.employers.append(Bee('employer', generateRandomValues(self.valueRanges)))
            self.employers[i].currFitnessScore = self.fitnessFunction(self.employers[i].values)
        print("***DONE INITIALIZING***")
            
    '''
     Assign a new position to the given bee, firstBee and secondBee are represented in the form of index values for the list of all bees 
     inside the employers list. 
     
     Careful not to assign the same bees upon running the function, as it will result in one invalid bee until the bee is updated by the 
     program once again, precautions were taken (while loop).
     
    '''
    def assignNewPositions(self, firstBee):
        valueTypes = [t[0] for t in self.valueRanges]
        secondBee = randint(0, len(self.employers) -1)
        while (secondBee == firstBee):
            secondBee = randint(0, len(self.employers) -1)
        self.onlooker.getPosition(self.employers, firstBee, secondBee, self.fitnessFunction, valueTypes)

    '''
    This function will collect the average of all the fitness scores across all employer bees. The average will be used as a model of comparison
    '''
    def getFitnessAverage(self):
        self.fitnessAverage = 0
        for employer in self.employers:
            self.fitnessAverage += employer.currFitnessScore

            if self.bestFitnessScore == None or employer.currFitnessScore < self.bestFitnessScore:
                self.bestFitnessScore = employer.currFitnessScore
                self.bestValues = employer.values
                
        self.fitnessAverage /= len(self.employers)

    '''
    Check if the new positions are better than the average fitness scores, if not assig a new random position to the employer bee 
    and calculate it's fitness score. 
    
    Fitness scores are calculated by running the function that was passed to the constructor upon initialization.
       
    '''
    def checkNewPositions(self, bee):
        if bee.currFitnessScore  > self.fitnessAverage:
            bee.values = generateRandomValues(self.valueRanges)
            bee.currFitnessScore = self.fitnessFunction(bee.values)

    '''
    Check if any current fitness scores are below the end value. If so exit the program, and log all the bee positions which are below
    the end value.
    
    If there is more than one bee with a fitness score of a current location below the end value, then log them all.
    
    ''' 
    def checkIfDone(self):
        keepGoing = True
        for employer in self.employers:
            if employer.currFitnessScore <= self.endValue:
                print("Fitness score =", employer.currFitnessScore)
                print("Values =", employer.values)
                keepGoing = False
        return keepGoing

    '''
    Run the artificial bee colony, generally this is the only method that you will need to use, as the other methods are called here.
    
    A good example of how to run this program properly are as follows :
    
    abc = ABC(valueRanges=values,amountOfEmployers=10,iterationAmount=10, fitnessFunction=runNeuralNet)
    abc.runABC()        # Run the artificial bee colony
    
    '''
    def runABC(self):
        running = True

        if self.endValue != None:
            while True:
                print("Assigning new positions")
                for i in range(len(self.employers)):
                    sys.stdout.flush()
                    sys.stdout.write('At bee number: %d \r' % (i+1))
                    self.assignNewPositions(i)
                print("Getting fitness average")
                self.getFitnessAverage()
                print("Checking if done")
                running = self.checkIfDone()
                saveScore(self.bestFitnessScore, self.bestValues)
                if running == False:
                    break
                print("Current fitness average:", self.fitnessAverage)
                print("Checking new positions, assigning random positions to bad ones")
                for employer in self.employers:
                    self.checkNewPositions(employer)

                print("Best score:", self.bestFitnessScore)
                print("Best value:", self.bestValues)
        
        elif self.iterationAmount != None:
            count = 0

            while count < self.iterationAmount:
                print("Assigning new positions")
                for i in range(len(self.employers)):
                    sys.stdout.flush()
                    sys.stdout.write('At bee number: %d \r' % (i+1))
                    self.assignNewPositions(i)
                print("Getting fitness average")
                self.getFitnessAverage()
                saveScore(self.bestFitnessScore, self.bestValues)
                print("Current fitness average:", self.fitnessAverage)
                print("Checking new positions, assigning random positions to bad ones")
                for employer in self.employers:
                    self.checkNewPositions(employer)

                print("Best score:", self.bestFitnessScore)
                print("Best value:", self.bestValues)
                print("Iteration {} / {}".format(count+1, self.iterationAmount))
                count+= 1
        
        return self.bestValues





'''
Class which contains the individual bees that will be used to run the artifical bee colony programmed in ABC.py. More information on how
that program works can be found within that file.
More information on how an artificial bee colony works can be found here : https://abc.erciyes.edu.tr/.
'''

class Bee:
    
    '''
    Each be must be given a type, which can be Worker/Onlooker/Scout, documentation on the responsibility of each bee can be found below, 
    and also at https://abc.erciyes.edu.tr/.
    
    Each employer bee will store its position, or set of values that can be run through the ECNet neural network in order to obtain the fitness score, 
    or RMSE values of the outputs produced the by the value sets.
    
    Each employer bee also stores its current fitness score.
    
    '''
    def __init__(self, beeType, values=[]):

        self.beeType = beeType
        
        if beeType == "employer":               # Only the employer bees should store values/fitness scores
            self.values = values            
            self.currFitnessScore = None

    '''
    Onlooker Bee Functions
    
    The onlooker bee is responsible for comparing the positions of random bees and passing it to the first worker bee, which will then 
    test the new position compared to its old, if the new position is better, it will store the new values, and replace its current fitness
    score.
    
    The valueFunction implementation can be found in functions.py, and is the mathematical standard when it comes to comparing positions
    in regards to artificial bee colonies. 
    
    
    '''

    def getPosition(self, beeList, firstBee, secondBee, fitnessFunction, valueTypes):
        newValues = []
        currValue = 0

        for i in range(len(valueTypes)):
            currValue = valueFunction(beeList[firstBee].values[i], beeList[secondBee].values[i])

            if valueTypes[i] == 'int':
                currValue = int(currValue)
            newValues.append(currValue)

        beeList[firstBee].getFitnessScore(newValues, fitnessFunction)

    '''
    Employer Bee Functions
    
    The employer bee's sole function is to calculate the fitness score of the new position it is passed, and compare it to its current 
    fitness score. If the new fitness score is better, then the old position will be replaced with the new position, as well as the fitness
    scores.
    
    '''

    def getFitnessScore(self, values, fitnessFunction):
        if self.beeType != "employer":
            raise RuntimeError("Cannot get fitness score on a non-employer bee")
        else:
            fitnessScore = fitnessFunction(values)  # Your fitness function must take a certain set of values that you would like to optimize
            if self.currFitnessScore == None or fitnessScore < self.currFitnessScore:
                self.value = values
                self.currFitnessScore = fitnessScore





'''
Private functions that are used to assist the ABC
'''

def generateRandomValues(value_ranges):
    values = []
    if value_ranges == None:
        raise RuntimeError("must set the type/range of possible values")
    else:
        for t in value_ranges:  # t[0] contains the type of the value, t[1] contains a tuple (min_value, max_value)
            if t[0] == 'int':   # If they type they would like this random value to be is an int
                values.append(randint(t[1][0], t[1][1]))
            elif t[0] == 'float':   # If the type they would this random value to be is a float
                values.append(np.random.uniform(t[1][0], t[1][1]))
            else:
                raise RuntimeError("value type must be either an 'int' or a 'float'")
    return values

def valueFunction(a, b):  # Method of generating a value in between the values given
    activationNum = np.random.uniform(-1, 1)
    return a + abs(activationNum * (a - b))

def saveScore(score, values, filename = 'scores.txt'):   # Function for saving the scores of each iteration onto a file
    f = open(filename, 'a')
    string = "Score: {} Values: {}".format(score, values)
    f.write(string)
    f.write('\n')
    f.close()
