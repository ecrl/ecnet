"""
EXAMPLE SCRIPT:
Find optimal values for a given set of value ranges, and a fitness function

Save the scores of each iteration in a text file called scores.txt
"""

from ecnet.abc import ABC

# Define a fitness function
def fitnessTest(values):
    fit = 0
    for val in values:
        fit+=val
    return fit

# Define a set of value ranges with types attached to them
values = [('int', (0,100)), ('int', (0,100)), ('int',(0,100)), ('float', (10,1000))]

# Create the abc object
abc = ABC(fitnessFunction = fitnessTest, amountOfEmployers = 100, valueRanges = values, endValue = 5)
# Run the colony until the fitness score reaches 5 or less
abc.runABC()
