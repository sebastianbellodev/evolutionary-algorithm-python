import numpy as np
import random

POPULATION_NUMBER = 100
BOARD_SIZE = 8
GENERATION_NUMBER = 100
PARENTS_NUMBER = 2
MUTATION_PROBABILITY = 0.8
POSITION_CHANGE = 1

## Check the number of attacks per array in the population matrix
def checkFitness(population):
    fitness = np.zeros((population[:,1].size, 1), dtype=int)
    for i, array in enumerate(population):
        for j, column in enumerate(array):
            for k, solution in enumerate(array[j:]):
                ## Check if there is a queen in the same diagonal
                if (abs(j-k) == abs(column-solution)):
                    fitness[i,0] += 1
    return fitness

def selectParent(population):
    SELECTION_NUMBER = 5
    DIMENSION = population[1,:].size
    parents = np.empty((SELECTION_NUMBER, DIMENSION), dtype=int)
    ## Select five random boards from the population
    for i in range(SELECTION_NUMBER):
        parents[i] = random.choice(population)
    ## Sort the boards by the number of attacks
    parents = parents[np.argsort(parents[:,-1])]
    ## Delete the three boards with the highest number of attacks
    parents = np.delete(parents, [-3,-2,-1], axis=0)
    return parents 

## Build array of boards
population = np.zeros((POPULATION_NUMBER, BOARD_SIZE), dtype=int)
for i in range(POPULATION_NUMBER):
    ## Random permutation of the matrix rows to position the queens
    population[i,:] = np.random.permutation(BOARD_SIZE)

## Check the number of attacks per array in the population matrix
fitness = checkFitness(population)
## Tuple that links the fitness value (number of attacks) with each row (board) in the population matrix
population = np.hstack((population, fitness))

mean = np.zeros(GENERATION_NUMBER, dtype=int)

parents = selectParent(population)
## Delete the fitness value column from each parent
parents = np.delete(parents, -1, axis=1)