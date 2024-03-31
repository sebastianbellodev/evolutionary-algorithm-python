import numpy as np
import random

POPULATION_NUMBER = 100
BOARD_SIZE = 8
GENERATION_NUMBER = 100


## Check the number of attacks per array in the population matrix
def checkFitness(population):
    fitness = np.zeros((population[:,1].size, 1), dtype=int)
    for i, array in enumerate(population):
        for j, column in enumerate(array):
            for k, solution in enumerate(array[j:]):
                ## Check if there is a queen in the same diagonal
                if (abs(j-k) == abs(column-solution)):
                    fitness[i, 0] += 1
    return fitness

## Build array of boards
population = np.zeros((POPULATION_NUMBER, BOARD_SIZE), dtype=int)
for i in range(POPULATION_NUMBER):
    ## Random permutation of the array that represents the queen row for each column in the board
    population[i,:] = np.random.permutation(BOARD_SIZE)

fitness = checkFitness(population)
## Add the fitness value (number of attacks) to each array (board) in the population matrix
population = np.hstack((population, fitness))
mean = np.zeros(GENERATION_NUMBER, dtype=int)

print(len(mean))