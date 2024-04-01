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

def checkFitness2():
    population = np.array([[0, 4, 7, 5, 2, 6, 1, 3],
                       [0, 4, 7, 5, 2, 6, 1, 3]])
    fitness = np.zeros((population[:,1].size, 1), dtype=int)
    for i, array in enumerate(population):
        for j, column in enumerate(array):
            for k, solution in enumerate(array[j:]):
                ## Check if there is a queen in the same diagonal
                if (abs(j-k) == abs(column-solution)):
                    fitness[i,0] += 1
    print(fitness)

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

def crossover(parents):
    first_position = random.sample(range(len(parents[0])), 4)
    second_position = [i for i in range(len(parents[0])) if i not in first_position]
    child = []
    for i in range(0,8):
        if i in first_position:
            #print("number", parents[0][i], "in position", i, "first parent")
            child.append(parents[0][i])
        if i in second_position:
            #print("number", parents[1][i], "in position", i, "second parent")
            child.append(parents[1][i])
    return mutation(child)

def mutation(child):
    if random.random() < MUTATION_PROBABILITY:
        #print("Mutation")
        position_mutated = random.randint(0,7)
        
        if(random.random() < 0.5):
            child[position_mutated] = child[position_mutated] + POSITION_CHANGE if child[position_mutated] + POSITION_CHANGE <= 7 else child[position_mutated] - POSITION_CHANGE
        else:
            child[position_mutated] = child[position_mutated] - POSITION_CHANGE if child[position_mutated] - POSITION_CHANGE >= 0 else child[position_mutated] + POSITION_CHANGE
    return child

## Build array of boards

def buildPopulation():
    population = np.zeros((POPULATION_NUMBER, BOARD_SIZE), dtype=int)
    for i in range(POPULATION_NUMBER):
        ## Random permutation of the matrix rows to position the queens
        population[i,:] = np.random.permutation(BOARD_SIZE)
    return population

def fitPopulation(population):
    fitness = checkFitness(population)
    population = np.hstack((population, fitness))
    return population

def getChilds(population):
    fitnessPopulation = fitPopulation(population)
    childs = np.empty((0, BOARD_SIZE), dtype=int)
    for i in range(10):
        parents = selectParent(fitnessPopulation)
        parents = np.delete(parents, -1, axis=1)
        child = crossover(parents)
        child = np.array(child)
        childs = np.vstack((childs, child))
    return childs

def reproductionPopulation(population):
    childs = getChilds(population)
    ##print("Childs", childs)
    fitnessChilds = fitPopulation(childs)
    ##
    sortIndexPopulation = np.argsort(population[:,8])
    populationReduced = np.delete(population, sortIndexPopulation[-10:], axis=0)
    ##
    populationNewGen = np.vstack((populationReduced, fitnessChilds))
    return populationNewGen

def geneticAlgorithm():
    population = buildPopulation()
    population = fitPopulation(population)
    for i in range(GENERATION_NUMBER):
        population = reproductionPopulation(population)
        
    populationSort = np.argsort(population[:,8]) 
    population = population[populationSort]
    # for i, array in enumerate(population):
    #     print("Board", i, "with", array[8], "attacks")
    #     print(array)
    print("Best board with", population[0][8], "attacks")
    if(population[0][8] == 0):
        print("Solution found")
        print(population[0])
        exit()
    
    
    
checkFitness2()   
##geneticAlgorithm()
    
    
# for i in range(1000):
#      geneticAlgorithm()

## Check the number of attacks per array in the population matrix
##fitness = checkFitness(population)
## Tuple that links the fitness value (number of attacks) with each row (board) in the population matrix
##population = np.hstack((population, fitness))

##mean = np.zeros(GENERATION_NUMBER, dtype=int)

##parents = selectParent(population)
## Delete the fitness value column from each parent
##parents = np.delete(parents, -1, axis=1)


