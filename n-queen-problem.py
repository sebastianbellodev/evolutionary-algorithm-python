import numpy as np
import random

POPULATION_NUMBER = 100
BOARD_SIZE = 8
GENERATION_NUMBER = 100
PARENTS_NUMBER = 2
MUTATION_PROBABILITY = 0.8
POSITION_CHANGE = 1

def getFitness(population):
    fitness = np.zeros((population[:,1].size, 1), dtype=int)
    for i, board in enumerate(population):
        fitness[i,0] += abs(len(board) - len(np.unique(board))) * 2
        for j in range(len(board)):
            for k in range(len(board)):
                if (j != k):
                    if (abs(j-k) == abs(board[j]-board[k])):
                        fitness[i,0] += 1
    return fitness

def setParents(population):
    SELECTION_NUMBER = 5
    DIMENSION = population[1,:].size
    parents = np.empty((SELECTION_NUMBER, DIMENSION), dtype=int)
    for i in range(SELECTION_NUMBER):
        parents[i] = random.choice(population)
    parents = parents[np.argsort(parents[:,-1])]
    parents = np.delete(parents, [-3,-2,-1], axis=0)
    return parents

def setVariation(parents):
    first_position = random.sample(range(len(parents[0])), 4)
    second_position = [i for i in range(len(parents[0])) if i not in first_position]
    offsprings = []
    for i in range(0,BOARD_SIZE):
        if i in first_position:
            offsprings.append(parents[0][i])
        if i in second_position:
            offsprings.append(parents[1][i])
    return setMutation(offsprings)

def setMutation(offsprings):
    if random.random() < MUTATION_PROBABILITY:
        position_mutated = random.randint(0,7)
        if(random.random() < 0.5):
            offsprings[position_mutated] = offsprings[position_mutated] + POSITION_CHANGE if offsprings[position_mutated] + POSITION_CHANGE <= 7 else offsprings[position_mutated] - POSITION_CHANGE
        else:
            offsprings[position_mutated] = offsprings[position_mutated] - POSITION_CHANGE if offsprings[position_mutated] - POSITION_CHANGE >= 0 else offsprings[position_mutated] + POSITION_CHANGE
    return offsprings

def setPopulation():
    population = np.zeros((POPULATION_NUMBER, BOARD_SIZE), dtype=int)
    for i in range(POPULATION_NUMBER):
        population[i,:] = np.random.permutation(BOARD_SIZE)
    return population

def setPopulationFitness(population):
    fitness = getFitness(population)
    population = np.hstack((population, fitness))
    return population

def getOffsprings(population):
    fitness = getFitness(population)
    population = np.hstack((population, fitness))
    offsprings = np.empty((0, BOARD_SIZE), dtype=int)
    for i in range(10):
        parents = setParents(population)
        parents = np.delete(parents, -1, axis=1)
        offspring = setVariation(parents)
        offspring = np.array([offspring])
        offsprings = np.vstack((offsprings, offspring))
    return offsprings

def setEvolution(population):
    childs = getOffsprings(population)
    fitnessChilds = setPopulationFitness(childs)
    sortIndexPopulation = np.argsort(population[:,8])
    populationReduced = np.delete(population, sortIndexPopulation[-10:], axis=0)
    populationNewGen = np.vstack((populationReduced, fitnessChilds))
    return populationNewGen

def main():
    population = setPopulation()
    population = setPopulationFitness(population)
    for i in range(GENERATION_NUMBER):
        population = setEvolution(population)
        
    populationSort = np.argsort(population[:,8]) 
    population = population[populationSort]
    solution = population[0][8]
    print("Best board with", solution, "attacks")
    if(solution == 0):
        return print("Solution found", solution)
      
main()
