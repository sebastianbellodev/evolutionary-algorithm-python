import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np

POPULATION_NUMBER = 100
BOARD_SIZE = 8
GENERATION_NUMBER = 10000
PARENTS_NUMBER = 2
MUTATION_PROBABILITY = 0.8
POSITION_CHANGE = 1
SELECTION_NUMBER = 5

def getFitness(population):
    ##Get the fitness of the population
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
    population = np.delete(population, -1, axis=1)
    DIMENSION = population[1,:].size
    parents = np.empty((SELECTION_NUMBER, DIMENSION), dtype=int)
    ##Get 10 random parents
    for i in range(SELECTION_NUMBER):
        parents[i] = random.choice(population)
    ##Sort the parents by the fitness
    parents = parents[np.argsort(parents[:,-1])]
    ##Get the best 2 parents
    parents = np.delete(parents, [-3,-2,-1], axis=0)
    return parents

def setVariation(parents):
    ##Get the position to make the crossover
    ##Get 4 random positions of the first parent
    first_position = random.sample(range(len(parents[0])), 4)
    ##Get the positions that are not in the first parent on the second parent
    second_position = [i for i in range(len(parents[0])) if i not in first_position]
    offsprings = []
    ##Make the crossover
    for i in range(0,BOARD_SIZE):
        if i in first_position:
            offsprings.append(parents[0][i])
        if i in second_position:
            offsprings.append(parents[1][i])
    return setMutation(offsprings)

def setMutation(offsprings):
    ##Make the mutation
    if random.random() < MUTATION_PROBABILITY:
        position_mutated = random.randint(0,7)
        ##Change the position of the queen
        ##Need to check if the position is not out of the board if is out change the position
        if(random.random() < 0.5):
            offsprings[position_mutated] = offsprings[position_mutated] + POSITION_CHANGE if offsprings[position_mutated] + POSITION_CHANGE <= 7 else offsprings[position_mutated] - POSITION_CHANGE
        else:
            offsprings[position_mutated] = offsprings[position_mutated] - POSITION_CHANGE if offsprings[position_mutated] - POSITION_CHANGE >= 0 else offsprings[position_mutated] + POSITION_CHANGE
    return offsprings

def setPopulation():
    ##Set the population with random boards
    population = np.zeros((POPULATION_NUMBER, BOARD_SIZE), dtype=int)
    for i in range(POPULATION_NUMBER):
        population[i,:] = np.random.permutation(BOARD_SIZE)
    return population

def setPopulationFitness(population):
    ##Get the fitness of the population
    fitness = getFitness(population)
    population = np.hstack((population, fitness))
    return population

def getOffsprings(population):
    ##Get the fitness of the population
    fitness = getFitness(population)
    population = np.hstack((population, fitness))
    offsprings = np.empty((0, BOARD_SIZE), dtype=int)
    ##Get the offsprings 10 times
    for i in range(10):
        parents = setParents(population)
        parents = np.delete(parents, -1, axis=1)
        offspring = setVariation(parents)
        offspring = np.array([offspring])
        offsprings = np.vstack((offsprings, offspring))
    return offsprings

def setEvolution(population):
    ##Get the offsprings
    childs = getOffsprings(population)
    fitnessChilds = setPopulationFitness(childs)
    ##Sort the population and get the worst 10 to replace with the childs
    sortIndexPopulation = np.argsort(population[:,8])
    populationReduced = np.delete(population, sortIndexPopulation[-10:], axis=0)
    populationNewGen = np.vstack((populationReduced, fitnessChilds))
    return populationNewGen

def getBinaryBoard(board):
    ##Show the board in binary representation
    binary = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    for i, queen in enumerate(board):
        for j in range(len(binary)):
            if j == queen:
                binary[i][j] = 1
    return binary

def geneticAlgorithm():
    ##Define population
    population = setPopulation()
    population = setPopulationFitness(population)
    ##Avoid find the solution in the population initialization
    sortedIndexPopulationCheck = np.argsort(population[:,8])
    populationSortedCheck = population[sortedIndexPopulationCheck]
    populationConvergenced = []
    if(populationSortedCheck[0][8] == 0):
        return geneticAlgorithm()
    ##Run the genetic algorithm until find the solution or reach the limit of generations
    for i in range(GENERATION_NUMBER):
        population = setEvolution(population)
        sortedIndexPopulation = np.argsort(population[:,8])
        populationSorted = population[sortedIndexPopulation]
        populationConvergenced.append(populationSorted[0][8])
        if(populationSorted[0][8] == 0):
            break
    ##Sort the population to get the best board
    sortedPopulation= np.argsort(population[:,8]) 
    population = population[sortedPopulation]
    # fitness = population[0][8]
    # print("Best board with", fitness, "attacks")
    # if(fitness == 0):
    #     solution = np.delete(population[0], -1, axis=0)
    #     board = getBinaryBoard(solution)
    #     print("Solution found!")
    #     print(board)
    ##Extract the fitness of each generation
    fitnessArrayGen = np.array(populationConvergenced)
    data = {"board": population[0], "fitness": population[0][8], "Success" : True if population[0][8] == 0 else False , "Generations Number": i, 'Fitness Gen': fitnessArrayGen}
    return data

def main():
    ##Run the genetic algorithm 30 times
    ##Extract the data of each run
    results = []
    for i in range(30):
        results.append(geneticAlgorithm())
    ##Set var to extract data of the fitness of each generation
    Success = 0
    SuccessGen = []
    for i in results:
        if i["Success"]:
            Success += 1
            SuccessGen.append(i["Generations Number"])
    ##Data
    print("Success number: ", Success)
    print("Success rate: ", Success/30)
    print("Average generations: ", sum(SuccessGen)/Success)
    print("Standard deviation: ", np.std(SuccessGen))
    print("Median: ", np.median(SuccessGen))
    print("Worse case: ", max(results[:8], key=lambda x:x["Generations Number"]))
    print("Best case: ", min(results[:8], key=lambda x:x["Generations Number"]))
    

    ##Graph the convergence of the fitness
    ##The best and the worst case to make the convergence analysis
    resultsSorted = sorted(results, key=lambda x: x["Generations Number"])
    resultsGenerationsFitness = []
    resultsGenerationsFitness.append(resultsSorted[0]['Fitness Gen'])
    resultsGenerationsFitness.append(resultsSorted[-1]['Fitness Gen'])
    ##print(getBinaryBoard(resultsSorted[0]['board']))
    # Colors to use in the plot
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # azul, verde, rojo, cyan, magenta, amarillo, negro
    # Graph the convergence 
    for i, array in enumerate(resultsGenerationsFitness):
        color = colors[i % len(colors)]  # Select Color
        plt.plot(array, color=color, label=f'Array {i+1}')  
    plt.xlabel('Índice')  # Tag X axis
    plt.ylabel('Valor')    # Tag Y axis
    plt.title('Convergence Graph')  
    plt.legend()  # Mostrar leyenda con etiquetas
    plt.grid(True)  # Mostrar una cuadrícula en el gráfico
    plt.show() 
    
main()
