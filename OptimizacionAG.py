import random
import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools

# Definir la función objetivo
def sphere(x):
    return sum(xi**2 for xi in x)

# Definir los parámetros del algoritmo genético
POP_SIZE = 50  # Tamaño de la población
CXPB = 0.9      # Probabilidad de cruce
MUTPB = 0.1     # Probabilidad de mutación
NUM_GENERATIONS = 50 # Número máximo de generaciones
HALL_OF_FAME_SIZE = 1 # Tamaño del Hall of Fame

# Definir la función principal
def main():
    # Crear la clase de aptitud
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Crear la clase de individuo
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Registrar los componentes en el toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -5.12, 5.12)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                    toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalSphere)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Crear la población inicial
    pop = toolbox.population(n=POP_SIZE)

    # Registrar el Hall of Fame
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # Registrar estadísticas de la evolución
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # Iniciar la evolución
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, 
                                    ngen=NUM_GENERATIONS, stats=stats, 
                                    halloffame=hof, verbose=True)

    # Imprimir la mejor solución encontrada
    best = hof[0]
    print("Mejor solucion: ", best, " Valor optimo: ", best.fitness.values[0])

    # Mostrar la evolución de la función de desempeño
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
    

    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_max, "b-")
    
    ax1.set_xlabel("Generacion")
    ax1.set_ylabel("Evolucion de la función de desempeño")
    ax1.legend(loc="lower right")
    plt.show()

# Definir la función de evaluación de la aptitud
def evalSphere(individual):
    return sphere(individual),

if __name__ == "__main__":
    main()



