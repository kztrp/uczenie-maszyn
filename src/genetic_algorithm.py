from individual import Individual
from math import ceil, floor
from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime
import random
import numpy as np


class GeneticAlgorithm:
    def __init__(self, data):
        self.best_scores = []
        self.data = data
        self.lowest_penalties = []
        self.population = []
        self.type = "Genetyczny"

    # def __repr__(self):
    #     if len(self.best_scores) == 0 or len(self.lowest_penalties) == 0:
    #         return f"GeneticAlgorithm({self.best_scores}, " \
    #                f"'{self.graph.name}', {self.lowest_penalties[-1]}, " \
    #                f" '{self.type}')"
    #     else:
    #         return f"GeneticAlgorithm({self.best_scores[-1]}, " \
    #                f"'{self.graph.name}', {self.lowest_penalties[-1]}, " \
    #                f"'{self.type}')"

    def generate_population(self, size, n_features):
        for i in range(size):
            individual = Individual(n_features, self.data)
            individual.generate_genotype()
            self.population.append(individual)

    def roulette_breeding(self, iteration, fitness_index, mutation_rate=0.05, crossing_rate=0.85):
        total_score = 0
        total_penalty = 0
        copied_size = floor((1 - crossing_rate) * len(self.population))
        bred_size = ceil(crossing_rate * len(self.population))
        if bred_size % 2 == 1:
            copied_size += 1
            bred_size -= 1
        self.population.sort(key=lambda x: (x.fitness_scores[fitness_index]), reverse=True)
        new_population = []
        for i in range(copied_size):
            new_population.append(self.population[i])
        for individual in self.population:
            total_score += individual.fitness_scores[fitness_index]
        for i in range(round(bred_size / 2)):
            found = [False, False]
            selected_value = [random.randint(0, int(total_score)),
                              random.randint(0, int(total_score))]
            parents = []
            for individual in self.population:
                selected_value[0] -= individual.fitness_scores[fitness_index]
                selected_value[1] -= individual.fitness_scores[fitness_index]
                if not found[0] and selected_value[0] <= 0:
                    parents.append(individual)
                    found[0] = True
                if not found[1] and selected_value[1] <= 0:
                    parents.append(individual)
                    found[1] = True
                if found[0] and found[1]:
                    break
            ind_a = Individual(self.data.shape[1]-1, self.data)
            ind_b = Individual(self.data.shape[1]-1, self.data)
            split_points = np.array([random.randint(0, ind_a.genotype_features.size-1), random.randint(0, ind_a.genotype_algorithms.size-1)])
            ind_a.breeding(parents[0], parents[1], mutation_rate, split_points)
            ind_b.breeding(parents[1], parents[0], mutation_rate, split_points)
            ind_a.iteration = iteration
            ind_b.iteration = iteration
            new_population.append(ind_a)
            new_population.append(ind_b)

        new_population.sort(key=lambda x: (x.fitness_scores[fitness_index]), reverse=True)
        self.population = new_population


    def run_algorithm(self, iterations, fitness_index, mutation_rate=0.05, crossing_rate=0.85):
        for i in range(iterations):
            self.roulette_breeding(i, fitness_index, mutation_rate, crossing_rate, )
            if i % 10 == 0 or i == iterations - 1:
                text = (f"Iteracja {i} \nNajlepszy osobnik:\n {self.population[0].genotype_features},{self.population[0].genotype_algorithms}\nWyniki:"\
                        f"{self.population[0].scores}\nFunkcje celu:{self.population[0].fitness_scores}")
                print(text)



    # def save_charts(self, console):
    #     fig, ax = plt.subplots(2, 1)
    #     ax[0].set_title(f"Graf o liczbie wierzchołków: {self.graph.nodes}"
    #                     f" i liczbie krawędzi: {self.graph.number_of_edges}")
    #     ax[0].plot(self.best_scores)
    #     ax[0].set_xlabel("Liczba iteracji")
    #     ax[0].set_ylabel("Liczba kolorów")
    #     ax[1].set_xlabel("Liczba iteracji")
    #     ax[1].set_ylabel("Liczba punktów karnych")
    #
    #     ax[1].plot(self.lowest_penalties)
    #     filename = str(Path(__file__).parent.parent) + \
    #                (f"/saved_charts/{str(datetime.now())}.png")
    #     fig.savefig(filename)
    #     print(f"Wyeksportowano wykres do '{filename}'.")
    #
    # def export_results(self, parameters, console):
    #     filename = str(Path(__file__).parent.parent) + \
    #                f"/exported_results/{str(datetime.now())}.txt"
    #     try:
    #         with open(filename, 'w+') as f:
    #             if parameters != "":
    #                 for x, y in parameters.items():
    #                     f.write("{}: {}\n".format(x, y))
    #             f.write("\n\n")
    #             for individual in self.population:
    #                 f.write(str(individual))
    #                 f.write('\n')
    #         print(f"Wyeksportowano wyniki do 'exported_results/{filename}'.")
    #         return True
    #     except NotADirectoryError and FileNotFoundError:
    #         return False
