# -*- coding: utf-8 -*-
"""
Created on Fri May 08 17:44:17 2024

@author: Lucas Friedrich
"""

import numpy as np

class GeneticAlgorithm:
    def __init__(self, num_generations, population_size, num_parents, professors_times, rooms_capacity, courses):
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents = num_parents
        self.professors_times = professors_times
        self.rooms_capacity = rooms_capacity
        self.courses = courses
        self.num_courses = len(courses)
        self.num_professors = len(professors_times)
        self.num_rooms = len(rooms_capacity)
        self.num_time_slots = 10
        self.best_solution = None
        self.best_fitness = float('-inf')

    def fitness_schedule(self, chromosome):
        conflicts = 0
        for i, course in enumerate(self.courses):
            professor, room, time_slot = chromosome[i]
            if time_slot not in self.professors_times[professor]:
                conflicts += 1
            if self.rooms_capacity[room] < course['students']:
                conflicts += 1
        return -conflicts 
    
    
    def generate_population(self):
        return [np.array([(np.random.randint(0, self.num_professors),
                           np.random.randint(0, self.num_rooms),
                           np.random.randint(0, self.num_time_slots)) for _ in range(self.num_courses)]) for _ in range(self.population_size)]

    def select_parents(self, population, fitness_scores):
        fitness_scores = fitness_scores - np.min(fitness_scores)

        total = np.sum(fitness_scores)
        if total <= 0:
            probabilities = np.ones_like(fitness_scores) / len(fitness_scores)
        else:
            probabilities = fitness_scores / total


        parents_indices = np.random.choice(np.arange(len(population)), size=self.num_parents, p=probabilities)
        return np.array(population)[parents_indices]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        return np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

    def mutation(self, chromosome, mutation_rate=0.05):
        for i in range(len(chromosome)):
            if np.random.rand() < mutation_rate:
                chromosome[i] = (np.random.randint(0, self.num_professors),
                                 np.random.randint(0, self.num_rooms),
                                 np.random.randint(0, self.num_time_slots))
        return chromosome

    def run(self):
        population = self.generate_population()
        for _ in range(self.num_generations):
            fitness_scores = np.array([self.fitness_schedule(individual) for individual in population])
            parents = self.select_parents(population, fitness_scores)

            next_population = []
            for _ in range(int(self.population_size / 2)):
                parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
                child1 = self.mutation(self.crossover(parent1, parent2))
                child2 = self.mutation(self.crossover(parent2, parent1))
                next_population.extend([child1, child2])

            population = next_population
            current_best = np.max(fitness_scores)
            if current_best > self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = population[np.argmax(fitness_scores)]

        return self.best_solution

professors_times = {0: [0, 1, 2], 1: [1, 2, 3]}
rooms_capacity = {0: 30, 1: 50}
courses = [{'name': 'Matemática', 'students': 25}, {'name': 'Física', 'students': 35}]

ga = GeneticAlgorithm(100, 50, 20, professors_times, rooms_capacity, courses)
best_schedule = ga.run()
print("Melhor horário:", best_schedule)
