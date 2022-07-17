# -*- coding: utf-8 -*-

import numpy as np
import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


########################################################################################################################
# build interface - capitalized variables indicate parameter inputs for the GA.
########################################################################################################################

title_label = st.header("Genetic Algorithm - Kordel France (kfranceF0DE80)")
desc_label = st.subheader("Set your hyperparameters on the left, and then click 'Run'")
DIMENSIONS: int = st.sidebar.slider("Select the dimensionality of the data", min_value=2, max_value=20, value=3, key='dimensions')
dimensions_caption = st.sidebar.caption("A graph of the population will be shown for dimensions less than 4. The"
                                        " boundaries for each function you select are automatically set according input"
                                        " domains in which they are valid.")

OBJ_FUNCTION_TYPE = st.sidebar.radio(
    "Select the function you would like to use as the objective function to optimize:",
    (
        'Sphere function', 'Rosenbrock function', 'Rastrigin function', 'Sum Squares function', 'Ackley function', 'Zakharov function'
    ), key='obj_function_type'
)
obj_function_type_button = st.sidebar.button('What are these?', key='function_type_button')
obj_function_type_caption = st.sidebar.empty()

OFFSET: float = st.sidebar.slider("Select the offset of zero", min_value=-1.0, max_value=1.0, value=0.5, key='offset')
offset_button = st.sidebar.button('What is this?')
offset_caption = st.sidebar.empty()

GENERATIONS: int = st.sidebar.slider("Select the number of generations", min_value=1, max_value=1000, value=60, key='generations')
POPULATION_SIZE: int = st.sidebar.slider("Select the population size", min_value=1, max_value=1000, value=400, key='population_size')
warning_label = st.sidebar.empty()

MUTATION_RATE: float = st.sidebar.slider("Select the mutation rate", min_value=0.01, max_value=0.99, value=0.4, key='mutation_rate')
mutation_button = st.sidebar.button('Tips')
mutation_caption = st.sidebar.empty()

CROSSOVER_TYPE = st.sidebar.radio(
    "Select how you would like to implement crossover:",
    (
        'Crossover by averaging', 'Crossover by combination'
    ), key='crossover_type'
)
crossover_type_button = st.sidebar.button('What is this?', key='crossover_type_button')
crossover_type_caption = st.sidebar.empty()

sep = st.sidebar.markdown('___')
run_button = st.sidebar.button('Run')
success_label = st.sidebar.empty()
progress_label = st.subheader('')
progress_bar = st.progress(0)
fitness_sublabel = st.subheader('')
fitness_chart = st.line_chart()
fitness_caption = st.empty()
f_sublabel = st.subheader('')
f_chart = st.line_chart()
f_caption = st.empty()
pop_sublabel = st.empty()
pop_graph = st.empty()
pop_caption = st.empty()
result0_label = st.empty()
result1_label = st.empty()
result2_label = st.empty()

if GENERATIONS > 100 or POPULATION_SIZE > 100:
    warning_label.warning("Generations and Population Sizes over 100 will require a longer time to run")

DEBUG_CONFIG: bool = False


########################################################################################################################
# objective function declarations
########################################################################################################################

# """# sphere
#
# *This constructs the sphere function - convex, continuous, unimodal; best for bowl-shaped distributions.*
#
# ###parameters
#
# * **shift** float: - This value indicates how far we should offset, or shift, each dimension of the input. In theory,
# no offset is needed if not desired. Each objective function has a global minimum at the zero vector without
# the offset.
#
# * **xs** float: The input vector of `DIMENSIONS` dimensionality.
#
# **returns** float: the evaluation of the input vector with the indicated offset over the optimization function..
# """

def sphere(shift, xs):
    return sum([(x - shift)**2 for x in xs])


if DEBUG_CONFIG:
    assert isinstance(sphere(0.5, [0, 0]), float), 'ERROR: invalid return type for <sphere> function.'


# """# rosenbrock
#
# *This constructs the Rosenbrock function - a non-convex optimization function sometimes called the 'banana function'
# due to its shape. Best for optimizing valley-shaped distributions*
#
# ###parameters
#
# * **shift** float: - This value indicates how far we should offset, or shift, each dimension of the input. In theory,
# no offset is needed if not desired. Each objective function has a global minimum at the zero vector without
# the offset.
#
# * **xs** float: The input vector of `DIMENSIONS` dimensionality.
#
# **returns** float: the evaluation of the input vector with the indicated offset over the optimization function..
# """

def rosenbrock(shift, individual: list):
    f_sum: float = 0.0
    x: list = individual
    for index in range(1, len(x)):
        x_0: float = ((x[index] - shift) - ((x[index - 1] - shift) ** 2)) ** 2
        x_1: float = (1 - (x[index] - shift) - 1) ** 2
        f_sum += 100 * (x_0 + x_1)
    return f_sum


if DEBUG_CONFIG:
    assert isinstance(rosenbrock(0.5, [0, 0]), float), 'ERROR: invalid return type for <rosenbrock> function.'


# """# rastirgin
#
# *This constructs the Rastrigin function - non-convex, non-linear, and multimodal; poses a risk for hillclimbing due
# to high probability of getting trapped in local optima.*
#
# ###parameters
#
# * **shift** float: - This value indicates how far we should offset, or shift, each dimension of the input. In theory,
# no offset is needed if not desired. Each objective function has a global minimum at the zero vector without
# the offset.
#
# * **xs** float: The input vector of `DIMENSIONS` dimensionality.
#
# **returns** float: the evaluation of the input vector with the indicated offset over the optimization function..
# """

def rastrigin(shift, individual: list):
    x: list = individual
    a: float = 10.0
    n: float = float(len(x))
    f_sum: float = 0.0
    for index in range(0, len(individual)):
        x_0: float = (x[index] - shift) ** 2
        x_1: float = a * np.cos(2 * np.pi * (x[index] - shift))
        f_sum += (x_0 - x_1)
    return float(a * n + f_sum)


if DEBUG_CONFIG:
    assert isinstance(rastrigin(0.5, [0, 0]), float), 'ERROR: invalid return type for <rastrigin> function.'


# """# sum_squares
#
# *This constructs the Sum Squares function - a convex, continuous, unimodal optimization function similar to sphere.
# ###parameters
#
# * **shift** float: - This value indicates how far we should offset, or shift, each dimension of the input. In theory,
# no offset is needed if not desired. Each objective function has a global minimum at the zero vector without
# the offset.
#
# * **xs** float: The input vector of `DIMENSIONS` dimensionality.
#
# **returns** float: the evaluation of the input vector with the indicated offset over the optimization function..
# """

def sum_squares(shift, individual: list):
    f_sum_0: float = 0.0
    x: list = individual
    for index in range(0, len(x)):
        x_0: float = (x[index] - shift) ** 2
        x_0 *= float(index)
        f_sum_0 += x_0
    return f_sum_0

if DEBUG_CONFIG:
    assert isinstance(sum_squares(0.5, [0, 0]), float), 'ERROR: invalid return type for <sum_squares> function.'


# """# Ackley
#
# *This constructs the Ackley function - non-convex, and multimodal; poses a risk for hillclimbing due
# to high probability of getting trapped in local optima.*
#
# ###parameters
#
# * **shift** float: - This value indicates how far we should offset, or shift, each dimension of the input. In theory,
# no offset is needed if not desired. Each objective function has a global minimum at the zero vector without
# the offset.
#
# * **xs** float: The input vector of `DIMENSIONS` dimensionality.
#
# **returns** float: the evaluation of the input vector with the indicated offset over the optimization function..
# """

def ackley(shift, individual: list):
    a: float = 20.0
    b: float = 0.2
    c: float = 6.28
    d: float = len(individual)
    f_sum_0: float = 0.0
    f_sum_1: float = 0.0
    x: list = individual
    for index in range(0, len(x)):
        f_sum_0 += ((x[index] - shift) ** 2)
    f_sum_0 /= d
    f_sum_0 **= 0.5
    f_sum_0 *= (-1 * b)
    f_sum_0 = np.e ** f_sum_0
    for index in range(0, len(x)):
        f_sum_1 += np.cos(c * (x[index] - shift))
    f_sum_1 /= d
    f_sum_1 = np.e ** f_sum_1
    final: float = -a * f_sum_0 - f_sum_1 + a + np.e
    return final

if DEBUG_CONFIG:
    assert isinstance(ackley(0.5, [0, 0]), float), 'ERROR: invalid return type for <ackley> function.'


# """# Zakharov
#
# *This constructs the Zakharov function - convex, continuous, unimodal; great for plate-shaped distributions.*
#
# ###parameters
#
# * **shift** float: - This value indicates how far we should offset, or shift, each dimension of the input. In theory,
# no offset is needed if not desired. Each objective function has a global minimum at the zero vector without
# the offset.
#
# * **xs** float: The input vector of `DIMENSIONS` dimensionality.
#
# **returns** float: the evaluation of the input vector with the indicated offset over the optimization function..
# """

def zakharov(shift, individual: list):
    f_sum_0: float = 0.0
    f_sum_1: float = 0.0
    x: list = individual
    for index in range(0, len(x)):
        x_0: float = ((x[index] - shift) - 1) ** 2
        f_sum_0 += x_0
    for index in range(1, len(x)):
        x_0: float = x[index] * float(index) * 0.5
        f_sum_1 += x_0

    f_sum_1 **= 2
    f_sum_2 = f_sum_1 ** 2
    return f_sum_0 + f_sum_1 + f_sum_2

if DEBUG_CONFIG:
    assert isinstance(zakharov(0.5, [0, 0]), float), 'ERROR: invalid return type for <zakharov> function.'


########################################################################################################################
# genetic algorithm operational function declarations
########################################################################################################################

# """# build_real_gene
#
# *This generates a gene in the form of a real value for an individual in the `real_ga` method.*
#
# ###parameters
#
# * **parameters** dict: a dictionary containing settings for the evolution of the genetic algorithm including
# population size, mutation rate, and crossover parameters.
#
# **returns** float: a value within the specified boundaries representing a "gene" or occupying one index of the
# minimization vector.
# """

def build_real_gene(parameters):
    lower_boundary: float = parameters["lower_boundary"]
    upper_boundary: float = parameters["upper_boundary"]
    gene: float = random.uniform(lower_boundary, upper_boundary)
    return gene

if DEBUG_CONFIG:
    assert build_real_gene({'lower_boundary': 0, 'upper_boundary': 10}) != None, 'ERROR: invalid parameters for real gene'
    assert isinstance(build_real_gene({'lower_boundary': 0, 'upper_boundary': 10}),
                      float), 'ERROR: invalid return type for real gene'


# """# generate_population
#
# *This generates the initial population for the genetic algorithm to begin evolution on. It initializes a vector
# (a 10-dimensional list of values representing a 10-gene chromosome) for each value of the specified input. This only
# generates the initial population, after which the function `make_next_generation` facilitates subsequent populations.*
#
# ###parameters
#
# * **parameters** dict: a dictionary containing settings for the evolution of the genetic algorithm including
# population size, mutation rate, and crossover parameters.
#
# **returns** list[float]: a list of `size` lists, each 10 values in length.
# """

def generate_population(parameters: dict):
    population = []
    size: int = parameters["population_size"]
    dimensions: int = parameters["dimensions"]

    for i in range(size):
        individual: list = []
        for j in range(0, dimensions):
            gene = build_real_gene(parameters)
            individual.append(gene)
        population.append(individual)

    return population


test_params_real = {
    "f": lambda xs: (0.5 - xs),
    "minimization": True,
    "dimensions": 10,
    "crossover_through_combo": True,
    "crossover_index": 0,
    "mutation_rate": 0.5,
    "generations": 2,
    "population_size": 2,
    "lower_boundary": 0,
    "upper_boundary": 10,
    "is_binary": False
}
global OBJ_FUNCTION

if DEBUG_CONFIG:
    assert len(generate_population(test_params_real)) > 0, 'ERROR: population size of zero returned'
    assert isinstance(generate_population(test_params_real),
                      list), 'ERROR: invalid return type for generate_population function.'


# """# apply_objective_function
#
# *This applies the objective function to each individual in the current generation (each chromosome in the current
# gene pool). In this case, the objective function is the sphere function specified in `sphere`. This is the function
# we test each individual's fitness with.*
#
# ###parameters
#
# * **individual** list[int]: a candidate minimization vector from the current generation represented by a list of floats.
#
# **returns** list[float]: a list of `size` lists, each 10 values in length.
# """

def apply_objective_function(individual):
    sum: float = 0.0
    if OBJ_FUNCTION_TYPE == 'Sphere function':
        func = sphere
        sum += func(OFFSET, individual)
    elif OBJ_FUNCTION_TYPE == 'Rosenbrock function':
        func = rosenbrock
        sum += func(OFFSET, individual)
    elif OBJ_FUNCTION_TYPE == 'Rastrigin function':
        func = rastrigin
        sum += func(OFFSET, individual)
    elif OBJ_FUNCTION_TYPE == 'Sum Squares function':
        func = sum_squares
        sum += func(OFFSET, individual)
    elif OBJ_FUNCTION_TYPE == 'Ackley function':
        func = ackley
        sum += func(OFFSET, individual)
    elif OBJ_FUNCTION_TYPE == 'Zakharov function':
        func = zakharov
        sum += func(OFFSET, individual)
    return sum

if DEBUG_CONFIG:
    assert isinstance(apply_objective_function([1.0, 2.0, 3.4, 5.0, 1.2, 3.23, 2.87, 4.23, 3.82, 4.61]),
                      float), 'ERROR: invalid format for individual - objective function cannot compute.'


# """# choice_by_roulette
#
# *This calculates fitness of selected individuals and applies the objective function to those with a fitness below a
# certain threshold. The individuals evaluated are selected "by roulette" through a random number generator.*
#
# ###parameters
#
# * **sorted_population** list[float]: the population of the current generation sorted by fitness.
#
# * **fitness** float: a value that indicates how close the minimization vector is to the actual solution.
#
# **returns** list[float]: a candidate minimization vector from the current generation represented by a list of floats.
# """

def choice_by_roulette(sorted_population, fitness):
    offset = 0
    current_fitness = fitness
    if current_fitness == 0: current_fitness += 1
    lowestFitness = apply_objective_function(sorted_population[-1])
    if lowestFitness < 0:
        offset = -lowestFitness
        current_fitness += offset * len(sorted_population)
    draw = random.uniform(0,1)
    accumulated = 0
    for individual in sorted_population:
        fitness = apply_objective_function(individual) + offset
        probability = fitness / current_fitness
        accumulated += probability
        if draw <= accumulated:
            return individual

if DEBUG_CONFIG:
    assert isinstance(choice_by_roulette([[0.5, 0.5], [3, 3]], 0.6),
                      list), "ERROR: invalid return type for choice_by_roulette"


# """# sort_population_by_fitness
#
# *This sorts the current population of minimzation vectors and sorts them in increasing order according to each vector's fitness value.*
#
# ###parameters
#
# * **population** list[float]: the population of the current generation unsorted.
#
# **returns** list[float]: the population of the current generation unsorted.
# """

def sort_population_by_fitness(population):
    return sorted(population, key=apply_objective_function, reverse=False)


# """# crossover_through_average (sub-routine)
#
# *This is a sub-routine of `crossover` that implements crossover by taking the genes at each index for `individual_a` and `individual_b` and averaging them. This average then constructs the gene at the same index for a new minimalization vector.*
#
# ###parameters
#
# * **individual_a** list[int]: a mother candidate minimization vector from the current generation represented by a list of floats.
#
# * **individual_b** list[int]: a father candidate minimization vector from the current generation represented by a list of floats.
#
# **returns** list[float]: a new candidate minimization vector generated from the two parent individuals represented by a list of floats.
# """

def crossover_through_average(individual_a: list, individual_b: list):
    try:
        crossover_individual: list = []
        for index in range(0, len(individual_a)):
            gene: float = (individual_a[index] + individual_b[index]) / 2
            crossover_individual.append(gene)
        return crossover_individual
    except TypeError:
        return [0] * len(individual_a)


if DEBUG_CONFIG:
    assert isinstance(crossover_through_average([1] * 1, [3] * 1),
                      list), "ERROR: invalid return type for crossover_through_average"
    assert crossover_through_average([1] * 1, [3] * 1)[0] == 2, "ERROR: invalid crossover method crossover_through_average"
    assert crossover_through_average([1] * 1, [3] * 1)[0] == 2, "ERROR: invalid crossover method crossover_through_average"


# """# crossover_through_combination (sub-routine)
#
# *This is another subroutine of `crossover` that implements crossover by copying the genes of `individual_a` up to the specified index and then copying the genes from `individual_b` to the ending index. The result is a new minimalization vector.*
#
# ###parameters
#
# * **individual_a** list[int]: a mother candidate minimization vector from the current generation represented by a list of floats.
#
# * **individual_b** list[int]: a father candidate minimization vector from the current generation represented by a list of floats.
#
# * **crossover_index** int: the index to stop copying the genes from `individual_a` and start copying the genes from `individual_b` if `through_combo` is selected.
#
# **returns** list[float]: a new candidate minimization vector generated from the two parent individuals represented by a list of floats.
# """

def crossover_through_combination(individual_a: list, individual_b: list, crossover_index: int):
    crossover_individual: list = []

    for index in range(0, crossover_index):
        crossover_individual.append(individual_a[index])

    for index in range(crossover_index, len(individual_b)):
        crossover_individual.append(individual_b[index])

    return crossover_individual

if DEBUG_CONFIG:
    assert isinstance(crossover_through_combination([1] * 2, [3] * 2, 0), list), "ERROR: invalid return type for crossover_through_combination"
    assert crossover_through_combination([1] * 2, [3] * 2, 0)[0] == 3, "ERROR: invalid crossover method crossover_through_combination"


# """# crossover (main routine)
#
# *This function implements the crossover principle between two genes through one of two subroutines. If crossover is not implemented, the genetic algorithm is simply making copies of old generations. It effectively takes two minimization vectors from the previous generation and computes a new vector based on the values from the two original vectors to use in the next generation. It does so by taking the first parts of `individual_a` and swap with `individual_b` to generate a new candidate. It is important to note that crossover simply reorganizes existing genes; it does not add new ones to the population. New candidates are added to the population through crossover, and the pre-crossover candidates are removed. `crossover` and `mutate` are the two functions that facilitate evolution between generations.*
#
# ###parameters
#
# * **individual_a** list[int]: a mother candidate minimization vector from the current generation represented by a list of floats.
#
# * **individual_b** list[int]: a father candidate minimization vector from the current generation represented by a list of floats.
#
# * **crossover_index** int: the index to stop copying the genes from `individual_a` and start copying the genes from `individual_b` if `through_combo` is selected.
#
# * **through_combo** bool: a flag indicating which of the two crossover methods to use. Enabling this alows crossover by copying the genes directly from `individual_a` and `individual_b`. Disabling this allows crossover to occur by averaging the values for both genes at each index.
#
# **returns** list[float]: a new candidate minimization vector generated from the two parent individuals represented by a list of floats.
# """

def crossover(individual_a: list, individual_b: list, crossover_index: int=5, through_combo: bool=True):
    if through_combo:
        return crossover_through_combination(individual_a, individual_b, crossover_index)
    else:
        return crossover_through_average(individual_a, individual_b)

if DEBUG_CONFIG:
    assert isinstance(crossover([1] * 10, [3] * 10, 1, True), list), "ERROR: invalid return type for crossover"
    assert isinstance(crossover([1] * 10, [3] * 10, 1, False), list), "ERROR: invalid return type for crossover"


# """# mutate
#
# *This function implements the mutation principle over the selected candidate. If mutation is not implemented, we are not bringing in new randomness or new candidates/individuals Mutation does not need to be implemented, but eventually all permutation of candidates with old genes from crossover will be implemented and evolution will simply repeat generations over and over again, effectively becoming a Monte Carlo simulation of sorts. Mutation is needed to create new candidates with NEW genes. Mutation brings more candidates in to population if population has been homogenized. It is very important to implement randomness for convergence. `crossover` and `mutate` are the two functions that facilitate evolution between generations.*
#
# ###parameters
#
# * **individual** list[int]: a candidate minimization vector from the current generation represented by a list of floats.
#
# * **parameters** dict: a dictionary containing settings for the evolution of the genetic algorithm including population size, mutation rate, and crossover parameters.
#
# **returns** list[float]: a new candidate minimization vector mutated from the argument individual and represented by a list of floats.
# """

def mutate(individual, parameters):
    count: int = 0
    is_binary: bool = parameters["is_binary"]
    lower_boundary: float = parameters["lower_boundary"]
    upper_boundary: float = parameters["upper_boundary"]
    mutation_rate: float = parameters["mutation_rate"]
    mutation_count: int = int(mutation_rate * len(individual))
    mutated_genes: list = []
    mutated_individual = individual
    while count != mutation_count:
        gene = random.randint(0, len(individual) - 1)
        if gene not in mutated_genes:
            mutated_genes.append(gene)
            count += 1

    for gene in mutated_genes:
        if not is_binary: mutated_individual[gene] += (random.randrange(int(lower_boundary), int(upper_boundary)) / 10)

    for index in range(0, len(mutated_genes)):
        if is_binary and len(mutated_genes) > 0:
            mutated_individual[index] = mutated_genes.pop()

    return mutated_individual

if DEBUG_CONFIG:
    assert isinstance(mutate([3] * 10, test_params_real), list), 'ERROR: Invalid return type for mutate function'
    assert ([3] * 10) != mutate([3] * 10, test_params_real), 'ERROR: No mutation occurred in mutate function'


# """# make_next_generation
#
# *This makes a new generation of minimization vectors based on the current generation. The population is replaced with the next generation of minimization vectors, not appended to (so overall population remains constant throughout evolution). The vectors (individuals) are sorted according to fitness, selected by roulette, and evolved with crossover and mutation. This function calls the `choice_by_roulette`, `crossover`, and `mutate` methods to facilitate these operations.*
#
# ###parameters
#
# * **previous_population** list[float]: the population of the current generation used to create the population of a new generation.
#
# * **parameters** dict: a dictionary containing settings for the evolution of the genetic algorithm including population size, mutation rate, and crossover parameters.
#
# * **debug** bool: indicates whether or not to print the best and worst performing individuals' fitness for the previous generation.
#
# **returns** list[float]: a new population of minimization vectors based on the previous generation.
# """

def make_next_generation(previous_population: list, parameters: dict, debug: bool=False):
    next_generation = []
    sorted_population = sort_population_by_fitness(previous_population)
    population_size = len(previous_population)
    fitness = sum(apply_objective_function(individual) for individual in sorted_population)
    fitness = 1 / (1 + fitness)
    best_individual: list = sorted_population[0]
    best_f: float = apply_objective_function(best_individual)
    best_fitness: float = 1 / (1 + best_f)
    worst_individual: list = sorted_population[-1]
    worst_f: float = apply_objective_function(worst_individual)
    worst_fitness: float = 1 / (1 + worst_f)
    if debug:
        print(f'\tbest individual: {best_individual}\n\t\tfitness = {best_fitness}\n\t\tfunction applied: {best_f}')
        print(f'\tworst individual: {worst_individual}\n\t\tfitness = {worst_fitness}\n\t\tfunction applied: {worst_f}')
    for i in range(population_size):
        mother_chromosome = choice_by_roulette(sorted_population, fitness)
        father_chromosome = choice_by_roulette(sorted_population, fitness)
        individual = crossover(mother_chromosome, father_chromosome, parameters['crossover_index'], parameters['crossover_through_combo'])
        individual = mutate(individual, parameters)
        next_generation.append(individual)
    return next_generation, best_individual, best_fitness, best_f


if DEBUG_CONFIG:
    assert ([[3] * 10] * 10) != make_next_generation([[3] * 10] * 10, test_params_real, False), 'ERROR: Next generation is same as previous generation in make_next_generation_function'


# """# real_ga
#
# *This is the driver function for the real-valued genetic algorithm. It sets the population, initializes the first generation, and coordinates bookkeeping of the best-performing individuals to deliver as the the final result of the algorithm's evolution.*
#
# ###parameters
#
# * **parameters** dict: a dictionary containing settings for the evolution of the genetic algorithm including population size, mutation rate, and crossover parameters.
#
# * **debug** bool: indicates whether or not to print the best and worst performing individuals' fitness for the previous generation.
#
# **returns** dict: a dictionary of values containing the minimization vector, fitness value, and optimization function value as specified by the assignment.
# """

def real_ga(parameters, debug=False):
    population: list = generate_population(parameters)
    generations: list = parameters["generations"]
    best_individual: list = []
    best_fitness: float = 0.0
    best_f: float = 0.0
    i = 1
    while True:
        if i == generations + 1: break
        if debug: print(f'\n\ngeneration {i}')
        i += 1
        population, individual, fitness, f = make_next_generation(population, parameters, debug)
        if fitness > best_fitness: best_individual, best_fitness, best_f = individual, fitness, f
        best_fitness = min(best_fitness, 1)
        # best_individual, best_fitness, best_f = individual, fitness, f
        update_interface(i - 1, best_fitness, best_f, best_individual)
    sorted_population: list = sort_population_by_fitness(population)
    best_individual = sorted_population[0]
    worst_individual = sorted_population[-1]
    if debug:
        print(f'\tbest individual: {best_individual}\n\t\tfunction applied: {apply_objective_function(best_individual)}')
        print(f'\tworst individual: {worst_individual}\n\t\tfunction applied: {apply_objective_function(worst_individual)}')
        print('\n\n\nfinal result - one weight for every dimension:\n__________________________________________')
    return {"solution": best_individual, "fitness": best_fitness, "f": best_f}


# if DEBUG_CONFIG:
#     assert isinstance(real_ga(test_params_real), dict), 'ERROR: invalid return type for real_ga function.'
#     assert 'solution' in real_ga(test_params_real).keys()
#     assert 'fitness' in real_ga(test_params_real).keys()
#     assert 'f' in real_ga(test_params_real).keys()


########################################################################################################################
# interface and main executional function declarations
########################################################################################################################

fitness_data: list = []
f_data: list = []
best_data: list = []

# """# set_parameters
#
# *This initializes and sets the hyperparameters for the genetic algorithm. It converts arguments provided by the UI
# commands interpretable by the GA functions.*
#
# ###parameters
#
# * None
#
# **returns** dict: a dictionary of hyperparameters that govern how the GA will execute.
# """

def set_parameters():
    if CROSSOVER_TYPE == 'Crossover by averaging': CROSSOVER = 0
    else: CROSSOVER = 1
    if OBJ_FUNCTION_TYPE == 'Sphere function':
        func = sphere
        OBJ_FUNCTION = func
        parameters = {
           "f": lambda xs: func(OFFSET, xs),
           "minimization": True,
           "dimensions": DIMENSIONS,
           "crossover_through_combo": CROSSOVER,
           "crossover_index": int(DIMENSIONS / 2),
           "mutation_rate": MUTATION_RATE,
           "generations": GENERATIONS,
           "population_size": POPULATION_SIZE,
           "lower_boundary": -36929 * (DIMENSIONS ** -2.644),
           "upper_boundary": 36929 * (DIMENSIONS ** -2.644),
           "is_binary": False
        }
    elif OBJ_FUNCTION_TYPE == 'Rosenbrock function':
        func = rosenbrock
        OBJ_FUNCTION = func
        parameters = {
           "f": lambda xs: func(OFFSET, xs),
           "minimization": True,
           "dimensions": DIMENSIONS,
           "crossover_through_combo": CROSSOVER,
           "crossover_index": int(DIMENSIONS / 2),
           "mutation_rate": MUTATION_RATE,
           "generations": GENERATIONS,
           "population_size": POPULATION_SIZE,
           "lower_boundary": -36929 * (DIMENSIONS ** -2.644),
           "upper_boundary": 36929 * (DIMENSIONS ** -2.644),
           "is_binary": False
        }
    elif OBJ_FUNCTION_TYPE == 'Rastrigin function':
        func = rastrigin
        OBJ_FUNCTION = func
        parameters = {
           "f": lambda xs: func(OFFSET, xs),
           "minimization": True,
           "dimensions": DIMENSIONS,
           "crossover_through_combo": CROSSOVER,
           "crossover_index": int(DIMENSIONS / 2),
           "mutation_rate": MUTATION_RATE,
           "generations": GENERATIONS,
           "population_size": POPULATION_SIZE,
           "lower_boundary": -5.12,
           "upper_boundary": 5.12,
           "is_binary": False
        }
    elif OBJ_FUNCTION_TYPE == 'Sum Squares function':
        func = sum_squares
        OBJ_FUNCTION = func
        parameters = {
           "f": lambda xs: func(OFFSET, xs),
           "minimization": True,
           "dimensions": DIMENSIONS,
           "crossover_through_combo": CROSSOVER,
           "crossover_index": int(DIMENSIONS / 2),
           "mutation_rate": MUTATION_RATE,
           "generations": GENERATIONS,
           "population_size": POPULATION_SIZE,
           "lower_boundary": -10,
           "upper_boundary": 10,
           "is_binary": False
        }
    elif OBJ_FUNCTION_TYPE == 'Ackley function':
        func = ackley
        OBJ_FUNCTION = func
        parameters = {
           "f": lambda xs: func(OFFSET, xs),
           "minimization": True,
           "dimensions": DIMENSIONS,
           "crossover_through_combo": CROSSOVER,
           "crossover_index": int(DIMENSIONS / 2),
           "mutation_rate": MUTATION_RATE,
           "generations": GENERATIONS,
           "population_size": POPULATION_SIZE,
           "lower_boundary": -32.768,
           "upper_boundary": 32.768,
           "is_binary": False
        }
    elif OBJ_FUNCTION_TYPE == 'Zakharov function':
        func = zakharov
        OBJ_FUNCTION = func
        parameters = {
           "f": lambda xs: func(OFFSET, xs),
           "minimization": True,
           "dimensions": DIMENSIONS,
           "crossover_through_combo": CROSSOVER,
           "crossover_index": int(DIMENSIONS / 2),
           "mutation_rate": MUTATION_RATE,
           "generations": GENERATIONS,
           "population_size": POPULATION_SIZE,
           "lower_boundary": -36929 * (DIMENSIONS ** -2.644),
           "upper_boundary": 36929 * (DIMENSIONS ** -2.644),
           "is_binary": False
        }
    else:
        func = sphere
        OBJ_FUNCTION = func
        parameters = {
           "f": lambda xs: func(OFFSET, xs),
           "minimization": True,
           "dimensions": DIMENSIONS,
           "crossover_through_combo": CROSSOVER,
           "crossover_index": int(DIMENSIONS / 2),
           "mutation_rate": MUTATION_RATE,
           "generations": GENERATIONS,
           "population_size": POPULATION_SIZE,
           "lower_boundary": -36929 * (DIMENSIONS ** -2.644),
           "upper_boundary": 36929 * (DIMENSIONS ** -2.644),
           "is_binary": False
        }
    return parameters



# """# update_interface
#
# *This handles interface updates. With each generation that is computed, the interface must update its state.*
#
# ###parameters
#
# * **generations** int: the number of generations completed.
#
# * **fitness** float: the fitness of the best individual from the previous generation.
#
# * **f** float: the f-value of the best individual from the previous generation.
#
# * **best_individual** list: the best individual from the previous generation.
#
#
# **returns** None
# """

def update_interface(generations: int, fitness: float, f: float, best_individual: list):
    f_data.append(f)
    fitness_data.append(fitness)
    best_data.append(best_individual)
    progress_data = {'f': f_data, 'fitness': fitness_data}
    dataframe = pd.DataFrame(progress_data, columns=['f', 'fitness'])
    fitness_chart.line_chart(dataframe['fitness'])
    fitness_sublabel.subheader('fitness for each generation')
    fitness_caption.caption('Ideally, we want this monotonically increasing and as close to 1.0 as possible, which '
                            'represents 100% fitness to the true solution.')
    f_chart.line_chart(dataframe['f'])
    f_sublabel.subheader('f for each generation')
    f_caption.caption('Ideally, we want this monotonically decreasing toward zero, which means we are minimizing the'
                      ' error between the true solution and the solution the GA is evolving.')

    perc_complete: float = min(float(generations / GENERATIONS), 1.0)
    progress_bar.progress(perc_complete)
    progress_label.subheader(f'{int(perc_complete * 100)} % Complete')
    graph_data(best_data, best_individual)


# """# graph_data
#
# *This constructs and updates three graphs - the fitness graph, the f-value graph, and the solution evolution graph.*
#
# ###parameters
#
# * **best_data** list: a bank of the best performing solutions from all previous generations.
#
# * **best_individual** list: the best individual from the previous generation.
#
#
# **returns** None
# """

def graph_data(best_data: list, best_individual: list):

    if DIMENSIONS == 2:
        pop_sublabel.subheader('Evolution of best performing solutions per generation')
        dataframe = pd.DataFrame(best_data, columns=['x0', 'x1'])
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(dataframe.iloc[:,0], dataframe.iloc[:,1], 'r.--')
        ax.plot(best_individual[0], best_individual[1], '-gD')
        ax.scatter(dataframe.iloc[:,0], dataframe.iloc[:,1], color='red')
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        pop_graph.pyplot(fig)
        pop_caption.caption('Each of the red points is the fittest solution from the population of a previous '
                            'generation. The dotted line constructs a path between solutions from the first and final '
                            'generations. The green square is the fittest solution for the current and final generation.'
                            ' Ideally, the optimal algorithm will show successively smaller distances between solutions '
                            ' as the GA converges on the globally fittest solution.')
        del dataframe
        del fig
    elif DIMENSIONS == 3:
        pop_sublabel.subheader('Evolution of best performing solutions per generation')
        dataframe = pd.DataFrame(best_data, columns=['x0', 'x1', 'x2'])
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(dataframe.iloc[:,0], dataframe.iloc[:,1], dataframe.iloc[:,2], 'r.--')
        ax.plot(best_individual[0], best_individual[1], best_individual[2], '-gD')
        ax.scatter(dataframe.iloc[:,0], dataframe.iloc[:,1], dataframe.iloc[:,2], color='red')
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')
        pop_graph.pyplot(fig)
        pop_caption.caption('Each of the red points is the fittest solution from the population of a previous '
                            'generation. The dotted line constructs a path between solutions from the first and final '
                            'generations. The green square is the fittest solution for the current and final generation.'
                            ' Ideally, the optimal algorithm will show successively smaller distances between solutions '
                            ' as the GA converges on the globally fittest solution.')
        del dataframe
        del fig


########################################################################################################################
# interface logic
########################################################################################################################


if obj_function_type_button:
    obj_function_type_caption.caption("I tried to select optimization algorithms that had a global minimum at the zero "
                                      " vector. I also tried to select one optimization function from each data shape - "
                                      " valley, bowl, plate, multimodal, and unimodal distribution shapes.\n\n"
                                      "Sphere function: convex, continuous, unimodal; best for bowl-shaped distributions.\n\n"
                                      "Rosenbrock function: a non-convex optimization function sometimes called the "
                                      "'banana function' due to its shape. Best for optimizing valley-shaped distributions.\n\n"
                                      "Rastrigin function: non-convex, non-linear, and multimodal; poses a risk for "
                                      "hillclimbing due to high probability of getting trapped in local optima.\n\n"
                                      "Sum Squares function: a convex, continuous, unimodal optimization function similar"
                                      " to sphere.\n\n"
                                      "Ackley function: non-convex, multimodal; poses a risk for hillclimbing due to high"
                                      "probability of getting trapped in local optima.\n\n"
                                      "Zakharov function: convex, continuous, unimodal; great for plate-shaped distributions.")
else:
    obj_function_type_caption.empty()

if offset_button:
    offset_caption.caption('This value indicates how far we should offset, or shift, each dimension of the input.'
                           ' In theory, no offset is needed if not desired. Each objective function has a global minimum'
                           ' at the zero vector without the offset.')
else:
    offset_caption.empty()

if mutation_button:
    mutation_caption.caption('Values of 0.5-0.7 work well for lower dimensions such as d=2 or d=3. Values of 0.2 work '
                             'best for higher dimensions such as d=10 or greater.')
else:
    mutation_caption.empty()

if crossover_type_button:
    crossover_type_caption.caption("Crossover by averaging works by averaging mother and father genes of the same index"
                                   " together to construct the child genes of a new chromosome.\n\nCrossover by combination directly"
                                   " copies the first half of the father chromosome and the first half of the mother chromosome "
                                   " into a new child chromosome.")
else:
    crossover_type_caption.empty()


if run_button:
    warning_label.empty()
    success_label.success('Running the genetic algorithm now...')

    parameters = set_parameters()
    result = real_ga(parameters, False)
    success_label.success('Run complete!')
    solution_final: list = result['solution']
    solution_final = [np.round(x, 4) for x in solution_final]
    solution_true: list = [OFFSET] * DIMENSIONS
    f_final = result['f']
    fitness_final = 100.0 * result['fitness']

    result1_label, result2_label = st.columns(2)
    result0_label.success(f'\nFittest solution: {solution_final}\n\nActual solution: {solution_true}')
    result1_label.metric(f'fitness', f'{np.round(fitness_final, 2)} %')
    result2_label.metric(f'f-value', f'{np.round(f_final, 4)}', delta=None)


