# This python script is used for solving the second design problem of the MTE 119 course:
# the script builds a genetic algorithm that simulates natural selection to arrive at the best solution for a robot arm
# with minial torque about the origin
import math
import np as np
import numpy as np
import matplotlib.pyplot as plt
import copy


def tor(n, d):
    """ we define tor to be a function that is capable of calculating the torque about the origin for any given arm
        that takes in n which contains the information about the current robot arm and d to be the position being checked
        n as the lengths of links 1 through three in index 0-2 and the set of angles for each position contained
        within indexes 3-8"""
    # check which scenario we are evaluating
    if d == 1:
        c = 0
        a = n[3]
        b = n[4]
        x = 0.5
        y = 0.5
    elif d == 2:
        c = math.radians(45)
        a = n[5]
        b = n[6]
        x = 0.2
        y = 0.6
    else:
        c = math.radians(-60)
        a = n[7]
        b = n[8]
        x = 0.75
        y = 0.1
    # We check to ensure the lengths and angles are within the error for the current position
    if not checkvalid(n) or not xreq(n, x, d) or not yreq(n, y, d):
        return 10 ** 100  # assign large cost to signify bad result
    # apply general torque equation:
    return -(2 * math.cos(a) * n[0] ** 2 + 2 * math.cos(a) *
             n[0] * n[1] + math.cos(b) * n[1] ** 2
             + math.cos(a) * n[0] * n[2] + math.cos(b)
             * n[1] * n[2] + 0.5 * math.cos(c) * n[2] ** 2
             + 5 * (math.cos(a) * n[0] + math.cos(b) *
                    n[1] + math.cos(c) * n[2]))


def torque(n):
    """function used to return the parameter "T" """
    return math.sqrt(pow(tor(n, 1), 2) + pow(tor(n, 2), 2) + pow(tor(n, 3), 2))


def checkvalid(n):
    """ensures that the sum of the lengths of the links is greater than or equal to one to ensure that the arm has a 1m
    reach minimum"""
    return n[0] + n[1] + n[2] >= 1


def xreq(n, x, d):
    """verifies that the current robot can reach the required x position"""
    # identify scenario
    if d == 1:
        c = 0
        a = n[3]
        b = n[4]
    elif d == 2:
        c = math.radians(45)
        a = n[5]
        b = n[6]
    else:
        c = math.radians(-60)
        a = n[7]
        b = n[8]
    # calculate sum of x components
    x_cur = n[0] * math.cos(a) + n[1] * math.cos(b) + n[2] * math.cos(c)
    return abs(x_cur - x) <= 0.01  # evaluate with 0.01m tolerance (exact value will likely be impossible)


def yreq(n, y, d):
    """verifies that the current robot can reach the required y position"""
    # identify scenario
    if d == 1:
        c = 0
        a = n[3]
        b = n[4]
    elif d == 2:
        c = math.radians(45)
        a = n[5]
        b = n[6]
    else:
        c = math.radians(-60)
        a = n[7]
        b = n[8]
    # calculate sum of y components
    y_cur = n[0] * math.sin(a) + n[1] * math.sin(b) + n[2] * math.sin(c)
    return abs(y_cur - y) <= 0.01  # evaluate with 0.01m tolerance (exact value will likely be impossible)


def roulette_wheel_select(p):
    """Defines the parent selection method"""
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]


def crossover(p1, p2):
    """mimic genetic inheritance from two parents with random genome crossover"""
    c1 = copy.deepcopy(p1)
    c2 = copy.deepcopy(p2)

    alpha = np.random.uniform(0, 1, *(c1['position'].shape))  # represents which alleles to keep/switch
    c1['position'] = alpha * p1['position'] + (1 - alpha) * p2['position']
    c2['position'] = alpha * p2['position'] + (1 - alpha) * p1['position']

    return c1, c2


def mutate(c, mu, sigma):
    """simulate genetic mutation between generations"""
    y = copy.deepcopy(c)
    flag = np.random.rand(*(c['position'].shape)) <= mu
    ind = np.argwhere(flag)
    y['position'][ind] += sigma * np.random.randn(*ind.shape)

    return y


def bounds(c, lenmin, lenmax, angmin, angmax):
    """ensures that no value possible robot arm has invalid lengths or angles"""
    c['position'][0:3] = np.maximum(c['position'][0:3], lenmin)
    c['position'][0:3] = np.minimum(c['position'][0:3], lenmax)
    c['position'][3:9] = np.maximum(c['position'][3:9], angmin)
    c['position'][3:9] = np.minimum(c['position'][3:9], angmax)


def sort(arr):
    """sort the population based on lowest cost to increase chance of selecting ideal parent"""
    n = len(arr)

    for i in range(n - 1):

        for j in range(0, n - i - 1):
            if arr[j]['cost'] > arr[j + 1]['cost']:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr


def ga(costfunc, lenmin, lenmax, angmin, angmax, maxit, npop, num_children, mu, sigma, beta):
    """genetic algorithm definition"""
    # placeholder for every individual
    population = {}
    # each individual gets populated with a valid initial solution:
    for i in range(npop):
        population[i] = {'position': [0.3585, 0.3585, 0.283, 1.86859, 0.4459161, 2.54976, 0.59183,
                                      0.73718, 0.294315], 'cost': None}

    bestsol = copy.deepcopy(population)
    bestsol_cost = np.inf

    for i in range(npop):
        # add slight variance to entire population to allow for recombination
        # vary lengths by up to 0.05mm and angle up to 0.5 deg
        variance = np.append(np.random.uniform(-0.00005, 0.00005, 3), np.random.uniform(-0.01, 0.01, 6))
        population[i]['position'] += variance
        population[i]['cost'] = costfunc(population[i]['position'])

        if population[i]['cost'] < bestsol_cost: # set initial lowest cost from starting population
            bestsol = copy.deepcopy(population[i])

    bestcost = np.empty(maxit)

    # main loop:
    for it in range(maxit):
        costs = []
        for i in range(len(population)):
            costs.append(population[i]['cost'])
        costs = np.array(costs)
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs / avg_cost
        probs = np.exp(-beta * costs) # calculate probabilities
        # begin creating children
        for _ in range(num_children // 2):

            p1 = population[roulette_wheel_select(probs)]
            p2 = population[roulette_wheel_select(probs)]

            c1, c2 = crossover(p1, p2)

            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            bounds(c1, lenmin, lenmax, angmin, angmax)
            bounds(c2, lenmin, lenmax, angmin, angmax)

            c1['cost'] = costfunc(c1['position'])
            # update lowest cost to account for new children
            if type(bestsol_cost) == float:
                if c1['cost'] < bestsol_cost:
                    bestsol_cost = copy.deepcopy(c1)
            else:
                if c1['cost'] < bestsol_cost['cost']:
                    bestsol_cost = copy.deepcopy(c1)

            if c2['cost'] < bestsol_cost['cost']:
                bestsol_cost = copy.deepcopy(c2)
        # add children to population:
        population[len(population)] = c1
        population[len(population)] = c2
        # sort population
        population = sort(population)

        # Store best cost
        bestcost[it] = bestsol_cost['cost']

        # Show generation information
        print('Iteration {}: Best Cost = {}'.format(it, bestcost[it]))

    out = population
    Bestsol = bestsol
    bestcost = bestcost
    return (out, Bestsol, bestcost, bestsol_cost)


# define starting parameters: (feel free to modify)
costfunc = torque
lenmin = 0.1
lenmax = 0.6
angmin = 0
angmax = math.pi

maxit = 1001
npop = 35
beta = 1
prop_children = 1
num_children = int(np.round(prop_children * npop / 2) * 2)
mu = 0.2
sigma = 0.1
# store output of GA
out = ga(costfunc, lenmin, lenmax, angmin, angmax, maxit, npop, num_children, mu, sigma, beta)
# output best solution:
print(out[3])
# output plot of T as a function of generation:
plt.plot(out[2])
plt.xlim(0, maxit)
plt.xlabel('Generations')
plt.ylabel('T [Nm]')
plt.title('Lowest T as a function of Generation')
plt.grid(True)
plt.show()

# the lowest cost I've ever gotten with these settings was 5.09...
