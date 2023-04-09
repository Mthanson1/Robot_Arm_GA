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
    elif d == 2:
        c = math.radians(45)
    else:
        c = math.radians(-60)

    # We check to ensure the lengths and angles are within the error for the current position
    if not checkvalid(n):
        return 10 ** 100  # assign large cost to signify bad result
    a, b = calcAngle(n, c)
    # apply general torque equation:
    g = 9.8
    return -g * (2 * math.cos(a) * n[0] ** 2 + 2 * math.cos(a) *
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
    l = n[0] + n[1]
    return l + n[2] >= 1 and l >= math.sqrt(0.5**2 + (0.5-n[2])**2) \
        and l >= math.sqrt((0.6-n[2]*math.sin(math.radians(45)))**2 + (0.2-n[2]*math.cos(math.radians(45)))**2) \
        and l >= math.sqrt((0.75-n[2]*math.cos(math.radians(-60)))**2 + (0.1+n[2]*math.sin(math.radians(-60)))**2)


def calcAngle(n, c):
    if c == 0:
        x1 = 0.5
        y1 = 0.5
    elif c == math.radians(45):
        x1 = 0.2
        y1 = 0.6
    elif c == math.radians(-60):
        x1 = 0.75
        y1 = 0.1
    x2 = x1 - n[2]*math.cos(c)
    y2 = y1 - n[2]*math.sin(c)
    ref = math.atan(y2 / x2)
    l_ref = math.sqrt(x2**2 + y2**2)
    if n[0] + n[1] == l_ref:
        return l_ref, l_ref
    a_cos = (-n[1]**2 + n[0]**2 + l_ref**2)/(2*l_ref*n[0]) % 1
    a_in = math.acos(a_cos)
    a = ref + a_in
    x3 = n[0]*math.cos(a)
    y3 = n[0]*math.sin(a)
    b = math.atan((y2-y3)/(x2-x3))
    return a, b


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
        population[i] = {'position': [0.51929811, 0.16306541, 0.333317], 'cost': None}
        # population[i] = {'position': [0.367413007842, 0.332208963082, 0.300378029076, 1.84284006553839, 0.4459161,
        #                               2.512234635, 0.553972645,
        #                               0.563326645, 0.513471569], 'cost': None}

    bestsol = copy.deepcopy(population)
    bestsol_cost = np.inf

    for i in range(npop):
        # add slight variance to entire population to allow for recombination
        # vary lengths by up to 0.5mm and angle up to 0.5 deg
        variance = np.random.uniform(-0.0005, 0.0005, 3)
        if i == 0:
            population[i]['position'] += np.zeros(3)
        else:
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
mu = 0.3
sigma = 0.1
# store output of GA
out = ga(costfunc, lenmin, lenmax, angmin, angmax, maxit, npop, num_children, mu, sigma, beta)
# output best solution:
print(out[3])
print(calcAngle(out[3]['position'], 0))
print(calcAngle(out[3]['position'], math.radians(45)))
print(calcAngle(out[3]['position'], math.radians(-60)))
# output plot of T as a function of generation:
plt.plot(out[2])
plt.xlim(0, maxit)
plt.xlabel('Generations')
plt.ylabel('T [Nm]')
plt.title('Lowest T as a function of Generation')
plt.grid(True)
plt.show()
# the lowest cost I've ever gotten with these settings was 5.09...
