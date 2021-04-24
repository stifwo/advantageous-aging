import math
import numpy as np
from typing import Tuple
from scipy.optimize import brentq

def get_fecundity(population_survivorship, fertility):
    # fertility: tuple (time of deposition, Number of female offspring in clutch)
    # population_survivorship er én av variantene for population_simulations[HYP_WILD_TYPE], altså dim (1000, 100)
    
    number_of_repetitions = population_survivorship.shape[0] 
    number_of_individuals = population_survivorship[0, 0] # assuming all individuals alive at beginning
    t_m = population_survivorship.shape[1]
    
    # m(t): Average number of female offspring produced by an individual at age/time t
    m_t = np.zeros((number_of_repetitions, t_m)) 
    for f in fertility:
        m_t[:, f[0]] = f[1]
    
    # l(t): Probability of being alive at age/time t
    l_t = population_survivorship * (1 / number_of_individuals)
    
    fecundity = np.multiply(l_t, m_t)
    
    return fecundity
    
def euler_lotka(r, f, t):
    return np.sum(f * np.exp(-r * t)) - 1

def get_mean_and_sem(arr: np.ndarray) -> Tuple[float, float]:
    # NOT normalized 
    number_of_repetitions = arr.shape[0]
    mean = np.mean(arr)
    sem = np.std(arr) / math.sqrt(number_of_repetitions)
    return mean, sem


def get_fitness_data(population_simulation, number_of_repetitions, t_m, fertility):
    # population_simulation er population_simulations[population]
    # fertility (TODO: rename?) er liste av tuple med (day, no. offspring)
    
    t_arr = np.arange(0, t_m)

    mean_r0_arr = []
    sem_r0_arr = []
    mean_r_arr = []
    sem_r_arr = []

    number_of_values = len(population_simulation)
    for i in range(number_of_values): 
        fecundity = get_fecundity(np.array(population_simulation[i]), fertility)
        r0 = np.sum(fecundity, axis=1)

        a, b = -2, 8  # TODO: hvor kommer disse fra

        r_arr = []
        for i in range(number_of_repetitions):
            f = fecundity[i, :]
            r = brentq(euler_lotka, a, b, args=(f, t_arr))
            r_arr.append(r)
    
        mean_r0, sem_r0 = get_mean_and_sem(r0)
        mean_r, sem_r = get_mean_and_sem(np.array(r_arr))

        mean_r0_arr.append(mean_r0)
        sem_r0_arr.append(sem_r0)
        mean_r_arr.append(mean_r)
        sem_r_arr.append(sem_r)

    return dict(mean_r0=mean_r0_arr, sem_r0=sem_r0_arr, mean_r=mean_r_arr, sem_r=sem_r_arr)
