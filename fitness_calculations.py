import math
import numpy as np
from typing import Tuple

def get_fecundity(population_survivorship):
    # population_survivorship er én av variantene for population_simulations[HYP_WILD_TYPE], altså dim (1000, 100)
    
    number_of_repetitions = population_survivorship.shape[0] 
    number_of_individuals = population_survivorship[0, 0] # assuming all individuals alive at beginning
    t_m = population_survivorship.shape[1]
    
    # m(t): Average number of female offspring produced by an individual at age t
    m_t = np.zeros((number_of_repetitions, t_m)) 
    m_t[:, 39] = 24.0 # Number of eggs in clutch at first deposition TODO: parameterize
    m_t[:, 79] = 24.0 # Number of eggs in clutch at second deposition TODO: parameterize
    
    # l(t): Probability of being alive at age t
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

def get_fig_3_data()
    