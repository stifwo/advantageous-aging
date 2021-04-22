import math
import numpy as np
from typing import Tuple
from scipy.optimize import brentq

def get_fecundity(population_survivorship, fertility):
    # fertility: (day of deposition, Number of female offspring in clutch)
    # population_survivorship er én av variantene for population_simulations[HYP_WILD_TYPE], altså dim (1000, 100)
    
    number_of_repetitions = population_survivorship.shape[0] 
    number_of_individuals = population_survivorship[0, 0] # assuming all individuals alive at beginning
    t_m = population_survivorship.shape[1]
    
    # TODO: Flytt magiske tall til parametere
    # m(t): Average number of female offspring produced by an individual at age t
    m_t = np.zeros((number_of_repetitions, t_m)) 
    
    for f in fertility:
        m_t[:, f[0]] = f[1]
    #m_t[:, 39] = 24.0 # Number of eggs in clutch at first deposition TODO: parameterize
    #m_t[:, 79] = 24.0 # Number of eggs in clutch at second deposition TODO: parameterize
    
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

def get_fig_3_data(populations: Tuple[str, str], population_simulations, number_of_repetitions, t, fertilities):
    # TODO: extract number_of_repetitions from pop_sim
    # Assumes 2 populations compared
    
    #number_of_values = population_simulations[population.shape[0] # Number of epsilon values explored
    
    fig_3_data = {populations[0]: {"R0": {"mean": [], "sem": []}, "r": {"mean": [], "sem": []}}, populations[1]: {"R0": {"mean": [], "sem": []}, "r": {"mean": [], "sem": []}}}
    
    for population in populations:
        number_of_values = len(population_simulations[population])
        # TODO: refactor sånn at det ikke er en 4x loop her
        for i in range(number_of_values): 
            fecundity = get_fecundity(np.array(population_simulations[population][i]), fertilities[population])
            r0 = np.sum(fecundity, axis=1)

            a, b = -2, 8  # TODO: hvor kommer disse fra

            r_arr = []
            for i in range(number_of_repetitions):
                f = fecundity[i, :]
                r = brentq(euler_lotka, a, b, args=(f, t))
                r_arr.append(r)


            mean_r0, sem_r0 = get_mean_and_sem(r0)
            mean_r, sem_r = get_mean_and_sem(np.array(r_arr))

            fig_3_data[population]["R0"]["mean"].append(mean_r0)
            fig_3_data[population]["R0"]["sem"].append(sem_r0)
            fig_3_data[population]["r"]["mean"].append(mean_r)
            fig_3_data[population]["r"]["sem"].append(sem_r)

            
    return fig_3_data
    