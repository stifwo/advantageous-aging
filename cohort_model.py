import numpy as np
import matplotlib.pyplot as plt

MUTANT_CAP = "mutant captivity"
MUTANT_WILD = "mutant wild"
HYP_WILD_TYPE = "hypothetical wild type"

def darwinian_lottery(cohort, hazard_rate_parameters):
    number_of_individuals = cohort.shape[0] 
    hazard = hazard_rate(**hazard_rate_parameters)
    lottery_tickets = cohort * np.random.random_sample(number_of_individuals)
    survivors = np.piecewise(lottery_tickets, [lottery_tickets <= hazard, lottery_tickets > hazard], [0, 1])
    return survivors
    

def cohort_survivorship_model(cohort_survivorship, hazard_rate_parameters):
    
    # On first run, make 1D array into 2D
    if len(cohort_survivorship.shape) == 1:
        cohort_survivorship = cohort_survivorship.reshape(1, -1)
    
    time = cohort_survivorship.shape[0] # Time variable t
    hazard_rate_parameters["time"] = time
    
    current_cohort = cohort_survivorship[-1]
    number_of_survivors = sum(current_cohort)
    
    if number_of_survivors <= 1: 
        return cohort_survivorship
    survivors = darwinian_lottery(current_cohort, hazard_rate_parameters)
    cohort_survivorship = np.append(cohort_survivorship, survivors.reshape(1, -1), axis=0)        
    return cohort_survivorship_model(cohort_survivorship, hazard_rate_parameters)

def hazard_rate(population, epsilon, hazard_rate_wild_type, alpha, kappa, time):    
    # TODO: Rename funksjon til get_hazard_rate
    # TODO: Rename time til t
    if population == "mutant wild":
        return (1 - epsilon) * hazard_rate_wild_type + alpha * (((1 + kappa)**(time + 1)) - 1)
    elif population == "mutant captivity":
        return alpha * (((1 + kappa)**(time + 1)) - 1)
    elif population == "hypothetical wild type":
        return hazard_rate_wild_type
    else:
        raise ValueError("Population argument must be set to either 'mutant wild', 'mutant captivity' or 'hypothetical wild type'.")

def run_simulation(number_of_repetitions, cohort_survivorship, hazard_rate_parameters, t_max):
    # TODO: Rename t_max til t_m
    results = []
    for _ in range(number_of_repetitions):
        n_survivors_per_timestep = np.sum(cohort_survivorship_model(cohort_survivorship, hazard_rate_parameters), axis=1, dtype=int)
        results.append(n_survivors_per_timestep)
    
    for repetition in results:
        if len(repetition) < t_max:
            raise Exception("Hva gjør vi når dette skjer?") # TODO
    
    return np.array([repetition[:t_max] for repetition in results])


def get_mean_and_std(survivors_per_timestep, number_of_individuals):
    # Normalized
    mean = np.mean(survivors_per_timestep, axis=0) * (1 / number_of_individuals) 
    std = np.std(survivors_per_timestep * (1 / number_of_individuals), axis=0)
    return mean, std


