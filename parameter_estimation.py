import numpy as np

from cohort_model import run_simulation, MUTANT_CAP

def mean_survivorship(t, alpha, kappa, t_m, number_of_individuals, number_of_repetitions):
    """Used for curve fitting.
    
    Returns the mean survivorship values indexed on t.
    """
    hazard_rate_parameters = dict(alpha=alpha, kappa=kappa)
    cohort = np.ones(number_of_individuals)
    population_survivorship = run_simulation(number_of_repetitions, cohort, hazard_rate_parameters, t_m, population=MUTANT_CAP)
    mean = np.mean(population_survivorship, axis=0)
    
    t = t.astype('int64') # In order to use for indexing
    return mean[t]