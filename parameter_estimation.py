"""This module does blah blah.""" # TODO (Slett denne modulen?)

import numpy as np

from cohort_model import run_cohort_simulation, MUTANT_CAP

def mean_survivorship(t, alpha, kappa, t_m, individual_count, repetition_count):
    """Return the mean survivorship values indexed on t."""
    hazard_rate_parameters = dict(alpha=alpha, kappa=kappa, population=MUTANT_CAP)
    population_survivorship = run_cohort_simulation(repetition_count, individual_count, hazard_rate_parameters, t_m)
    survivorship_mean = np.mean(population_survivorship, axis=0)
    
    t = t.astype('int64') # In order to use for indexing
    return survivorship_mean[t]