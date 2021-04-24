from collections import defaultdict
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

MUTANT_CAP = "mutant captivity"
MUTANT_WILD = "mutant wild"
HYP_WILD_TYPE = "hypothetical wild type"

# TODO: https://stackoverflow.com/questions/48721582/how-to-choose-proper-variable-names-for-long-names-in-python
# TODO: Add typing
# TODO: Add docstrings
# TODO: Add checks for incorrect parameter sets


def get_hazard_rate(
    population: str,
    epsilon: float,
    hazard_rate_wild_type: float,
    alpha: float,
    kappa: float,
    t: int,
) -> float:
    """[summary]

    Parameters
    ----------
    population : str
        [description]
    epsilon : float
        [description]
    hazard_rate_wild_type : float
        [description]
    alpha : float
        [description]
    kappa : float
        [description]
    t : int
        [description]

    Returns
    -------
    float
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    if population == "mutant wild":
        return (1 - epsilon) * hazard_rate_wild_type + alpha * (
            ((1 + kappa) ** (t + 1)) - 1
        )
    if population == "mutant captivity":
        return alpha * (((1 + kappa) ** (t + 1)) - 1)
    if population == "hypothetical wild type":
        return hazard_rate_wild_type

    raise ValueError(
        "Population argument must be set to either 'mutant wild', 'mutant captivity' or 'hypothetical wild type'."
    )


def darwinian_lottery(cohort: np.ndarray, hazard_rate_parameters: dict) -> np.ndarray:
    """[summary]

    Parameters
    ----------
    cohort : np.ndarray
        [description]
    hazard_rate_parameters : dict
        [description]

    Returns
    -------
    np.ndarray
        [description]
    """
    number_of_individuals = cohort.shape[0]
    hazard = get_hazard_rate(**hazard_rate_parameters)
    lottery_tickets = cohort * np.random.random_sample(number_of_individuals)
    survivors = np.piecewise(
        lottery_tickets, [lottery_tickets <= hazard, lottery_tickets > hazard], [0, 1]
    )
    return survivors


def cohort_survivorship_model(
    cohort_survivorship: np.ndarray, hazard_rate_parameters: dict, t_m: int
) -> np.ndarray:
    """[summary]

    Parameters
    ----------
    cohort_survivorship : np.ndarray
        [description]
    hazard_rate_parameters : dict
        [description]
    t_m : int
        [description]

    Returns
    -------
    np.ndarray
        [description]
    """
    # On first run, make 1D array into 2D
    if len(cohort_survivorship.shape) == 1:
        cohort_survivorship = cohort_survivorship.reshape(1, -1)

    t = cohort_survivorship.shape[0]  # Time step variable
    hazard_rate_parameters["t"] = t

    current_cohort = cohort_survivorship[-1]

    if t >= t_m:
        return cohort_survivorship
    survivors = darwinian_lottery(current_cohort, hazard_rate_parameters)
    cohort_survivorship = np.append(
        cohort_survivorship, survivors.reshape(1, -1), axis=0
    )
    return cohort_survivorship_model(cohort_survivorship, hazard_rate_parameters, t_m)


def run_simulation(
    number_of_repetitions: int,
    cohort: np.ndarray,
    hazard_rate_parameters: dict,
    t_m: int,
    population: str,
) -> np.ndarray:
    """[summary]

    Parameters
    ----------
    number_of_repetitions : int
        [description]
    cohort : np.ndarray
        [description]
    hazard_rate_parameters : dict
        [description]
    t_m : int
        [description]
    population : str
        [description]

    Returns
    -------
    np.ndarray
        [description]
    """
    hazard_rate_parameters["population"] = population

    population_survivorship = []
    for _ in range(number_of_repetitions):
        cohort_survivorship = np.sum(
            cohort_survivorship_model(cohort, hazard_rate_parameters, t_m),
            axis=1,
            dtype=int,
        )
        population_survivorship.append(cohort_survivorship)

    return np.array(population_survivorship)


def get_cohort_model_data(
    t_m_captivity: int,
    t_m_wild: int,
    t_m_hyp_wt: int, 
    number_of_individuals: int,
    alpha: float,
    kappa: float,
    epsilon: float,
    hazard_rate_wild_type: float,
    number_of_repetitions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[summary]

    Parameters
    ----------
    t_m_captivity : int
        [description]
    t_m_wild : int
        [description]
    t_m_hyp_wt : int
        [description]
    number_of_individuals : int
        [description]
    alpha : float
        [description]
    kappa : float
        [description]
    epsilon : float
        [description]
    hazard_rate_wild_type : float
        [description]
    number_of_repetitions : int
        [description]

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        [description]
    """

    cohort = np.ones(number_of_individuals)

    hazard_rate_parameters = dict(epsilon=epsilon, hazard_rate_wild_type=hazard_rate_wild_type, alpha=alpha, kappa=kappa)

    captivity_population = run_simulation(number_of_repetitions, cohort, hazard_rate_parameters, t_m_captivity, MUTANT_CAP)
    wild_population = run_simulation(number_of_repetitions, cohort, hazard_rate_parameters, t_m_wild, MUTANT_WILD)
    hyp_wt_population = run_simulation(number_of_repetitions, cohort, hazard_rate_parameters, t_m_hyp_wt, HYP_WILD_TYPE)

    return captivity_population, wild_population, hyp_wt_population


def get_mean_and_std(population_survivorship: np.ndarray) -> Tuple[float, float]:
    """[summary] # Normalized # Assumes simulation always starts with all individuals alive

    Parameters
    ----------
    population_survivorship : np.ndarray
        [description]

    Returns
    -------
    Tuple[float, float]
        [description]
    """
    number_of_individuals = population_survivorship[0, 0]
    mean = np.mean(population_survivorship, axis=0) * (1 / number_of_individuals)
    std = np.std(population_survivorship * (1 / number_of_individuals), axis=0)
    return mean, std


def population_survivorship_difference(number_of_individuals, number_of_repetitions, epsilons, hazard_rates_wt, alpha, kappa, t_m, populations=(MUTANT_WILD, HYP_WILD_TYPE)):
    # #Calculates the difference in number of survivors between hypothetical wild type and mutant across the given time span 
    cohort = np.ones(number_of_individuals)

    population_simulations = defaultdict(list)

    for population in populations:
        np.random.seed(1729) # Reset seed to produce the same pseudo-random number sequence for each population
        for epsilon, hazard_rate_wt in zip(epsilons, hazard_rates_wt):
            hazard_rate_parameters = dict(epsilon=epsilon, hazard_rate_wild_type=hazard_rate_wt, alpha=alpha, kappa=kappa)
            cohort_simulation = run_simulation(number_of_repetitions, cohort, hazard_rate_parameters, t_m, population)
            population_simulations[population].append(cohort_simulation)
    
    # TODO: litt dårlig kode her med både populations og MUTANT_WILD, HYP_WILD_TYPE konstantene 
    diff = np.array(population_simulations[MUTANT_WILD]) - np.array(population_simulations[HYP_WILD_TYPE])
    mean_diff = np.mean(diff, axis=1)
    std_diff = np.std(diff, axis=1)

    return population_simulations, mean_diff, std_diff


