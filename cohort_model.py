from collections import defaultdict
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

MUTANT_CAP = "mutant captivity"
MUTANT_WILD = "mutant wild"
HYP_WILD_TYPE = "hypothetical wild type"


def get_hazard_rate(
    population: str,
    t: int,
    epsilon: float = None,
    hazard_rate_wt: float = None,
    alpha: float = None,
    kappa: float = None,
    beta: float = 0,
    t_m: int = np.inf,
    omega: float = 0,
    tau: float = 0,
) -> float:
    # OBS! beta = 0 by default and  t_m = 100, bare så vi ikke trenger å stille inn det i tidligere kode
    # Ditto for omega og tau
    # TODO: Sett alpha og kappa til 0 også?
    # TODO: Sett inn tester?
    """Calculate the hazard rate for the specified population using the provided parameters.

    Explanations for each parameter and their application are given in Omholt and Kirkwood (2021).

    Parameters
    ----------
    population : str
        Either 'mutant wild', 'mutant captivity' or 'hypothetical wild type'. 
        Determines which hazard rate version is used.
    t : int
        The current time step
    epsilon : float, optional
        The epsilon value, by default None
    hazard_rate_wt : float, optional
        The hazard rate of the wild type, by default None
    alpha : float, optional
        The alpha value, by default None
    kappa : float, optional
        The kappa value, by default None
    beta : float, optional
        The beta value, by default 0
    t_m : int, optional
        The maximum time step, by default np.inf
    omega : float, optional
        The omega value, by default 0
    tau : float, optional
        The tau value, by default 0

    Returns
    -------
    float
        The calculated hazard rate

    Raises
    ------
    ValueError
        When population parameter is not one of the three valid options
    """

    if population == MUTANT_WILD:
        return (
            (1 - epsilon) * hazard_rate_wt * (1 - beta * t / t_m)
            + alpha * (((1 + kappa) ** (t + 1)) - 1)
            + (omega * t ** tau)
        )
    if population == MUTANT_CAP:
        return alpha * (((1 + kappa) ** (t + 1)) - 1)
    if population == HYP_WILD_TYPE:
        return hazard_rate_wt * (1 - beta * t / t_m) + (omega * t ** tau)

    raise ValueError(
        "Population argument must be set to either 'mutant wild', 'mutant captivity' or 'hypothetical wild type'."
    )


def darwinian_lottery(cohort: np.ndarray, hazard_rate_parameters: dict) -> np.ndarray:
    """Determine which individuals in cohort remain alive from time step t to t + 1.

    Parameters
    ----------
    cohort : np.ndarray
        A 1D array of 1s and 0s, representing living and dead individuals, respectively
    hazard_rate_parameters : dict
        A dictionary containing minimally the chosen population and time step t, and any
        other parameters necessary for calculating the hazard rate (alpha, kappa, etc.)

    Returns
    -------
    np.ndarray
        A 1D array of 1s and 0s, where some of the original 1s are now 0s
    """
    individual_count = cohort.shape[0]
    hazard = get_hazard_rate(**hazard_rate_parameters)
    lottery_tickets = cohort * np.random.random_sample(
        individual_count
    )  # An array with a random number [0.0, 1.0) for each remaining individual in cohort
    survivors = np.piecewise(
        lottery_tickets, [lottery_tickets <= hazard, lottery_tickets > hazard], [0, 1]
    )
    return survivors


def cohort_survivorship_model(
    cohort: np.ndarray, hazard_rate_parameters: dict, t_m: int
) -> np.ndarray:
    """Recursively generate cohort survivorship data.

    The term "cohort survivorship" is taken in this context as the proportion of a
    cohort surviving to a given age, i.e. to any given time step. The "raw"
    suffix of the cohort_survivorship_raw variable is meant to indicate that it
    has not yet been aggregated.

    Parameters
    ----------
    cohort : np.ndarray
        A 1D array containing all 1s, representing living individuals
        (On subsequent recursive calls, this is a 2D array of 1s and 0s, where
        each column represents one individual and each row a single time step
        for the entire cohort)
    hazard_rate_parameters : dict
        A dictionary containing minimally the chosen population and the hazard rate of
        the wild type, and optionally any other parameters necessary for
        calculating the hazard rate for the given population (alpha, kappa, etc.)
    t_m : int
        The maximum time step

    Returns
    -------
    np.ndarray
        The unaggregated cohort survivorship, i.e. a 2D array of 1s and 0s with
        shape (t_m, number of individuals), where each column represents one
        individual and each row a single time step for the entire cohort
    """
    # TODO: Refactor sånn at t_m er inne i hazard_rate_params?

    # On first run, reshape 1D input array to 2D
    if len(cohort.shape) == 1:
        cohort_survivorship_raw = cohort.reshape(1, -1)
    else:
        cohort_survivorship_raw = cohort

    t = cohort_survivorship_raw.shape[0]  # Time step variable
    hazard_rate_parameters["t"] = t

    # Select cohort at its latest time step
    current_cohort = cohort_survivorship_raw[-1]

    if t >= t_m:
        return cohort_survivorship_raw
    survivors = darwinian_lottery(current_cohort, hazard_rate_parameters)
    cohort_survivorship_raw = np.append(
        cohort_survivorship_raw, survivors.reshape(1, -1), axis=0
    )
    return cohort_survivorship_model(
        cohort_survivorship_raw, hazard_rate_parameters, t_m
    )


def run_simulation(
    repetition_count: int,
    individual_count: int,
    hazard_rate_parameters: dict,
    t_m: int,
) -> np.ndarray:
    """[summary]

    The term "population" is chosen to represent a set of cohorts
    subject to the same hereditary and environmental conditions. 

    Thus, the term "population survivorship" is taken in this context as a set
    of survivorship arrays for multiple cohorts from the same population.

    Parameters
    ----------
    repetition_count : int
        The number of repetitions to perform
    individual_count : int
        The number of individuals in each cohort
    hazard_rate_parameters : dict
        A dictionary containing minimally the chosen population and the hazard rate of
        the wild type, and optionally any other parameters necessary for
        calculating the hazard rate for the given population (alpha, kappa, etc.)
    t_m : int
        The maximum time step

    Returns
    -------
    np.ndarray
        A 2D array with shape (repetition_count, t_m), where each column
        represents a time step and each row is a separate cohort survivorship array
    """
    cohort = np.ones(individual_count)

    population_survivorship = []
    for _ in range(repetition_count):
        cohort_survivorship = np.sum(
            cohort_survivorship_model(cohort, hazard_rate_parameters, t_m),
            axis=1,
            dtype=int,
        )
        population_survivorship.append(cohort_survivorship)

    return np.array(population_survivorship)


def get_mean_and_std(population_survivorship: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the normalized mean and standard deviation of a population survivorship.

    The function assumes that each cohort always starts out with all individuals
    alive.

    The term "population survivorship" is taken in this context as a set
    of survivorship arrays for multiple cohorts from the same population.

    Parameters
    ----------
    population_survivorship : np.ndarray
        A 2D array with shape (repetition count, max time step), where each column
        represents a time step and each row is a separate cohort survivorship array

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        1: A 1D array with the normalized mean values for each time step
        2: A 1D array with the normalized standard deviations for each time step
    """
    individual_count = population_survivorship[0, 0]
    mean = np.mean(population_survivorship, axis=0) * (1 / individual_count)
    std = np.std(population_survivorship * (1 / individual_count), axis=0)
    return mean, std


def population_survivorship_difference(
    populations: Tuple[str, str], 
    individual_count: int,
    repetition_count: int,
    epsilons: np.ndarray,
    hazard_rates_wt: np.ndarray,
    t_m: int,
    hazard_rate_parameters: dict,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Compare survivorship data of two populations.

    Calculates the difference in number of survivors between two populations
    across the given time span, for each (epsilon, hazard_rate_wt) pair

    Returns the survivorship data for each population (for each 
    (epsilon, hazard_rate_wt) pair), and the calculated mean of the difference
    and standard deviation of the difference between them (for each (epsilon,
    hazard_rate_wt) pair).

    Parameters
    ----------
    populations : Tuple[str, str]
        A tuple of the two populations to compare, e.g. (MUTANT_WILD, HYP_WILD_TYPE)
        NB! The data of the second population will be subtracted from the first population
    individual_count : int
        The number of individuals in each cohort
    repetition_count : int
        The number of repetitions to perform per population
    epsilons : np.ndarray
        An array of epsilon values to evaluate
    hazard_rates_wt : np.ndarray
        An array of hazard rate (wild type) values to evaluate
    t_m : int
        The maximum time step
    hazard_rate_parameters : dict
        A dictionary defining values for alpha and kappa, 
        and optionally additional values for for beta, omega, tau 

    Returns
    -------
    Tuple[dict, np.ndarray, np.ndarray]
        1: Dictionary with population survivorship data for each population 
        (for each (epsilon, hazard_rate_wt) pair), with population names as keys
        2: A 1D array with the mean of the difference between the populations 
        (for each (epsilon, hazard_rate_wt) pair)
        3: A 1D array with the standard deviation of the difference between the
        populations (for each (epsilon, hazard_rate_wt) pair)
    """
    population_simulations = defaultdict(list)

    for population in populations:
        hazard_rate_parameters["population"] = population
        # Reset seed to produce the same pseudo-random number sequence for each
        # population to ensure accurate comparison
        np.random.seed(1729)  
        for epsilon, hazard_rate_wt in zip(epsilons, hazard_rates_wt):
            hazard_rate_parameters["epsilon"] = epsilon
            hazard_rate_parameters["hazard_rate_wt"] = hazard_rate_wt
            cohort_simulation = run_simulation(
                repetition_count, individual_count, hazard_rate_parameters, t_m
            )
            population_simulations[population].append(cohort_simulation)

    survivorship_diff = np.array(population_simulations[populations[0]]) - np.array(
        population_simulations[populations[1]]
    )
    mean_diff = np.mean(survivorship_diff, axis=1)
    std_diff = np.std(survivorship_diff, axis=1)

    return population_simulations, mean_diff, std_diff
