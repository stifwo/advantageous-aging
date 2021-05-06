import math
import numpy as np
from typing import Tuple
from scipy.optimize import brentq

from cohort_model import HYP_WILD_TYPE, MUTANT_WILD


def get_fecundity(
    population_survivorship: np.ndarray, fertility: "list[Tuple[int, float]]"
) -> np.ndarray:
    """Calculate the fecundity of a population.

    Fecundity at time t is m(t) * l(t), i.e. the average number of female
    offspring produced by a female at time t * the probability of being alive
    at time t. The fecundity is calculated separately for each cohort in the
    population.

    This function assumes all individuals are alive at the beginning.

    Parameters
    ----------
    population_survivorship : np.ndarray
        A 2D array with shape (repetition_count, t_m), where each column
        represents a time step and each row is a separate cohort survivorship array
    fertility : list[Tuple[int, float]]
        A list of tuples each indicating the time of a reproductive event and
        number of female offspring in this reproductive event

    Returns
    -------
    np.ndarray
        A 2D array with shape (repetition_count, t_m), where each column
        represents a time step and each row the fecundity data for a single cohort
    """
    repetition_count = population_survivorship.shape[0]
    individual_count = population_survivorship[0, 0]
    t_m = population_survivorship.shape[1]

    # m(t): Average number of female offspring produced by an individual at time t
    m_t = np.zeros((repetition_count, t_m))
    for reproduction_index, female_offspring_count in fertility:
        m_t[:, reproduction_index] = female_offspring_count

    # l(t): Probability of being alive at age/time t
    l_t = population_survivorship * (1 / individual_count)

    fecundity = np.multiply(l_t, m_t)
    return fecundity


def euler_lotka(r, f, t):
    return np.sum(f * np.exp(-r * t)) - 1


def get_mean_and_sem(arr: np.ndarray) -> Tuple[float, float]:
    # NOT normalized
    repetition_count = arr.shape[0]
    mean = np.mean(arr)
    sem = np.std(arr) / math.sqrt(repetition_count)
    return mean, sem


def get_fitness_data(population_simulation, repetition_count, t_m, fertility):
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
        for i in range(repetition_count):
            f = fecundity[i, :]
            r = brentq(euler_lotka, a, b, args=(f, t_arr))
            r_arr.append(r)

        mean_r0, sem_r0 = get_mean_and_sem(r0)
        mean_r, sem_r = get_mean_and_sem(np.array(r_arr))

        mean_r0_arr.append(mean_r0)
        sem_r0_arr.append(sem_r0)
        mean_r_arr.append(mean_r)
        sem_r_arr.append(sem_r)

    return dict(
        mean_r0=mean_r0_arr, sem_r0=sem_r0_arr, mean_r=mean_r_arr, sem_r=sem_r_arr
    )


def homarus_fertility(
    t_m, sigma, gamma, frequency, population, mu=None, alpha=None, kappa=None
):
    t_arr = np.arange(0, t_m)
    fertility = sigma * (1 + gamma * t_arr / t_m)
    for t in t_arr:
        if (
            t % frequency != frequency - 1
        ):  # First birth term starts at t = frequency - 1 (i.e. not t = 0)
            fertility[t] = 0.0

    if population == HYP_WILD_TYPE:
        return fertility

    if population == MUTANT_WILD:
        for t in t_arr:
            fertility[t] = fertility[t] * (
                1 - mu * alpha * ((1 + kappa) ** (t + 1) - 1)
            )
        return fertility
