"""Fitness calculation utilities.

This module provides functions for calculating fitness (R0 and r). 

The primary function is get_fitness_data, which takes in an array of cohort
survivorship simulations. Please see the usage in the provided case studies
for context.
""" 

import math
import numpy as np
from typing import Tuple
from scipy.optimize import brentq

from cohort_model import HYP_WILDTYPE, MUT_WILD


def get_fecundity(
    population_survivorship: np.ndarray, fertility: 'list[Tuple[int, float]]'
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


def euler_lotka(r: float, f: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The Euler–Lotka equation.

    See Eq. 7 in Omholt and Kirkwood (2021).

    Parameters
    ----------
    r : float
        The intrinsic rate of natural increase
    f : np.ndarray[float]
        A 1D array with fecundity data for a single cohort with shape (t_m,)
    t : np.ndarray[int]
        A 1D array with time steps with shape (t_m,)

    Returns
    -------
    np.ndarray
        A 1D array
    """
    return np.sum(f * np.exp(-r * t)) - 1


def get_mean_and_sem(arr: np.ndarray) -> Tuple[float, float]:
    """Return mean and standard error of the mean of an array.

    The mean and sem are not normalized.

    Parameters
    ----------
    arr : np.ndarray
        A 1D array

    Returns
    -------
    Tuple[float, float]
        1: The mean of the input array
        2: The standard error of the mean of the input array
    """
    repetition_count = arr.shape[0]
    mean = np.mean(arr)
    sem = np.std(arr) / math.sqrt(repetition_count)
    return mean, sem


def get_fitness_data(
    population_simulation: np.ndarray, fertility: 'list[Tuple[int, float]]'
) -> dict:
    """Get R0 and r data for a population for each epsilon–hazard rate wt pair.

    The method for solving for r is adapted from 
    https://scipython.com/book/chapter-8-scipy/examples/solving-the-euler-lotka-equation/ 
    (accessed 2021-05-10)

    Parameters
    ----------
    population_simulation : np.ndarray
        A 3D array of shape (number of epsilon–hazard rate wt pairs,
        repetition_count, t_m), representing population survivorship data for
        each of the epsilon–hazard rate wt pairs
    fertility : list[Tuple[int, float]]
        A list of tuples each indicating the time of a reproductive event and
        number of female offspring in this reproductive event

    Returns
    -------
    dict
        A dictionary with mean R0, SEM R0, mean r and SEM r values for each
        epsilon–hazard rate wt pair
    """
    repetition_count = population_simulation.shape[1]
    t_m = population_simulation.shape[2]

    t_steps = np.arange(0, t_m)

    r0_mean_arr = []
    r0_sem_arr = []
    r_mean_arr = []
    r_sem_arr = []

    value_count = len(population_simulation)
    for value_index in range(value_count):
        fecundity = get_fecundity(population_simulation[value_index], fertility)
        r0 = np.sum(fecundity, axis=1)  # R0: net reproductive rate

        a, b = -2, 8  # Bracketing interval for Brent's method (may need adjustments)

        r_arr = []
        for row in range(repetition_count):
            f = fecundity[row, :]
            # r: intrinsic rate of natural increase
            r = brentq(euler_lotka, a, b, args=(f, t_steps))
            r_arr.append(r)

        r0_mean, r0_sem = get_mean_and_sem(r0)
        r_mean, r_sem = get_mean_and_sem(np.array(r_arr))

        r0_mean_arr.append(r0_mean)
        r0_sem_arr.append(r0_sem)
        r_mean_arr.append(r_mean)
        r_sem_arr.append(r_sem)

    return dict(
        r0_mean=r0_mean_arr, r0_sem=r0_sem_arr, r_mean=r_mean_arr, r_sem=r_sem_arr
    )


def homarus_fertility(
    t_m: int,
    sigma: float,
    gamma: float,
    frequency: int,
    population: str,
    mu: float = None,
    alpha: float = None,
    kappa: float = None,
) -> np.ndarray:
    # TODO: Is the summary correct with respect to female offspring?
    """Get the average number of female offspring at age t based on model of Homarus gammarus fertility.

    Explanations for parameters are given in Omholt and Kirkwood (2021).

    Parameters
    ----------
    t_m : int
        The maximum time step
    sigma : float
        The sigma value
    gamma : float
        The gamma value
    frequency : int
        Reproductive events occur every x time steps
    population : str
        Either 'mutant wild' or 'hypothetical wild type'
    mu : float, optional
        The mu value, by default None
    alpha : float, optional
        The alpha value, by default None
    kappa : float, optional
        The kappa value, by default None

    Returns
    -------
    np.ndarray
        A 1D array with shape (t_m,) representing average number of female offspring produced at time t
    """
    t_steps = np.arange(0, t_m)
    fertility = sigma * (1 + gamma * t_steps / t_m)
    for t in t_steps:
        # First reproductive event starts at t = frequency - 1 (i.e. not t = 0)
        if t % frequency != frequency - 1:
            fertility[t] = 0.0

    if population == HYP_WILDTYPE:
        return fertility

    if population == MUT_WILD:
        for t in t_steps:
            fertility[t] = fertility[t] * (
                1 - mu * alpha * ((1 + kappa) ** (t + 1) - 1)
            )
        return fertility
