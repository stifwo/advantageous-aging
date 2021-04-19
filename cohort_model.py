from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

MUTANT_CAP = "mutant captivity"
MUTANT_WILD = "mutant wild"
HYP_WILD_TYPE = "hypothetical wild type"

# TODO: https://stackoverflow.com/questions/48721582/how-to-choose-proper-variable-names-for-long-names-in-python
# TODO: Add typing
# TODO: Add docstrings
# TODO: Add checks for incorrect parameter sets


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


def cohort_survivorship_model(cohort_survivorship: np.ndarray, hazard_rate_parameters: dict, t_m: int) -> np.ndarray:
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
    hazard_rate_parameters["time"] = t

    current_cohort = cohort_survivorship[-1]

    if t >= t_m:
        return cohort_survivorship
    survivors = darwinian_lottery(current_cohort, hazard_rate_parameters)
    cohort_survivorship = np.append(
        cohort_survivorship, survivors.reshape(1, -1), axis=0
    )
    return cohort_survivorship_model(cohort_survivorship, hazard_rate_parameters, t_m)


def get_hazard_rate(population: str, epsilon: float, hazard_rate_wild_type: float, alpha: float, kappa: float, t: int) -> float:
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


def run_simulation(
    number_of_repetitions: int, cohort_survivorship: np.ndarray, hazard_rate_parameters: dict, t_m: int, population: str
) -> np.ndarray:
    """[summary]

    Parameters
    ----------
    number_of_repetitions : int
        [description]
    cohort_survivorship : np.ndarray
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
            cohort_survivorship_model(cohort_survivorship, hazard_rate_parameters, t_m),
            axis=1,
            dtype=int,
        )
        population_survivorship.append(cohort_survivorship)

    return np.array(population_survivorship)


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


def plot_fig_1(
    t_m_captivity: int,
    t_m_wild: int,
    t_m_hyp_wt:int,
    captivity_population: np.ndarray,
    wild_population: np.ndarray,
    hyp_wt_population: np.ndarray,
):
    """[summary]

    Parameters
    ----------
    t_m_captivity : int
        [description]
    t_m_wild : int
        [description]
    t_m_hyp_wt : int
        [description]
    captivity_population : np.ndarray
        [description]
    wild_population : np.ndarray
        [description]
    hyp_wt_population : np.ndarray
        [description]
    """
    # TODO: Én eller to variabler for de to t_m som skal være like?

    fig, ax = plt.subplots(figsize=(6, 6))

    captivity_x = np.arange(0, t_m_captivity, 1, dtype=int)
    wild_x = np.arange(0, t_m_wild, 1, dtype=int)

    mean_captivity, std_captivity = get_mean_and_std(captivity_population)
    mean_wild, _ = get_mean_and_std(wild_population)
    mean_hyp_wt, std_hyp_wt = get_mean_and_std(hyp_wt_population)

    ax.plot(captivity_x, mean_captivity, "r-")
    ax.fill_between(
        range(t_m_captivity),
        mean_captivity - 3.0 * std_captivity,
        mean_captivity + 3.0 * std_captivity,
        color="pink",
        alpha=0.5,
    )

    ax.plot(wild_x, mean_wild, "b-")
    ax.fill_between(
        range(t_m_hyp_wt),
        mean_hyp_wt - 3.0 * std_hyp_wt,
        mean_hyp_wt + 3.0 * std_hyp_wt,
        color="lightblue",
        alpha=0.5,
    )

    # Empirical data from Austad (1989)
    # TODO: Import from text file, show calculations etc.
    X_C = [0, 30 * 217 / 65, 40 * 217 / 65, 51 * 217 / 65, 60 * 217 / 65]
    Y_C = [1.0, 0.8895, 0.592, 0.174, 0.0239]
    ax.plot(X_C, Y_C, "ro", markersize=4)

    X_W = [0, 5 * 100 / 30, 10 * 100 / 30, 15 * 100 / 30, 20 * 100 / 30, 30 * 100 / 30]
    Y_W = [1.0, 0.5498, 0.268, 0.148, 0.05, 0.0]
    ax.plot(X_W, Y_W, "bo", markersize=4)

    afont = {"fontname": "Arial"}
    ax.set_xlabel("Time", fontsize=12, **afont)
    ax.set_ylabel("Cohort survivorship", fontsize=12, **afont)

    x_size = 10
    plt.rc("xtick", labelsize=x_size)
    y_size = 10
    plt.rc("ytick", labelsize=y_size)

    fig.tight_layout()

    figure = plt.gcf()
    figure.set_size_inches(3.42, 3.42)

    plt.savefig("PNAS_fig1_Frontinella.pdf", dpi=1200, bbox_inches="tight")


def plot_fig_2():
    pass


if __name__ == "__main__":
    pass
