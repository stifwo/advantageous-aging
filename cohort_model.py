import numpy as np
import matplotlib.pyplot as plt

MUTANT_CAP = "mutant captivity"
MUTANT_WILD = "mutant wild"
HYP_WILD_TYPE = "hypothetical wild type"

# TODO: https://stackoverflow.com/questions/48721582/how-to-choose-proper-variable-names-for-long-names-in-python

#rng = np.random.default_rng() # Random number generator
# lottery_tickets = cohort * rng.random(number_of_individuals)

def darwinian_lottery(cohort, hazard_rate_parameters):
    number_of_individuals = cohort.shape[0] 
    hazard = hazard_rate(**hazard_rate_parameters)
    lottery_tickets = cohort * np.random.random_sample(number_of_individuals)
    survivors = np.piecewise(lottery_tickets, [lottery_tickets <= hazard, lottery_tickets > hazard], [0, 1])
    return survivors
    

def cohort_survivorship_model(cohort_survivorship, hazard_rate_parameters, t_m):

    # On first run, make 1D array into 2D
    if len(cohort_survivorship.shape) == 1:
        cohort_survivorship = cohort_survivorship.reshape(1, -1)
    
    time = cohort_survivorship.shape[0] # Time variable t
    hazard_rate_parameters["time"] = time
    
    current_cohort = cohort_survivorship[-1]
    number_of_survivors = sum(current_cohort)
    
    # TODO: hvilken sammenligning av > og >= stemmer?
    if time >= t_m:
    #if number_of_survivors <= 1: 
        return cohort_survivorship
    survivors = darwinian_lottery(current_cohort, hazard_rate_parameters)
    cohort_survivorship = np.append(cohort_survivorship, survivors.reshape(1, -1), axis=0)        
    return cohort_survivorship_model(cohort_survivorship, hazard_rate_parameters, t_m)

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

def run_simulation(number_of_repetitions, cohort_survivorship, hazard_rate_parameters, t_max, population):
    # TODO: Rename t_max til t_m

    hazard_rate_parameters["population"] = population

    results = []
    for _ in range(number_of_repetitions):
        n_survivors_per_timestep = np.sum(cohort_survivorship_model(cohort_survivorship, hazard_rate_parameters, t_max), axis=1, dtype=int)
        results.append(n_survivors_per_timestep)
    
    # TODO: Bruker t_m og går ikke til null lengre, derfor ikke nødvendig med denne sjekken
    for repetition in results:
        if len(repetition) < t_max:
            raise Exception("Hva gjør vi når dette skjer?") # TODO
    
    return np.array(results)
    #return np.array([repetition[:t_max] for repetition in results])


def get_mean_and_std(survivors_per_timestep):
    # Normalized
    number_of_individuals = survivors_per_timestep[0, 0] # Assumes simulation always starts with all individuals alive
    mean = np.mean(survivors_per_timestep, axis=0) * (1 / number_of_individuals) 
    std = np.std(survivors_per_timestep * (1 / number_of_individuals), axis=0)
    return mean, std


def plot_fig_1(t_m_captivity, t_m_wild, t_m_hyp_wt, captivity_population, wild_population, hyp_wt_population):
    # TODO: Én eller to variabler for de to t_m som skal være like?
    # Trenger egentlig ikke mate noe som helst inn i denne funksjonen, den burde kalle på andre funksjoner?
    
    fig1, ax = plt.subplots(figsize=(6, 6))
    
    captivity_x = np.arange(0, t_m_captivity, 1, dtype=int)
    wild_x = np.arange(0, t_m_wild, 1, dtype=int)
    
    
    mean_captivity, std_captivity = get_mean_and_std(captivity_population)
    mean_wild, _ = get_mean_and_std(wild_population)
    mean_hyp_wt, std_hyp_wt = get_mean_and_std(hyp_wt_population)
    
    
    ax.plot(captivity_x, mean_captivity, 'r-')
    ax.fill_between(range(t_m_captivity), 
                    mean_captivity - 3.0 * std_captivity, 
                    mean_captivity + 3.0 * std_captivity, 
                    color='pink', 
                    alpha=0.5)

    ax.plot(wild_x, mean_wild, 'b-')
    ax.fill_between(range(t_m_hyp_wt), 
                    mean_hyp_wt - 3.0 * std_hyp_wt, 
                    mean_hyp_wt + 3.0 * std_hyp_wt, 
                    color='lightblue', 
                    alpha=0.5)
    
    # Empirical data from Austad (1989)
    # TODO: Import from text file, show calculations etc. 
    X_C = [0, 30 * 217 / 65, 40 * 217 / 65, 51 * 217 / 65, 60 * 217 / 65]
    Y_C = [1.0,0.8895,0.592,0.174,0.0239]
    ax.plot(X_C,Y_C,'ro',markersize=4)

    X_W = [0, 5*100/30, 10*100/30,15*100/30,20*100/30,30*100/30]
    Y_W = [1.0,0.5498,0.268,0.148,0.05,0.0]
    ax.plot(X_W,Y_W,'bo',markersize=4)
    
    afont = {'fontname': 'Arial'}
    ax.set_xlabel("Time", fontsize = 12, **afont)
    ax.set_ylabel("Cohort survivorship", fontsize=12, **afont)

    x_size = 10
    plt.rc('xtick', labelsize=x_size)
    y_size = 10
    plt.rc('ytick', labelsize=y_size)

    fig1.tight_layout()

    figure = plt.gcf()
    figure.set_size_inches(3.42, 3.42)

    plt.savefig('PNAS_fig1_Frontinella.pdf', dpi=1200, bbox_inches="tight")
    

    

if __name__ == "__main__":
    pass

