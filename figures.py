import numpy as np
from matplotlib import pyplot as plt


def plot_fig_1(t_m_captivity, t_m_wild, t_m_hyp_wt, mean_captivity, std_captivity, mean_wild, mean_hyp_wt, std_hyp_wt):
    
    
    captivity_x = np.arange(0, t_m_captivity)
    wild_x = np.arange(0, t_m_wild)

    fig, ax = plt.subplots(figsize=(6, 6))

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

    plt.savefig(f"PNAS_fig1_Frontinella.pdf", dpi=1200, bbox_inches="tight")

def plot_fig_2(t_m, mean_diff, std_diff, number_of_repetitions):
    C = np.arange(0, t_m, 1, dtype=int)
    fig1, ax = plt.subplots(figsize=(6,6))
    l1, = ax.plot(C, mean_diff[0])
    ax.fill_between(range(t_m), mean_diff[0] - 3.0 * std_diff[0] / np.sqrt(number_of_repetitions), mean_diff[0] + 3.0 * std_diff[0] / np.sqrt(number_of_repetitions), alpha=0.5)

    l2, = ax.plot(C, mean_diff[1])
    ax.fill_between(range(t_m), mean_diff[1] - 3.0 * std_diff[1] / np.sqrt(number_of_repetitions), mean_diff[1] + 3.0 * std_diff[1] / np.sqrt(number_of_repetitions), alpha=0.5)

    l3, = ax.plot(C, mean_diff[2])
    ax.fill_between(range(t_m), mean_diff[2] - 3.0 * std_diff[2] / np.sqrt(number_of_repetitions), mean_diff[2] + 3.0 * std_diff[2] / np.sqrt(number_of_repetitions), alpha=0.5)

    l4, = ax.plot(C, mean_diff[3])
    ax.fill_between(range(t_m), mean_diff[3] - 3.0 * std_diff[3] / np.sqrt(number_of_repetitions), mean_diff[3] + 3.0 * std_diff[3] / np.sqrt(number_of_repetitions), alpha=0.5)

    ax.legend((l1,l2,l3,l4),('$\epsilon$=0.01', '$\epsilon$=0.02', '$\epsilon$=0.03','$\epsilon$=0.04'))
    afont = {'fontname': 'Arial'}
    ax.set_xlabel("Time", fontsize=12, **afont)
    ax.set_ylabel("# of mutant - wild type individuals", fontsize = 12,**afont)

    x_size = 10
    plt.rc('xtick', labelsize=x_size)
    y_size = 10
    plt.rc('ytick', labelsize=y_size)

    fig1.tight_layout()

    figure = plt.gcf()
    figure.set_size_inches(3.42, 3.42)
    plt.savefig('PNAS_fig2_Frontinella.pdf', dpi=1200, bbox_inches="tight")


def plot_fig_3(fitness_stats_wt, fitness_stats_mut):
    mean_r0_wt = fitness_stats_wt["mean_r0"]
    sem_r0_wt = fitness_stats_wt["sem_r0"]
    mean_r_wt = fitness_stats_wt["mean_r"]
    sem_r_wt = fitness_stats_wt["sem_r"]

    mean_r0_mut = fitness_stats_mut["mean_r0"]
    sem_r0_mut = fitness_stats_mut["sem_r0"]
    mean_r_mut = fitness_stats_mut["mean_r"]
    sem_r_mut = fitness_stats_mut["sem_r"]

    y1_pos = np.array([0,3,6,9])
    y2_pos = np.array([1,4,7,10])
    y3_pos = np.array([2,5,8,11])

    y4_pos = np.array([12,15,18,21])
    y5_pos = np.array([13,16,19,22])
    y6_pos = np.array([14,17,20])

    y7_pos = [11]
    dummy_R0 = np.zeros(4)
    dummy_r = np.zeros(3)

    fig,ax1 = plt.subplots(figsize=(6,6))

    ax1.bar(y1_pos, mean_r0_wt,width = 1.0,yerr=sem_r0_wt, align='center', alpha=0.4, ecolor='black', capsize=3, color = 'C0')
    ax1.bar(y2_pos, mean_r0_mut,width = 1.0,yerr=sem_r0_mut, align='center', alpha=0.8, ecolor='black', capsize=3, color = 'C0')
    ax1.bar(y3_pos, dummy_R0, width=0.3, color='w') #The width spec does not seem to work. width only refers to the relative width in the slot..

    ax1.set_ylabel('R0',fontsize=14)
    
    ax1.set_ylim(0.92 * mean_r0_wt[0], 1.005 * mean_r0_wt[0])

    ax2 = ax1.twinx()

    ax2.bar(y4_pos,mean_r_wt,width= 1.0,yerr=sem_r_wt, align='center', alpha=0.4, ecolor='black', capsize=3,color = 'C3')
    ax2.bar(y5_pos,mean_r_mut,width = 1.0,yerr=sem_r_mut, align='center', alpha=0.8, ecolor='black', capsize=3,color = 'C3')
    ax2.bar(y6_pos,dummy_r,width=0.3,color = 'w') 

    ax2.set_ylabel('r ', fontsize=14)
    ax2.set_ylim(0.95*mean_r_wt[0],1.0045*mean_r_wt[0])

    ax2.set_xticks(y7_pos)

    xticks = [' wt($\epsilon$) vs mut($\epsilon$)']
    ax1.set_xticklabels(xticks, fontsize=13)

    fig.tight_layout()

    figure = plt.gcf()
    figure.set_size_inches(3.42, 3.42)
    plt.savefig('PNAS_fig3_Frontinella.pdf', dpi=1200, bbox_inches="tight")

def plot_fig_4():
    pass

def plot_fig_5():
    pass
