"""Figure plotting utilities for figures in Omholt and Kirkwood (2021).""" 

import os

import numpy as np
from matplotlib import pyplot as plt


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

def plot_fig_1(
    t_m_captivity,
    t_m_wild,
    t_m_hyp_wt,
    captivity_mean,
    captivity_std,
    wild_mean,
    hyp_wt_mean,
    hyp_wt_std,
):

    captivity_x = np.arange(0, t_m_captivity)
    wild_x = np.arange(0, t_m_wild)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(captivity_x, captivity_mean, 'r-')
    ax.fill_between(
        range(t_m_captivity),
        captivity_mean - 3.0 * captivity_std,
        captivity_mean + 3.0 * captivity_std,
        color='pink',
        alpha=0.5,
    )

    ax.plot(wild_x, wild_mean, 'b-')
    ax.fill_between(
        range(t_m_hyp_wt),
        hyp_wt_mean - 3.0 * hyp_wt_std,
        hyp_wt_mean + 3.0 * hyp_wt_std,
        color='lightblue',
        alpha=0.5,
    )

    # Plotting wild data (extraction from Austad (1989))
    X_C = np.genfromtxt(f'{ROOT_DIR}/data/austad_1989/captivity_x.txt')
    Y_C = np.genfromtxt(f'{ROOT_DIR}/data/austad_1989/captivity_y.txt')
    ax.plot(X_C, Y_C, 'ro', markersize=4)

    X_W = [0, 5 * 100 / 30, 10 * 100 / 30, 15 * 100 / 30, 20 * 100 / 30, 30 * 100 / 30]
    Y_W = [1.0, 0.5498, 0.268, 0.148, 0.05, 0.0]
    ax.plot(X_W, Y_W, 'bo', markersize=4)

    afont = {'fontname': 'Arial'}
    ax.set_xlabel('Time', fontsize=12, **afont)
    ax.set_ylabel('Cohort survivorship', fontsize=12, **afont)

    x_size = 10
    plt.rc('xtick', labelsize=x_size)
    y_size = 10
    plt.rc('ytick', labelsize=y_size)

    fig.tight_layout()

    figure = plt.gcf()
    figure.set_size_inches(3.42, 3.42)

    figure.savefig(f'{ROOT_DIR}/figures/PNAS_fig1_Frontinella.pdf', dpi=1200, bbox_inches='tight')


def plot_fig_2(t_m, mean_diff, std_diff, repetition_count, save_pdf=True):
    C = np.arange(0, t_m, 1, dtype=int)
    fig1, ax = plt.subplots(figsize=(6, 6))
    (l1,) = ax.plot(C, mean_diff[0])
    ax.fill_between(
        range(t_m),
        mean_diff[0] - 3.0 * std_diff[0] / np.sqrt(repetition_count),
        mean_diff[0] + 3.0 * std_diff[0] / np.sqrt(repetition_count),
        alpha=0.5,
    )

    (l2,) = ax.plot(C, mean_diff[1])
    ax.fill_between(
        range(t_m),
        mean_diff[1] - 3.0 * std_diff[1] / np.sqrt(repetition_count),
        mean_diff[1] + 3.0 * std_diff[1] / np.sqrt(repetition_count),
        alpha=0.5,
    )

    (l3,) = ax.plot(C, mean_diff[2])
    ax.fill_between(
        range(t_m),
        mean_diff[2] - 3.0 * std_diff[2] / np.sqrt(repetition_count),
        mean_diff[2] + 3.0 * std_diff[2] / np.sqrt(repetition_count),
        alpha=0.5,
    )

    (l4,) = ax.plot(C, mean_diff[3])
    ax.fill_between(
        range(t_m),
        mean_diff[3] - 3.0 * std_diff[3] / np.sqrt(repetition_count),
        mean_diff[3] + 3.0 * std_diff[3] / np.sqrt(repetition_count),
        alpha=0.5,
    )

    ax.legend(
        (l1, l2, l3, l4),
        ('$\epsilon$=0.01', '$\epsilon$=0.02', '$\epsilon$=0.03', '$\epsilon$=0.04'),
    )
    afont = {'fontname': 'Arial'}
    ax.set_xlabel('Time', fontsize=12, **afont)
    ax.set_ylabel("# of mutant - wild type individuals", fontsize=12, **afont)

    x_size = 10
    plt.rc('xtick', labelsize=x_size)
    y_size = 10
    plt.rc('ytick', labelsize=y_size)

    fig1.tight_layout()

    figure = plt.gcf()
    figure.set_size_inches(3.42, 3.42)
    if save_pdf:
        plt.savefig(f'{ROOT_DIR}/figures/PNAS_fig2_Frontinella.pdf', dpi=1200, bbox_inches='tight')


def plot_fig_3(fitness_stats_wt, fitness_stats_mut):
    r0_wt_mean = fitness_stats_wt['r0_mean']
    r0_wt_sem = fitness_stats_wt['r0_sem']
    r_wt_mean = fitness_stats_wt['r_mean']
    r_wt_sem = fitness_stats_wt['r_sem']

    r0_mut_mean = fitness_stats_mut['r0_mean']
    r0_mut_sem = fitness_stats_mut['r0_sem']
    r_mut_mean = fitness_stats_mut['r_mean']
    r_mut_sem = fitness_stats_mut['r_sem']

    y1_pos = np.array([0, 3, 6, 9])
    y2_pos = np.array([1, 4, 7, 10])
    y3_pos = np.array([2, 5, 8, 11])

    y4_pos = np.array([12, 15, 18, 21])
    y5_pos = np.array([13, 16, 19, 22])
    y6_pos = np.array([14, 17, 20])

    y7_pos = [11]
    dummy_R0 = np.zeros(4)
    dummy_r = np.zeros(3)

    fig, ax1 = plt.subplots(figsize=(6, 6))

    ax1.bar(
        y1_pos,
        r0_wt_mean,
        width=1.0,
        yerr=r0_wt_sem,
        align='center',
        alpha=0.4,
        ecolor='black',
        capsize=3,
        color='C0',
    )
    ax1.bar(
        y2_pos,
        r0_mut_mean,
        width=1.0,
        yerr=r0_mut_sem,
        align='center',
        alpha=0.8,
        ecolor='black',
        capsize=3,
        color='C0',
    )
    ax1.bar(
        y3_pos, dummy_R0, width=0.3, color='w'
    )  # The width spec does not seem to work. width only refers to the relative width in the slot.

    ax1.set_ylabel('R0', fontsize=14)

    ax1.set_ylim(0.92 * r0_wt_mean[0], 1.005 * r0_wt_mean[0])

    ax2 = ax1.twinx()

    ax2.bar(
        y4_pos,
        r_wt_mean,
        width=1.0,
        yerr=r_wt_sem,
        align='center',
        alpha=0.4,
        ecolor='black',
        capsize=3,
        color='C3',
    )
    ax2.bar(
        y5_pos,
        r_mut_mean,
        width=1.0,
        yerr=r_mut_sem,
        align='center',
        alpha=0.8,
        ecolor='black',
        capsize=3,
        color='C3',
    )
    ax2.bar(y6_pos, dummy_r, width=0.3, color='w')

    ax2.set_ylabel('r', fontsize=14)
    ax2.set_ylim(0.95 * r_wt_mean[0], 1.0045 * r_wt_mean[0])

    ax2.set_xticks(y7_pos)

    xticks = ['wt($\epsilon$) vs mut($\epsilon$)']
    ax1.set_xticklabels(xticks, fontsize=13)

    fig.tight_layout()

    figure = plt.gcf()
    figure.set_size_inches(3.42, 3.42)
    plt.savefig(f'{ROOT_DIR}/figures/PNAS_fig3_Frontinella.pdf', dpi=1200, bbox_inches='tight')


def plot_fig_4(
    t_m_cap_f,
    t_m_cap_m,
    t_m_wild_f,
    t_m_wild_m,
    mean_cap_f,
    mean_cap_m,
    mean_wild_f,
    mean_wild_m,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plotting captive females (extraction from Kawasaki et al (2008))
    X_C_F = np.genfromtxt(f'{ROOT_DIR}/data/kawasaki_2008/captivity_females_x.txt')
    Y_C_F = np.genfromtxt(f'{ROOT_DIR}/data/kawasaki_2008/captivity_females_y.txt') 
    ax.plot(X_C_F, Y_C_F, 'ro', markersize=4)

    C1 = np.arange(0, t_m_cap_f)
    ax.plot(C1, mean_cap_f, 'r-')

    # Plotting captive males (extraction from Kawasaki et al (2008))
    X_C_M = np.genfromtxt(f'{ROOT_DIR}/data/kawasaki_2008/captivity_males_x.txt') 
    Y_C_M = np.genfromtxt(f'{ROOT_DIR}/data/kawasaki_2008/captivity_males_y.txt') 
    ax.plot(X_C_M, Y_C_M, 'bo', markersize=4)

    C2 = np.arange(0, t_m_cap_m)
    ax.plot(C2, mean_cap_m, 'b-')

    # Plotting wild females (extraction from Kawasaki et al (2008))
    X_W_F = np.genfromtxt(f'{ROOT_DIR}/data/kawasaki_2008/wild_females_x.txt') 
    Y_W_F = np.genfromtxt(f'{ROOT_DIR}/data/kawasaki_2008/wild_females_y.txt') 
    ax.plot(X_W_F, Y_W_F, 'ro', markersize=4)

    C3 = np.arange(0, t_m_wild_f)
    ax.plot(C3, mean_wild_f, 'r-')

    # Plotting wild males (extraction from Kawasaki et al (2008))
    X_W_M = np.genfromtxt(f'{ROOT_DIR}/data/kawasaki_2008/wild_males_x.txt') 
    Y_W_M = np.genfromtxt(f'{ROOT_DIR}/data/kawasaki_2008/wild_males_y.txt')
    ax.plot(X_W_M, Y_W_M, 'bo', markersize=4)

    C4 = np.arange(0, t_m_wild_m)
    ax.plot(C4, mean_wild_m, 'b-')

    afont = {'fontname': 'Arial'}
    ax.set_xlabel('Time', fontsize=12, **afont)
    ax.set_ylabel("Cohort survivorship", fontsize=12, **afont)

    x_size = 10
    plt.rc('xtick', labelsize=x_size)
    y_size = 10
    plt.rc('ytick', labelsize=y_size)

    fig.tight_layout()

    figure = plt.gcf()
    figure.set_size_inches(3.42, 3.42)

    plt.savefig(f'{ROOT_DIR}/figures/PNAS_fig4_Telostylinus.pdf', dpi=1200, bbox_inches='tight')


def plot_fig_5(fitness_stats_wt, fitness_stats_mut):
    r0_wt_mean = fitness_stats_wt['r0_mean']
    r0_wt_sem = fitness_stats_wt['r0_sem']
    r_wt_mean = fitness_stats_wt['r_mean']
    r_wt_sem = fitness_stats_wt['r_sem']

    r0_mut_mean = fitness_stats_mut['r0_mean']
    r0_mut_sem = fitness_stats_mut['r0_sem']
    r_mut_mean = fitness_stats_mut['r_mean']
    r_mut_sem = fitness_stats_mut['r_sem']

    y1_pos = [0]
    y2_pos = [1, 2, 3, 4]
    y3_pos = [5]
    y4_pos = [6, 7, 8, 9]
    y5_pos = [4]

    fig, ax1 = plt.subplots(figsize=(6, 6))
    afont = {'fontname': 'Arial'}

    ax1.bar(
        y1_pos,
        r0_wt_mean[0],
        yerr=r0_wt_sem[0],
        align='center',
        alpha=0.4,
        ecolor='black',
        capsize=3,
        color='C0',
    )
    ax1.bar(
        y2_pos,
        r0_mut_mean,
        yerr=r0_mut_sem,
        align='center',
        alpha=0.8,
        ecolor='black',
        capsize=3,
        color='C0',
    )

    ax1.set_ylabel('R0', fontsize=14, **afont)
    ax1.set_ylim(0.96 * r0_wt_mean[0], 1.035 * r0_wt_mean[0])

    ax2 = ax1.twinx()

    ax2.bar(
        y3_pos,
        r_wt_mean[0],
        yerr=r_wt_sem[0],
        align='center',
        alpha=0.4,
        ecolor='black',
        capsize=3,
        color='C3',
    )
    ax2.bar(
        y4_pos,
        r_mut_mean,
        yerr=r_mut_sem,
        align='center',
        alpha=0.8,
        ecolor='black',
        capsize=3,
        color='C3',
    )

    ax2.set_ylabel('r', fontsize=16, **afont)
    ax2.set_ylim(0.98 * r_wt_mean[0], 1.019 * r_wt_mean[0])

    ax2.set_xticks(y5_pos)
    xticks = ['wt($\epsilon$) vs mut($\epsilon$)']
    ax1.set_xticklabels(xticks, fontsize=13)

    fig.tight_layout()

    figure = plt.gcf()
    figure.set_size_inches(3.42, 3.42)
    plt.savefig(f'{ROOT_DIR}/figures/PNAS_fig5_Homarus.pdf', dpi=1200, bbox_inches='tight')
