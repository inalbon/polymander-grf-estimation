"""
Created on Wed July 29 15:24:03 2026

@author: Malika In-Albon
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from polymander.signal_processing import detect_initial_sequence_polymander

def plot_aligned_signals(t_s, fbck_position, Fxyz, frequency):
    minima = detect_initial_sequence_polymander(fbck_position, frequency)
    fig, ax = plt.subplots()
    ax.set_title('Alignment of signals with initial sequence')
    ax.set(xlabel='time [s]')
    ax.plot(t_s, Fxyz[:, 2], label='Fz [N]')
    ax.plot(t_s, fbck_position[:, 9], label='9FbckPosition [rad]')
    ax.vlines(t_s[minima[0:4]], 0, max(Fxyz[:, 2]), colors='lime', linestyles='dashed', label='first 4 steps')
    ax.legend(loc='upper right')


def subplots_currents_and_forces(t_s, hip_current, calf_current, Fz):
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t_s, Fz, label='Fz')
    axs[0].set(xlabel='time [s]', ylabel='Force [N]')
    axs[1].plot(t_s, hip_current, label='Hip motor')
    axs[1].set(xlabel='time [s]', ylabel='Current [mA]')
    axs[2].plot(t_s, calf_current, label='Calf motor')
    axs[2].set(xlabel='time [s]', ylabel='Current [mA]')
    for ax in axs.flat:
        ax.legend()
        ax.label_outer()


def plot_3d_curents_time(t_s, hip_current, calf_current):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Currents vs time')
    ax.plot(hip_current, calf_current, t_s)
    ax.set(xlabel='8FbckCurrent [mA]', ylabel='9FbckCurrent [mA]', zlabel='time [s]')


def plot_3d_currents_force(Fz, hip_current, calf_current):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Currents vs force')
    ax.plot(hip_current, calf_current, Fz)
    ax.set(xlabel='8FbckCurrent [mA]', ylabel='9FbckCurrent [mA]', zlabel='Fz [N]')


def plot_3d_metrics(metrics, metric_name, errors=None):
    """
    Plot the metrics in function of amplitude and frequency modulations

    :param metrics: array with metrics at each amplitude and frequency
    :param metric_name: name of the metric
    :param errors: difference between minimum and maximum value when using k-fold
                   cross validation
    """
    results = metrics

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Amplitude [rad]', labelpad=10)
    ax.set_ylabel('Frequency [Hz]', labelpad=10)
    ax.set_zlabel(metric_name)

    xlabels = np.array(['0.2', '0.35', '0.5'])
    ylabels = np.array(['0.1', '0.5', '1.0'])

    x, y = np.random.rand(2, 100) * 3
    hist, xedges, yedges = np.histogram2d(x, y, bins=3, range=[[0, 3], [0, 3]])

    # Construct arrays for the anchor positions of the 9 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")

    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 9 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = results

    # Set ticks
    ax.w_xaxis.set_ticks(ypos[0:3] + dx / 2.)
    ax.w_xaxis.set_ticklabels(xlabels)

    ax.w_yaxis.set_ticks(ypos[0:3] + dy / 2.)
    ax.w_yaxis.set_ticklabels(ylabels)

    # Set colors
    values = np.linspace(0.2, 1., xpos.ravel().shape[0])
    colors = cm.rainbow(values)

    # Plot 3D bars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=0.6, color=colors)

    if errors is not None:
        # Plot error bars
        x_error = [[x + dx] * 3 for x in range(3)]
        x_error = [item for sublist in x_error for item in sublist]
        y_error = [(y + dy) % 3 for y in range(9)]

        for (i, j, k, e) in zip(x_error, y_error, dz, errors):
            ax.errorbar(i, j, k, e, color='black', capsize=4)

    ax.view_init(30, 130)


