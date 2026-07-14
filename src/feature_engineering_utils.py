"""
Created on Mon Oct 24 11:09:06 2022

@author: Malika In-Albon
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import numpy as np


def prepare_data_for_training(poly, force_plate):
    """
    Prepare polymander motors currents and forces measured by the force plate
    for training the model. Steps: filtering, resampling, manage delay between
    sensors and remove initial frequency

    :param poly: object of LogForcePlates class
    :param force_plate: object of LogPolymander class
    :return: time, fbck position, fbck currents, forces
    """
    # 1) Filter signal
    fbck_current_filtered = filtering_signal(poly.fbck_current_data, 10)
    Fxyz_filtered = filtering_signal(force_plate.Fxyz, 20)

    # 2) Resampling force plate signal based on polymander signal
    t_s_fp_resampled, Fxyz_resampled = resample_signal(Fxyz_filtered, force_plate.t_s, poly.t_s)

    # 3) Manage delay
    t_s_cut, fbck_position_cut, fbck_current_cut, Fxyz_cut = manage_delay_between_poly_and_fp(poly.t_s,
                                                                                              poly.fbck_position_data,
                                                                                              fbck_current_filtered,
                                                                                              t_s_fp_resampled,
                                                                                              Fxyz_resampled,
                                                                                              poly.frequency)

    # 4) Remove initial sequence
    t_s_final, fbck_position_final, fbck_current_final, Fxyz_final = remove_initial_sequence(t_s_cut, fbck_position_cut,
                                                                                            fbck_current_cut,
                                                                                            Fxyz_cut, poly.frequency)
    return t_s_final, fbck_position_final, fbck_current_final, Fxyz_final


def filtering_signal(signal, sigma=6):
    filtered_signal = gaussian_filter1d(signal, sigma=sigma, axis=0)
    return filtered_signal


def resample_signal(signal_to_resample, time_to_resample, time_wanted):
    # Calculate nb_steps to have a matching frequency between the two signals
    nb_steps = int(len(time_wanted)*time_to_resample[-1]/time_wanted[-1])

    # [DEBUG]
    # period = []
    # for i in range(len(time_to_resample)-1):
    #     period.append(time_to_resample[i+1] - time_to_resample[i])
    # print('is period of polymander reliable ?', period)
    # answer no -> t_s_resampled is false because period of t_s_polymander is not constant

    # Resampling
    signal_resampled, t_s_resampled = signal.resample(signal_to_resample, nb_steps, time_to_resample)

    return t_s_resampled, signal_resampled


def cut_time(t_s, start, end):
    t_s_cut = t_s[int(start):int(end)]
    return t_s_cut


def cut_signal(init_signal, start, end):
    signal_cut = init_signal[int(start):int(end), :]
    return signal_cut


def detect_minima_of_fbck_pos(fbck_position, frequency):
    """
    Detect when the limb touches the ground (minima in feedback position of
    calf motor)

    :param fbck_position: fbck position of calf motor
    :param frequency: frequency at which the limb makes steps
    :return: indices at which polymander touches the ground
    """
    ratio = 0.5 / frequency  # width and distance of peaks are influenced by the frequency
    minima, _ = signal.find_peaks(-fbck_position[:, 9], prominence=-0.5, width=100 * ratio, distance=150 * ratio)
    return minima


def detect_initial_sequence_polymander(fbck_position, frequency, plot=False):
    """
    Detects initial sequence of polymander. It corresponds to when the limb of
    polymander touches the force plate four times before starting the movement
    for training the model

    :param fbck_position: fbck position of calf motor
    :param frequency: frequency at which polymander makes a step
    :param plot: set to True to plot the minima detection
    :return: indices of the first four minima detected
    """
    # Find minima in feedback position -> when the limb touches the ground
    minima = detect_minima_of_fbck_pos(fbck_position, frequency)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(fbck_position[:, 9], label='9FbckPosition')
        ax.plot(minima, fbck_position[minima, 9], "x")
        plt.legend()
    return minima[0:4]


def detect_initial_sequence_force_plate(Fz, frequency, plot=False):
    """
    Detects initial sequence of the force plate. It corresponds to when
    polymander touches the force plate four times before starting the
    movement for training the model

    :param Fz: force detected by the force plate in z direction
    :param frequency: frequency at which polymander makes a step
    :param plot: set to True to plot the maxima detection
    :return: indices of the first four maxima detected
    """
    # Find peaks in force plate -> when the limb touches the ground
    ratio = 0.5/frequency  # width and distance of peaks are influenced by the frequency
    peaks, _ = signal.find_peaks(Fz, prominence=1, width=100*ratio, distance=150*ratio)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(Fz, label='Fz')
        ax.plot(peaks, Fz[peaks], "x")
        ax.legend()
    return peaks[0:4]


def compute_delay_between_poly_and_force_plate(fbck_position, Fxyz, frequency):
    """
    Compute delay in steps between data measured with polymander motor sensors
    and data measured with the force plate

    :param fbck_position: fbck position of calf motor
    :param Fxyz: forces measured by the force plate
    :param frequency: frequency at which polymander makes a step
    :return:
    """
    # Detect initial sequence (4 steps)
    minima = detect_initial_sequence_polymander(fbck_position, frequency)
    peaks = detect_initial_sequence_force_plate(Fxyz[:, 2], frequency)
    # Compute offset between two signals
    offset = []
    for k in range(4):
        offset.append(peaks[k] - minima[k])
    offset = np.mean(offset)
    return offset


def manage_delay_between_poly_and_fp(t_s_poly, fbck_position, fbck_current, t_s_fp, Fxyz, frequency):
    """
    Cut signals of polymander and force plate to manage the delay and align the signals

    :param t_s_poly: time in seconds of polymander
    :param fbck_position: fbck positions of polymander motors
    :param fbck_current: fbck currents of polymander motors
    :param t_s_fp: time in seconds of force plate
    :param Fxyz: forces in x, y, z directions measured by the force plate
    :param frequency: frequency at which polymander makes a step
    :return: time, fbck positions, fbck currents and forces with same number of steps and aligned
    """

    # 1) Find delay between polymander and force plate
    delay = compute_delay_between_poly_and_force_plate(fbck_position, Fxyz, frequency)

    # 2) Cut signal to ensure that both signals are aligned according to the peaks
    # remove first seconds in force plate signal
    t_s_fp_cut_start = cut_time(t_s_fp, delay, delay + len(t_s_poly))
    Fxyz_cut_start = cut_signal(Fxyz, delay, delay + len(t_s_poly))

    # remove extra seconds in polymander
    t_s_poly_cut_end = cut_time(t_s_poly, 0, len(t_s_fp_cut_start))
    fbck_position_cut_end = cut_signal(fbck_position, 0, len(t_s_fp_cut_start))
    fbck_current_cut_end = cut_signal(fbck_current, 0, len(t_s_fp_cut_start))

    # adjust time on polymander time
    t_s_aligned = t_s_poly_cut_end

    return t_s_aligned, fbck_position_cut_end, fbck_current_cut_end, Fxyz_cut_start


def remove_initial_sequence(t_s, fbck_position, fbck_current, Fxyz, frequency):
    """
    Removes the initial sequence with the four steps to prepare data for model
    training

    :param t_s: time in seconds
    :param fbck_position: fbck positions of polymander motors
    :param fbck_current: fbck currents of polymander motors
    :param Fxyz: forces in x, y, z directions measured by the force plate
    :param frequency: frequency at which polymander makes a step
    :return: time, fbck_position, fbck_current and forces ready for model training
    """
    minima = detect_minima_of_fbck_pos(fbck_position, frequency)
    t_s_final = cut_time(t_s, minima[4], minima[-1])
    fbck_position_final = cut_signal(fbck_position, minima[4], minima[-1])
    fbck_current_final = cut_signal(fbck_current, minima[4], minima[-1])
    Fxyz_final = cut_signal(Fxyz, minima[4], minima[-1])

    return t_s_final, fbck_position_final, fbck_current_final, Fxyz_final


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


