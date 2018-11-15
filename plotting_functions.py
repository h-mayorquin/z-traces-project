import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances
from analysis_functions import calculate_patterns_timings


class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def set_text(ax, coordinate_from, coordinate_to, fontsize=25, color='black'):
    """
    Set text in an axis
    :param ax:  The axis
    :param coordinate_from: From pattern
    :param coordinate_to: To pattern
    :param fontsize: The fontsize
    :return:
    """
    message = str(coordinate_from) + '->' + str(coordinate_to)
    ax.text(coordinate_from, coordinate_to, message, ha='center', va='center',
            rotation=315, fontsize=fontsize, color=color)


def plot_sequences(sequences, minicolumns):
    sns.set_style("whitegrid", {'axes.grid': False})
    sequence_matrix = np.zeros((len(sequences), minicolumns))
    for index, sequence in enumerate(sequences):
        sequence_matrix[index, sequence] = index + 1

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)

    cmap = matplotlib.cm.Paired
    cmap = matplotlib.cm.prism
    cmap.set_under('white')

    ax.imshow(sequence_matrix, cmap=cmap, vmin=0.5)
    sns.set()


def plot_weight_matrix(manager, one_hypercolum=True, ax=None, vmin=None, title=True):
    with sns.axes_style("whitegrid", {'axes.grid': False}):

        w = manager.nn.w

        if one_hypercolum:
            w = w[:manager.nn.minicolumns, :manager.nn.minicolumns]

        # aux_max = np.max(np.abs(w))
        norm = MidpointNormalize(midpoint=0)
        cmap = matplotlib.cm.RdBu_r

        if ax is None:
            # sns.set_style("whitegrid", {'axes.grid': False})
            fig = plt.figure()
            ax = fig.add_subplot(111)

        im = ax.imshow(w, cmap=cmap, interpolation='nearest', norm=norm, vmin=vmin)

        if title:
            ax.set_title('w connectivity')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = ax.get_figure().colorbar(im, ax=ax, cax=cax)

    return ax


def plot_persistent_matrix(manager, ax=None):
    with sns.axes_style("whitegrid", {'axes.grid': False}):
        title = r'$T_{persistence}$'

        cmap = matplotlib.cm.Reds_r
        cmap.set_bad(color='white')
        cmap.set_under('black')

        if ax is None:
            # sns.set_style("whitegrid", {'axes.grid': False})
            fig = plt.figure()
            ax = fig.add_subplot(111)

        im = ax.imshow(manager.T, cmap=cmap, interpolation='None', vmin=0.0)
        ax.set_title(title)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax.get_figure().colorbar(im, ax=ax, cax=cax)


def plot_winning_pattern(manager, ax=None, separators=False, remove=0):
    """
    Plots the winning pattern for the sequences
    :param manager: A network manager instance
    :param ax: an axis instance
    :return:
    """

    n_patterns = manager.nn.minicolumns
    T_total = manager.T_training_total
    # Get the angles
    angles = calculate_angle_from_history(manager)
    winning = calculate_winning_pattern_from_distances(angles) + 1  # Get them in the color bounds
    timings = calculate_patterns_timings(winning, manager.dt, remove)
    winners = [x[0] for x in timings]
    pattern_times = [x[2] + 0.5 * x[1] for x in timings]
    # 0.5 is for half of the time that the pattern lasts ( that is x[1])
    start_times = [x[2] for x in timings]

    # Filter the data
    angles[angles < 0.1] = 0
    filter = np.arange(1, angles.shape[1] + 1)
    angles = angles * filter

    # Add a column of zeros and of the winners to the stack
    zeros = np.zeros_like(winning)
    angles = np.column_stack((angles, zeros, winning))

    # Plot
    with sns.axes_style("whitegrid", {'axes.grid': False}):
        if ax is None:
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)

        fig = ax.figure

        cmap = matplotlib.cm.Paired
        cmap.set_under('white')
        extent = [0, n_patterns + 2, T_total, 0]

        im = ax.imshow(angles, aspect='auto', interpolation='None', cmap=cmap, vmax=filter[-1], vmin=0.9, extent=extent)
        ax.set_title('Sequence of patterns')
        ax.set_xlabel('Patterns')
        ax.set_ylabel('Time')

        # Put labels in both axis
        ax.tick_params(labeltop=False, labelright=False)

        # Add seperator
        ax.axvline(n_patterns, color='k', linewidth=2)
        ax.axvline(n_patterns + 1, color='k', linewidth=2)
        ax.axvspan(n_patterns, n_patterns + 1, facecolor='gray', alpha=0.3)

        # Add the sequence as a text in a column
        x_min = n_patterns * 1.0/ (n_patterns + 2)
        x_max = (n_patterns + 1) * 1.0 / (n_patterns + 2)

        for winning_pattern, time, start_time in zip(winners, pattern_times, start_times):
            ax.text(n_patterns + 0.5, time, str(winning_pattern), va='center', ha='center')
            if separators:
                ax.axhline(y=start_time, xmin=x_min, xmax=x_max, linewidth=2, color='black')

        # Colorbar
        bounds = np.arange(0.5, n_patterns + 1.5, 1)
        ticks = np.arange(1, n_patterns + 1, 1)

        # Set the ticks positions
        ax.set_xticks(bounds)
        # Set the strings in those ticks positions
        strings = [str(int(x + 1)) for x in bounds[:-1]]
        strings.append('Winner')
        ax.xaxis.set_major_formatter(plt.FixedFormatter(strings))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
        fig.colorbar(im, cax=cbar_ax, boundaries=bounds, cmap=cmap, ticks=ticks, spacing='proportional')


def plot_sequence(manager):

    T_total = manager.T_training_total
    # Get the angles
    angles = calculate_angle_from_history(manager)
    winning = calculate_winning_pattern_from_distances(angles)
    winning = winning[np.newaxis]

    # Plot
    sns.set_style("whitegrid", {'axes.grid': False})

    filter = np.arange(1, angles.shape[1] + 1)
    angles = angles * filter

    cmap = matplotlib.cm.Paired
    cmap.set_under('white')

    extent = [0, T_total, manager.nn.minicolumns, 0]
    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(111)
    im1 = ax1.imshow(winning, aspect=2, interpolation='None', cmap=cmap, vmax=filter[-1], vmin=0.9, extent=extent)
    ax1.set_title('Winning pattern')

    # Colorbar
    bounds = np.arange(0, manager.nn.minicolumns + 1, 0.5)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
    cb = fig.colorbar(im1, cax=cbar_ax, boundaries=bounds)


def plot_network_activity_angle(manager, recall=True, cmap=None, ax=None, title=True, time_y=True):
    if recall:
        T_total = manager.T_recall_total
    else:
        T_total = manager.T_training_total

    history = manager.history
    # Get the angles
    angles = calculate_angle_from_history(manager)
    patterns_dic = manager.patterns_dic
    n_patters = len(patterns_dic)
    # Plot
    sns.set_style("whitegrid", {'axes.grid': False})

    if cmap is None:
        cmap = 'plasma'

    if title:
        title1 = 'Unit activation'
        title2 = 'Angles with stored'
    else:
        title1 = ''
        title2 = ''

    if ax is None:
        fig = plt.figure(figsize=(16, 12))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        ax1, ax2 = ax
        fig = ax1.figure

    if time_y:
        extent1 = [0, manager.nn.minicolumns * manager.nn.hypercolumns, T_total, 0]
        extent2 = [0, n_patters, T_total, 0]

        im1 = ax1.imshow(history['o'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent1)
        ax1.set_title(title1)

        ax1.set_xlabel('Units')
        ax1.set_ylabel('Time (s)')

        im2 = ax2.imshow(angles, aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent2)
        ax2.set_title(title2)
        ax2.set_xlabel('Patterns')

    else:
        extent1 = [0, T_total, 0, manager.nn.minicolumns * manager.nn.hypercolumns]
        extent2 = [0, T_total, 0, n_patters]

        im1 = ax1.imshow(history['o'].T, aspect='auto', origin='lower', cmap=cmap, vmax=1, vmin=0, extent=extent1)
        ax1.set_title(title1)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Units')

        im2 = ax2.imshow(angles.T, aspect='auto', origin='lower', cmap=cmap, vmax=1, vmin=0, extent=extent2)
        ax2.set_title(title2)
        ax2.set_ylabel('Patterns')
        ax2.set_xlabel('Time (s)')

    if ax is None:
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
        fig.colorbar(im1, cax=cbar_ax)



    return [ax1, ax2]


def plot_network_activity(manager):

    T_total = manager.T_total

    history = manager.history
    sns.set_style("whitegrid", {'axes.grid': False})

    cmap = 'plasma'
    extent = [0, manager.nn.minicolumns * manager.nn.hypercolumns, T_total, 0]

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(history['o'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)
    ax1.set_title('Unit activation')

    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(history['z_pre'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)
    ax2.set_title('Traces of activity (z)')

    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(history['a'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)
    ax3.set_title('Adaptation')

    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(history['p_pre'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)
    ax4.set_title('Probability')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
    fig.colorbar(im1, cax=cbar_ax)
