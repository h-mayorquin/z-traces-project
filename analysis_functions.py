import numpy as np
import itertools


def deterministic_solution(time, beta, w, tau_s, tau_a, g_a, s0, a0, unit_active):
    """

    :param time: the time of the solution
    :param beta: the bias
    :param w: the weight that its receiving
    :param tau_s: time constant of the unit
    :param tau_a: adaptation time constatn
    :param g_a: adaptation gain
    :param s0: initial value for the synaptic current
    :param a0: initial value of the adaptation curent
    :param unit_active: whether the unit is active or not.
    :return:
    """
    fixed_point = beta + w
    charge = a0
    r = tau_s / tau_a
    f = 1.0 / (1 - r)

    if unit_active:
        fixed_point -= g_a
        charge -= 1.0

    slow_component = g_a * f * charge * np.exp(-time / tau_a)
    fast_component = (s0 - fixed_point + g_a * charge * f) * np.exp(-time / tau_s)
    s = fixed_point - slow_component + fast_component

    return s


def calculate_persistence_time(tau_a, w_diff, beta_diff, g_a, tau_s, perfect=False):
    """
    Formula for approximating the persistence time, the assumption for this is
    that the persistent time is >> than tau_s
    :param tau_a: the time constant of the adaptation
    :param w_diff: the difference between the weighs
    :param b_diff: the difference in the bias
    :param g_a: the adaptation current gain
    :param tau_s: the time constant of the unit
    :param perfect: whether the unit is a perfect integrator (capacitor)
    :return:
    """

    B = (w_diff + beta_diff)/ g_a
    T = tau_a * np.log(1 / (1 - B))
    if not perfect:
        r = tau_s / tau_a
        T += tau_a * np.log(1 / (1 - r))
    return T


def calculate_recall_quantities(manager, nr, T_recall, T_cue, remove=0.009, reset=True, empty_history=True, NMDA=False):
    n_seq = nr.shape[0]
    I_cue = nr[0]

    # Do the recall
    manager.run_network_recall(T_recall=T_recall, I_cue=I_cue, T_cue=T_cue,
                               reset=reset, empty_history=empty_history, NMDA=NMDA)

    distances = calculate_angle_from_history(manager)
    winning = calculate_winning_pattern_from_distances(distances)
    timings = calculate_patterns_timings(winning, manager.dt, remove=remove)

    # Get the element of the sequence without consecutive duplicates
    aux = [x[0] for x in timings]
    pattern_sequence = [i for i, x in itertools.groupby(aux)]

    # Assume successful until proven otherwise
    success = 1.0
    for index, pattern_index in enumerate(pattern_sequence[:n_seq]):
        pattern = manager.patterns_dic[pattern_index]
        goal_pattern = nr[index]
        # Compare arrays of the recalled pattern with the goal
        if not np.array_equal(pattern, goal_pattern):
            success = 0.0
            break

    if len(pattern_sequence) < n_seq:
        success = 0.0
    persistent_times = [x[1] for x in timings]
    return success, pattern_sequence, persistent_times, timings


def calculate_angle_from_history(manager):
    """
    :param manager: A manager of neural networks, it is used to obtain the history of the activity and
     the patterns that were stored

    :return: A vector with the distances to the stored patterns at every point in time.
    """
    if manager.patterns_dic is None:
        raise ValueError('You have to run a protocol before or provide a patterns dic')

    history = manager.history
    patterns_dic = manager.patterns_dic
    stored_pattern_indexes = np.array(list(patterns_dic.keys()))
    num_patterns = max(stored_pattern_indexes) + 1

    o = history['o'][1:]
    if o.shape[0] == 0:
        raise ValueError('You did not record the history of unit activities o')

    distances = np.zeros((o.shape[0], num_patterns))

    for index, state in enumerate(o):
        # Obtain the dot product between the state of the network at each point in time and each pattern
        nominator = [np.dot(state, patterns_dic[pattern_index]) for pattern_index in stored_pattern_indexes]
        # Obtain the norm of both the state and the patterns to normalize
        denominator = [np.linalg.norm(state) * np.linalg.norm(patterns_dic[pattern_index])
                       for pattern_index in stored_pattern_indexes]

        # Get the angles and store them
        dis = [a / b for (a, b) in zip(nominator, denominator)]
        distances[index, stored_pattern_indexes] = dis

    return distances


def calculate_winning_pattern_from_distances(distances):
    # Returns the number of the winning pattern
    return np.argmax(distances, axis=1)


def calculate_patterns_timings(winning_patterns, dt, remove=0):
    """

    :param winning_patterns: A vector with the winning pattern for each point in time
    :param dt: the amount that the time moves at each step
    :param remove: only add the patterns if they are bigger than this number, used a small number to remove fluctuations

    :return: pattern_timins, a vector with information about the winning pattern, how long the network stayed at that
     configuration, when it got there, etc
    """

    # First we calculate where the change of pattern occurs
    change = np.diff(winning_patterns)
    indexes = np.where(change != 0)[0]

    # Add the end of the sequence
    indexes = np.append(indexes, winning_patterns.size - 1)

    patterns = winning_patterns[indexes]
    patterns_timings = []

    previous = 0
    for pattern, index in zip(patterns, indexes):
        time = (index - previous + 1) * dt  # The one is because of the shift with np.change
        if time >= remove:
            patterns_timings.append((pattern, time, previous*dt, index * dt))
        previous = index

    return patterns_timings


def calculate_probability_theo2(Tp, Tstart, Ttotal, tau_z):
    """
    Calculate the probability of the unit being activated.
    :param tau_z: the time constant of the uncertainty or z-filters
    :param Tp: The training time, the time that the unit was activated
    :param Tstart: The time at which the unit was activated
    :param Ttotal: The total time of observation
    :return: the probability of the unit being active
    """
    p = Tp - tau_z * np.exp((Tstart - Ttotal) / tau_z) * (np.exp(Tp / tau_z) - 1)

    return p / Ttotal


def calculate_probability_theo(Tp, Tstart, Ttotal, tau_z):
    """
    Calculate the probability of the unit being activated.
    :param tau_z: the time constant of the uncertainty or z-filters
    :param Tp: The training time, the time that the unit was activated
    :param Tstart: The time at which the unit was activated
    :param Ttotal: The total time of observation
    :return: the probability of the unit being active
    """
    M = 1 - np.exp(-Tp / tau_z)
    #p = Tp + tau_z * M * (2  - np.exp(-(Ttotal - Tp)/tau_z))

    p = Tp - tau_z * M + tau_z * M * (1 - np.exp(-(Ttotal - Tp) / tau_z))
    return p / Ttotal


def calculate_joint_probabilities_theo(T1, Ts, T2, Tt, tau1, tau2):
    """
    Calcualtes the joint probability of unit 1 and 2.
    :param T1:The time that the unit 1 remained activated (training time)
    :param Ts: The time at which the second unit becomes activated
    :param T2: The time the unit 2 remained activated (training time)
    :param Tt: The total time of observation
    :param tau1: the time constant of the z-filter of pre-synaptic unit (z-filter)
    :param tau2: the time constant of the z-filter of post-synaptic unit (z-filter)
    :return: the joint probability.
    """
    tau_p = tau1 * tau2 / (tau1 + tau2)
    M1 = 1 - np.exp(-T1 / tau1)
    M2 = 1 - np.exp(-T2 / tau2)

    aux1 = M1 * tau1 * (np.exp(-(Ts - T1) / tau1) - np.exp(-(Ts + T2 - T1) / tau1))

    A1arg = T1 / tau1 + Ts / tau2 - (Ts + T2) / tau_p
    A1 = np.exp(A1arg)
    A2arg = T1 / tau1 + Ts / tau2 - Ts / tau_p
    A2 = np.exp(A2arg)
    aux2 = M1 * tau_p * (A1 - A2)

    B1arg = T1 / tau1 + (Ts + T2) / tau2 - Tt / tau_p
    B1 = np.exp(B1arg)
    B2arg = T1 / tau1 + (Ts + T2) / tau2 - (Ts + T2) / tau_p
    B2 = np.exp(B2arg)

    aux3 = M1 * M2 * tau_p * (B1 - B2)

    P = (aux1 + aux2 - aux3) / Tt

    return P


def calculate_self_probability_theo(T1, Tt, tau1, tau2):
    """
    The joint probability of unit with itself with different uncertainty for pre-unit and
    post-unit
    :param T1: the time the unit remained activated (training time)
    :param Tt: total time of observation
    :param tau1: the pre-syanptic time constant.
    :param tau2: the post-synaptic time constant.
    :return:
    """
    tau_p = tau1 * tau2 / (tau1 + tau2)

    m1 = 1 - np.exp(-T1 / tau1)
    m2 = 1 - np.exp(-T1 / tau2)
    mp = 1 - np.exp(-T1 / tau_p)

    aux1 = T1 - tau1 * m1 - tau2 * m2 + tau_p * mp
    aux2 = tau_p * m1 * m2 * (1 - np.exp(-(Tt - T1) / tau_p))

    P_self = aux1 + aux2

    return P_self / Tt


def calculate_get_weights_theo(T1, T2, Tt, tau_pre, tau_post, Tr=None, IPI=None):
    Tstart = 0.0
    if Tr is None:
        Tr = T2
    if IPI is None:
        IPI = 0.0

    # Calculate the self weight
    pi = calculate_probability_theo(T1, Tstart, Tt, tau_pre)
    pii = calculate_self_probability_theo(T1, Tt, tau_pre, tau_post)
    w_self = np.log10(pii / (pi * pi))

    # Calculate the next weight
    Ts = T1 + IPI
    pij = calculate_joint_probabilities_theo(T1, Ts, T2, Tt, tau_pre, tau_post)
    pj = calculate_probability_theo(T2, Tstart, Tt, tau_post)
    w_next = np.log10(pij / (pi * pj))

    # Calculate the rest weight
    pk = calculate_probability_theo(Tr, Tstart, Tt, tau_post)
    Ts = T1 + IPI + T2 + IPI
    pik = calculate_joint_probabilities_theo(T1, Ts, Tr, Tt, tau_pre, tau_post)
    w_rest = np.log10(pik / (pi * pk))

    # Calculate the back weight
    Ts = T1 + IPI
    pji = calculate_joint_probabilities_theo(T1, Ts, T2, Tt, tau_post, tau_pre)
    w_back = np.log10(pji / (pi * pj))

    return w_self, w_next, w_rest, w_back


def calculate_triad_connectivity(tt1, tt2, tt3, ipi1, ipi2, tau_z_pre, tau_z_post,
                                 base_time, base_ipi, resting_time, n_patterns):
    """
    This function gives you the connectivity among a triad, assuming that all the other temporal structure outside of
    the trial is homogeneus
    :param tt1:
    :param tt2:
    :param tt3:
    :param ipi1:
    :param ipi2:
    :param tau_z_pre:
    :param tau_z_post:
    :param base_time:
    :param base_ipi:
    :param resting_time:
    :param n_patterns:
    :return:
    """

    Tt = (n_patterns - 3) * base_time + tt1 + tt2 + tt3 + ipi1 + ipi2 + \
         (n_patterns - 2) * base_ipi + resting_time

    # Single probabilities
    p1_pre = calculate_probability_theo(Tp=tt1, Tstart=0.0, Ttotal=Tt, tau_z=tau_z_pre)
    p2_pre = calculate_probability_theo(Tp=tt2, Tstart=0.0, Ttotal=Tt, tau_z=tau_z_pre)
    p3_pre = calculate_probability_theo(Tp=tt3, Tstart=0.0, Ttotal=Tt, tau_z=tau_z_pre)

    p1_post = calculate_probability_theo(Tp=tt1, Tstart=0.0, Ttotal=Tt, tau_z=tau_z_post)
    p2_post = calculate_probability_theo(Tp=tt2, Tstart=0.0, Ttotal=Tt, tau_z=tau_z_post)
    p3_post = calculate_probability_theo(Tp=tt3, Tstart=0.0, Ttotal=Tt, tau_z=tau_z_post)

    # joint-self probabilities
    p11 = calculate_self_probability_theo(T1=tt1, Tt=Tt, tau1=tau_z_pre, tau2=tau_z_post)
    p22 = calculate_self_probability_theo(T1=tt2, Tt=Tt, tau1=tau_z_pre, tau2=tau_z_post)
    p33 = calculate_self_probability_theo(T1=tt3, Tt=Tt, tau1=tau_z_pre, tau2=tau_z_post)

    # Joint probabilities
    Ts = tt1 + ipi1
    p21 = calculate_joint_probabilities_theo(T1=tt1, Ts=Ts, T2=tt2, Tt=Tt, tau1=tau_z_pre, tau2=tau_z_post)
    Ts = tt1 + ipi1 + tt2 + ipi2
    p31 = calculate_joint_probabilities_theo(T1=tt1, Ts=Ts, T2=tt3, Tt=Tt, tau1=tau_z_pre, tau2=tau_z_post)
    Ts = tt1 + ipi1
    p12 = calculate_joint_probabilities_theo(T1=tt1, Ts=Ts, T2=tt2, Tt=Tt, tau1=tau_z_post, tau2=tau_z_pre)
    Ts = tt2 + ipi2
    p32 = calculate_joint_probabilities_theo(T1=tt2, Ts=Ts, T2=tt3, Tt=Tt, tau1=tau_z_pre, tau2=tau_z_post)
    Ts = tt1 + ipi1 + tt2 + ipi2
    p13 = calculate_joint_probabilities_theo(T1=tt1, Ts=Ts, T2=tt3, Tt=Tt, tau1=tau_z_post, tau2=tau_z_pre)
    Ts = tt2 + ipi2
    p23 = calculate_joint_probabilities_theo(T1=tt2, Ts=Ts, T2=tt3, Tt=Tt, tau1=tau_z_post, tau2=tau_z_pre)

    # Weights
    w11 = np.log10(p11 / (p1_pre * p1_post))
    w12 = np.log10(p12 / (p1_pre * p2_post))
    w13 = np.log10(p13 / (p1_pre * p3_post))
    w21 = np.log10(p21 / (p2_pre * p1_post))
    w22 = np.log10(p22 / (p2_pre * p2_post))
    w23 = np.log10(p23 / (p2_pre * p3_post))
    w31 = np.log10(p31 / (p3_pre * p1_post))
    w32 = np.log10(p32 / (p3_pre * p2_post))
    w33 = np.log10(p33 / (p3_pre * p3_post))

    # Betas
    beta1 = np.log10(p1_post)
    beta2 = np.log10(p2_post)
    beta3 = np.log10(p3_post)

    # Bs (un-normalized)
    B12 = w11 - w12 + beta2 - beta1
    B13 = w11 - w31 + beta3 - beta1
    B21 = w11 - w21 + beta1 - beta2
    B23 = w33 - w32 + beta3 - beta2
    B31 = w11 - w31 + beta1 - beta3
    B32 = w22 - w32 + beta2 - beta3

    return locals()
