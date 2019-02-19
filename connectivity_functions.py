import numpy as np
import IPython
import numbers


def log_epsilon(x, epsilon):

    return np.log10(np.maximum(x, epsilon))


def get_w_pre_post(P, p_pre, p_post, epsilon=1e-20, diagonal_zero=True):

    outer = np.outer(p_post, p_pre)
    # w = np.log(p * P) - np.log(outer)
    # x = P / outer
    # w = np.log(x)
    # P_qual zero and outer is bigger than epsilon
    P_equal_zero = (P < epsilon) * (outer > epsilon)
    w = log_epsilon(P, epsilon) - log_epsilon(outer, epsilon)
    w[P_equal_zero] = np.log10(epsilon)

    if diagonal_zero:
        w[np.diag_indices_from(w)] = 0

    return w


def get_beta(p, epsilon=1e-10):

    probability = np.copy(p)
    probability[p < epsilon] = epsilon

    beta = np.log10(probability)

    return beta


def softmax(input_vector, G=1.0, minicolumns=2):
    """Calculate the softmax of a list of numbers w.

    Parameters
    ----------
    input_vector : the vector to softmax
    G : the constant for softmax, the bigger the G the more of a max it is

    Return
    ------
    a list of the same length as input_vectorof non-negative numbers

    Examples
    --------
    """

    # The lower bounds keeps the overflow from happening
    lower_bound = -600
    upper_bound = 600

    x = np.copy(input_vector)
    x_size = x.size
    x = np.reshape(x, (x_size // minicolumns, minicolumns))
    x = G * np.array(x)

    x[x < lower_bound] = lower_bound
    x[x > upper_bound] = upper_bound

    e = np.exp(x)
    dist = normalize_array(e)

    dist = np.reshape(dist, x_size)

    return dist


def normalize_array(array):
    """
    "Normalize an array over the second axis"

    :param array: the array to normalize
    :return: the normalized array
    """

    return array / np.sum(array, axis=1)[:, np.newaxis]


def strict_max(x, minicolumns):
    """
    A strict max that returns an array with 1 where the maximum of every minicolumn is
    :param x: the array
    :param minicolumns: number of minicolumns
    :return: the stric_max of the array
    """

    x = np.reshape(x, (x.size // minicolumns, minicolumns))
    z = np.zeros_like(x)
    maxes = np.argmax(x, axis=1)
    for max_index, max_aux in enumerate(maxes):
        z[max_index, max_aux] = 1

    return z.reshape(x.size)


def fill_connection(w, state_from, state_to, minicolumns, value):
    for hypercolumn_from, minicolumn_from in enumerate(state_from):
        for hypercolum_to, minicolumn_to in enumerate(state_to):
            index_from = hypercolumn_from * minicolumns + minicolumn_from
            index_to = hypercolum_to * minicolumns + minicolumn_to
            w[index_to, index_from] = value


def create_weight_matrix(minicolumns, sequence, ws, wn, wb, alpha, alpha_back=None,
                         extension=2, w=None):
    if isinstance(ws, numbers.Number):
        ws = np.full(minicolumns, ws)
    if isinstance(wn, numbers.Number):
        wn = np.full(minicolumns, wn)
    if isinstance(wb, numbers.Number):
        wb = np.full(minicolumns, wb)

    if alpha_back is None:
        alpha_back = alpha

    hypercolumns = len(sequence[0])
    if w is None:
        w_min = min(wb) - extension * alpha - 1.0
        w = np.ones((minicolumns * hypercolumns, minicolumns * hypercolumns)) * w_min

    n_states = len(sequence)

    # Let's fill the self-connections
    for state_index, value in enumerate(ws):
        state_from = sequence[state_index]
        state_to = sequence[state_index]
        fill_connection(w, state_from, state_to, minicolumns, value)

    # Let's fill backwards
    for state_index, value in enumerate(wb):
        state_from = sequence[state_index]
        effective_extension = min(extension + 1, state_index)
        for last_index in range(effective_extension):
            effective_value = value - last_index * alpha_back
            state_to = sequence[state_index - 1 - last_index]
            fill_connection(w, state_from, state_to, minicolumns, effective_value)

    # Let's fill forwards
    for state_index, value in enumerate(wn):
        state_from = sequence[state_index]
        effective_extension = min(extension + 1, n_states - 1 - state_index)
        for next_index in range(effective_extension):
            effective_value = value - next_index * alpha
            state_to = sequence[state_index + 1 + next_index]
            fill_connection(w, state_from, state_to, minicolumns, effective_value)

    return w