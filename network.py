import random
import numpy as np
import scipy as sp
from connectivity_functions import softmax, get_w_pre_post, get_beta, strict_max
from connectivity_functions import create_weight_matrix
from patterns_representation import create_canonical_activity_representation, build_network_representation
import IPython
import numbers


epoch_end_string = 'epoch_end'


class Network:
    def __init__(self, hypercolumns, minicolumns, G=1.0, tau_s=0.010, tau_z_pre=0.050, tau_z_post=0.005,
                 tau_a=0.250, g_a=1.0, g_I=10.0, sigma_out=0.0, epsilon=1e-60, g_beta=1.0, prng=np.random,
                 strict_maximum=True, perfect=False, normalized_currents=True):

        # Random number generator
        self.prng = prng
        self.sigma_out = sigma_out  # The variance that the system would have on the steady state if were to have it
        self.sigma_in = sigma_out * np.sqrt(2 / tau_s)  # Ornstein-Uhlenbeck process
        self.epsilon = epsilon

        # Network parameters
        self.hypercolumns = hypercolumns
        self.minicolumns = minicolumns

        self.n_units = self.hypercolumns * self.minicolumns

        # Network variables
        self.strict_maximum = strict_maximum
        self.perfect = perfect
        self.normalized_current = normalized_currents
        if self.normalized_current:
            self.normalized_constant = self.hypercolumns
        else:
            self.normalized_constant = 1.0

        # Dynamic Parameters
        self.tau_s = tau_s
        self.tau_a = tau_a
        self.r = self.tau_s / self.tau_a
        self.g_beta = g_beta
        self.g_a = g_a
        self.g_I = g_I
        self.tau_z_pre = tau_z_pre
        self.tau_z_post = tau_z_post
        self.G = G

        # State variables
        self.o = np.full(shape=self.n_units, fill_value=0.0)
        self.s = np.full(shape=self.n_units, fill_value=0.0)
        self.a = np.full(shape=self.n_units, fill_value=0.0)
        self.I = np.full(shape=self.n_units, fill_value=0.0)

        # Current values
        self.i = np.full(shape=self.n_units, fill_value=0.0)
        self.z_pre = np.full(shape=self.n_units, fill_value=0.0)
        self.z_post = np.full(shape=self.n_units, fill_value=0.0)
        self.z_co = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

        # Keeping track of the probability / connectivity
        self.t_p = 0.0
        self.p_pre = np.full(shape=self.n_units, fill_value=0.0)
        self.p_post = np.full(shape=self.n_units, fill_value=0.0)
        self.P = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)
        self.w = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)
        self.beta = np.full(shape=self.n_units, fill_value=0.0)

    def parameters(self):
        """
        Get the parameters of the model

        :return: a dictionary with the parameters
        """
        parameters = {'tau_s': self.tau_s, 'tau_z_post': self.tau_z_post, 'tau_z_pre': self.tau_z_pre,
                      'tau_a': self.tau_a, 'g_a': self.g_a, 'g_I': self.g_I, 'epsilon': self.epsilon,
                      'G': self.G, 'sigma_out': self.sigma_out, 'sigma_in': self.sigma_in,
                      'perfect': self.perfect, 'strict_maximum': self.strict_maximum}

        return parameters

    def reset_values(self, keep_connectivity=True):
        # State variables
        self.o = np.full(shape=self.n_units, fill_value=0.0)
        self.s = np.full(shape=self.n_units, fill_value=0.0)

        self.a = np.full(shape=self.n_units, fill_value=0.0)
        self.I = np.full(shape=self.n_units, fill_value=0.0)

        # Current values
        self.i = np.full(shape=self.n_units, fill_value=0.0)
        self.z_pre = np.full(shape=self.n_units, fill_value=0.0)
        self.z_post = np.full(shape=self.n_units, fill_value=0.0)
        self.z_co = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

        if not keep_connectivity:
            self.beta = np.full(shape=self.n_units, fill_value=0.0)
            self.w = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

            self.p_pre = np.full(shape=self.n_units, fill_vale=0.0)
            self.p_post = np.full(shape=self.n_units, fill_value=0.0)
            self.P = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

    def update_continuous(self, dt=1.0, sigma=None):
        # Get the noise
        if sigma is None:
            noise = self.sigma_in * np.sqrt(dt) * self.prng.normal(0, 1.0, self.n_units)
        else:
            noise = sigma

        # Calculate currents
        self.i = self.w @ self.z_pre / self.normalized_constant
        if self.perfect:
            self.s = self.i + self.g_beta * self.beta - self.g_a * self.a + self.g_I * self.I + noise
        else:
            self.s += (dt / self.tau_s) * (self.i  # Current
                                           + self.g_beta * self.beta  # Bias
                                           + self.g_I * self.I  # Input current
                                           - self.g_a * self.a  # Adaptation
                                           - self.s)  # s follow all of the s above
            self.s += noise
        # Non-linearity
        if self.strict_maximum:
            self.o = strict_max(self.s, minicolumns=self.minicolumns)
        else:
            self.o = softmax(self.s, G=self.G, minicolumns=self.minicolumns)

        # Update the z-traces
        self.z_pre += (dt / self.tau_z_pre) * (self.o - self.z_pre)
        self.z_post += (dt / self.tau_z_post) * (self.o - self.z_post)
        self.z_co = np.outer(self.z_post, self.z_pre)

        # Update the adaptation
        self.a += (dt / self.tau_a) * (self.o - self.a)

    def update_probabilities(self, dt):
        if self.t_p > 0.0:
            time_factor = dt / self.t_p
            self.p_pre += time_factor * (self.z_pre - self.p_pre)
            self.p_post += time_factor * (self.z_post - self.p_post)
            self.P += time_factor * (self.z_co - self.P)
        self.t_p += dt

    def update_weights(self):
        # Update the connectivity
        self.beta = get_beta(self.p_post, self.epsilon)
        self.w = get_w_pre_post(self.P, self.p_pre, self.p_post, self.epsilon, diagonal_zero=False)


class NetworkNMDA:
    def __init__(self, hypercolumns, minicolumns, G=1.0, tau_s=0.010, tau_z_pre=0.050, tau_z_post=0.005,
                 tau_a=0.250, g_a=1.0, g_I=10.0, g_AMPA=1.0, g_NMDA=1.0, sigma_out=0.0,
                 epsilon=1e-60, g_beta=1.0, prng=np.random, strict_maximum=True, perfect=False, normalized_currents=True):

        # Random number generator
        self.prng = prng
        self.sigma_out = sigma_out  # The variance that the system would have on the steady state if were to have it
        self.sigma_in = sigma_out * np.sqrt(2 / tau_s)  # Ornstein-Uhlenbeck process
        self.epsilon = epsilon

        # Network parameters
        self.hypercolumns = hypercolumns
        self.minicolumns = minicolumns

        self.n_units = self.hypercolumns * self.minicolumns

        # Network variables
        self.strict_maximum = strict_maximum
        self.perfect = perfect
        self.normalized_current = normalized_currents
        if self.normalized_current:
            self.normalized_constant = self.hypercolumns
        else:
            self.normalized_constant = 1.0

        # Dynamic Parameters
        self.tau_s = tau_s
        self.tau_a = tau_a
        self.r = self.tau_s / self.tau_a
        self.g_beta = g_beta
        self.g_AMPA = g_AMPA
        self.g_NMDA = g_NMDA
        self.g_a = g_a
        self.g_I = g_I
        self.tau_z_pre = tau_z_pre
        self.tau_z_post = tau_z_post
        self.tau_z_pre_ampa = 0.005
        self.tau_z_post_ampa = 0.005
        self.G = G

        # State variables
        self.o = np.full(shape=self.n_units, fill_value=0.0)
        self.s = np.full(shape=self.n_units, fill_value=0.0)
        self.a = np.full(shape=self.n_units, fill_value=0.0)
        self.I = np.full(shape=self.n_units, fill_value=0.0)

        # Current values
        self.i = np.full(shape=self.n_units, fill_value=0.0)
        self.z_pre = np.full(shape=self.n_units, fill_value=0.0)
        self.z_post = np.full(shape=self.n_units, fill_value=0.0)
        self.z_co = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

        # Current values AMPA
        self.i_ampa = np.full(shape=self.n_units, fill_value=0.0)
        self.z_ampa_pre = np.full(shape=self.n_units, fill_value=0.0)
        self.z_post_ampa = np.full(shape=self.n_units, fill_value=0.0)
        self.z_co_ampa = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

        # Keeping track of the probability / connectivity
        self.t_p = 0.0
        self.p_pre = np.full(shape=self.n_units, fill_value=0.0)
        self.p_post = np.full(shape=self.n_units, fill_value=0.0)
        self.P = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)
        self.w = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

        self.p_pre_ampa = np.full(shape=self.n_units, fill_value=0.0)
        self.p_post_ampa = np.full(shape=self.n_units, fill_value=0.0)
        self.P_ampa = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)
        self.w_ampa = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

        self.beta = np.full(shape=self.n_units, fill_value=0.0)

    def parameters(self):
        """
        Get the parameters of the model

        :return: a dictionary with the parameters
        """
        parameters = {'tau_s': self.tau_s, 'tau_z_post': self.tau_z_post, 'tau_z_pre': self.tau_z_pre,
                      'tau_a': self.tau_a, 'g_a': self.g_a, 'g_I': self.g_I, 'epsilon': self.epsilon,
                      'G': self.G, 'sigma_out': self.sigma_out, 'sigma_in': self.sigma_in,
                      'perfect': self.perfect, 'strict_maximum': self.strict_maximum}

        return parameters

    def reset_values(self, keep_connectivity=True):
        # State variables
        self.o = np.full(shape=self.n_units, fill_value=0.0)
        self.s = np.full(shape=self.n_units, fill_value=0.0)

        self.a = np.full(shape=self.n_units, fill_value=0.0)
        self.I = np.full(shape=self.n_units, fill_value=0.0)

        # Current values
        self.i = np.full(shape=self.n_units, fill_value=0.0)
        self.z_pre = np.full(shape=self.n_units, fill_value=0.0)
        self.z_post = np.full(shape=self.n_units, fill_value=0.0)
        self.z_co = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

        self.i_ampa = np.full(shape=self.n_units, fill_value=0.0)
        self.z_pre_ampa = np.full(shape=self.n_units, fill_value=0.0)
        self.z_post_ampa = np.full(shape=self.n_units, fill_value=0.0)
        self.z_co_ampa = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)


        if not keep_connectivity:
            self.beta = np.full(shape=self.n_units, fill_value=0.0)
            self.w = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)
            self.w_ampa = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

            self.p_pre = np.full(shape=self.n_units, fill_vale=0.0)
            self.p_post = np.full(shape=self.n_units, fill_value=0.0)
            self.P = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

            self.p_pre_ampa = np.full(shape=self.n_units, fill_vale=0.0)
            self.p_post_ampa = np.full(shape=self.n_units, fill_value=0.0)
            self.P_ampa = np.full(shape=(self.n_units, self.n_units), fill_value=0.0)

    def update_continuous(self, dt=1.0, sigma=None):
        # Get the noise
        if sigma is None:
            noise = self.sigma_in * np.sqrt(dt) * self.prng.normal(0, 1.0, self.n_units)
        else:
            noise = sigma

        # Calculate currents
        self.i = self.w @ self.z_pre / self.normalized_constant
        self.i_ampa = self.w @ self.z_pre_ampa / self.normalized_constant

        if self.perfect:
            self.s = self.i + self.i_ampa + self.g_beta * self.beta - self.g_a * self.a + self.g_I * self.I + noise
        else:
            self.s += (dt / self.tau_s) * (self.g_AMPA * self.i_ampa + self.g_NMDA * self.i # Current
                                           + self.g_beta * self.beta  # Bias
                                           + self.g_I * self.I  # Input current
                                           - self.g_a * self.a  # Adaptation
                                           - self.s)  # s follow all of the s above
            self.s += noise
        # Non-linearity
        if self.strict_maximum:
            self.o = strict_max(self.s, minicolumns=self.minicolumns)
        else:
            self.o = softmax(self.s, G=self.G, minicolumns=self.minicolumns)

        # Update the z-traces
        self.z_pre += (dt / self.tau_z_pre) * (self.o - self.z_pre)
        self.z_post += (dt / self.tau_z_post) * (self.o - self.z_post)
        self.z_co = np.outer(self.z_post, self.z_pre)

        self.z_pre_ampa += (dt / self.tau_z_pre_ampa) * (self.o - self.z_pre_ampa)
        self.z_post_ampa += (dt / self.tau_z_post_ampa) * (self.o - self.z_post_ampa)
        self.z_co_ampa = np.outer(self.z_post_ampa, self.z_pre_ampa)

        # Update the adaptation
        self.a += (dt / self.tau_a) * (self.o - self.a)

    def update_probabilities(self, dt):
        if self.t_p > 0.0:
            time_factor = dt / self.t_p
            self.p_pre += time_factor * (self.z_pre - self.p_pre)
            self.p_post += time_factor * (self.z_post - self.p_post)
            self.P += time_factor * (self.z_co - self.P)

            self.p_pre_ampa += time_factor * (self.z_pre_ampa - self.p_pre_ampa)
            self.p_post_ampa += time_factor * (self.z_post_ampa - self.p_post_ampa)
            self.P_ampa += time_factor * (self.z_co_ampa - self.P_ampa)

        self.t_p += dt

    def update_weights(self):
        # Update the connectivity
        self.beta = get_beta(self.p_post, self.epsilon)
        self.w = get_w_pre_post(self.P, self.p_pre, self.p_post, self.epsilon, diagonal_zero=False)
        self.w_ampa = get_w_pre_post(self.P_ampa, self.p_pre_ampa, self.p_post_ampa, self.epsilon, diagonal_zero=False)


class NetworkManager:
    """
    This class will run the Network. Everything from running, saving and calculating quantities should be
    methods in this class.  In short this will do the running of the network, the learning protocols, etcera.

    Note that data analysis should be conducted into another class preferably.
    """

    def __init__(self, nn=None, dt=0.001, values_to_save=[]):
        """
        :param nn: A network instance
        :param time: A numpy array with the time to run
        :param values_to_save: a list with the values as strings of the state variables that should be saved
        """

        self.nn = nn

        # Timing variables
        self.dt = dt
        self.T_training_total = 0.0
        self.T_recall_total = 0.0
        self.n_time_total = 0
        self.time = None

        # Initialize saving dictionary
        self.saving_dictionary = self.get_saving_dictionary(values_to_save)

        # Initialize the history dictionary for saving values
        self.history = None
        self.empty_history()

        # Get reference representations
        self.canonical_activity_representation = create_canonical_activity_representation(self.nn.minicolumns,
                                                                                          self.nn.hypercolumns)
        self.canonical_network_representation = build_network_representation(self.canonical_activity_representation,
                                                                             self.nn.minicolumns)
        # Dictionary to see what has been taught to the network
        # self.n_patterns = 0
        self.patterns_dic = None
        self.network_representation = np.array([]).reshape(0, self.nn.n_units)

        # Training matrices
        self.B = None
        self.T = None
        self.w_diff = np.zeros_like(self.nn.w)
        self.beta_diff = np.zeros_like(self.nn.w)

    def get_saving_dictionary(self, values_to_save):
        """
        This resets the saving dictionary and only activates the values in values_to_save
        """

        # Reinitialize the dictionary
        saving_dictionary = {'o': False, 's': False, 'a': False,
                             'z_pre': False, 'z_post': False, 'z_co': False,
                             'p_pre': False, 'p_post': False, 'P': False,
                             'i': False, 'w': False, 'beta': False}

        # Activate the values passed to the function
        for state_variable in values_to_save:
            saving_dictionary[state_variable] = True

        return saving_dictionary

    def empty_history(self):
        """
        A function to empty the history
        """
        empty_array = np.array([]).reshape(0, self.nn.n_units)
        empty_array_square = np.array([]).reshape(0, self.nn.n_units, self.nn.n_units)

        self.history = {'o': empty_array, 's': empty_array, 'a': empty_array,
                        'z_pre': empty_array, 'z_post': empty_array, 'z_co': empty_array_square,
                        'p_pre': empty_array, 'p_post': empty_array, 'P': empty_array_square,
                        'i': empty_array, 'w': empty_array_square, 'beta': empty_array}

    def append_history(self, history, saving_dictionary):
        """
        This function is used at every step of a process that is going to be saved. The items given by
        saving dictinoary will be appended to the elements of the history dictionary.

        :param history: is the dictionary with the saved values
        :param saving_dictionary:  a saving dictionary with keys as the parameters that should be saved
        and items as boolean indicating whether that parameters should be saved or not
        """

        # Dynamical variables
        if saving_dictionary['o']:
            history['o'].append(np.copy(self.nn.o))
        if saving_dictionary['s']:
            history['s'].append(np.copy(self.nn.s))
        if saving_dictionary['a']:
            history['a'].append(np.copy(self.nn.a))
        if saving_dictionary['i']:
            history['i'].append(np.copy(self.nn.i))
        if saving_dictionary['z_pre']:
            history['z_pre'].append(np.copy(self.nn.z_pre))
        if saving_dictionary['z_post']:
            history['z_post'].append(np.copy(self.nn.z_post))
        if saving_dictionary['z_co']:
            history['z_co'].append(np.copy(self.nn.z_co))
        if saving_dictionary['p_pre']:
            history['p_pre'].append(np.copy(self.nn.p_pre))
        if saving_dictionary['p_post']:
            history['p_post'].append(np.copy(self.nn.p_post))
        if saving_dictionary['P']:
            history['P'].append(np.copy(self.nn.P))
        if saving_dictionary['w']:
            history['w'].append(np.copy(self.nn.w))
        if saving_dictionary['beta']:
            history['beta'].append(np.copy(self.nn.beta))

    def update_patterns(self, nr):
        self.network_representation = np.concatenate((self.network_representation, nr))
        aux, indexes = np.unique(self.network_representation, axis=0, return_index=True)
        patterns_dic = {index: pattern for (index, pattern) in zip(indexes, aux)}
        self.patterns_dic = patterns_dic

    def run_network(self, time=None, I=None, train_network=False, plasticity_on=False):
        # Change the time if given

        if time is None or len(time) == 0:
            raise ValueError('Time should be given and be an array')

        # Load the clamping if available
        if I is None:
            self.nn.I = np.zeros_like(self.nn.o)
        elif isinstance(I, (float, int)):
            self.nn.I = self.patterns_dic[I]
        else:
            self.nn.I = I  # The pattern is the input
        # Create a vector of noise
        noise = self.nn.prng.normal(loc=0, scale=1, size=(time.size, self.nn.n_units))
        noise *= self.nn.sigma_in * np.sqrt(self.dt)

        # Initialize run history
        step_history = {}

        # Create a list for the values that are in the saving dictionary
        for quantity, boolean in self.saving_dictionary.items():
            if boolean:
                step_history[quantity] = []

        # Run the simulation and save the values
        for index_t, t in enumerate(time):
            # Append the history first
            self.append_history(step_history, self.saving_dictionary)
            # Update the system dynamic variables
            self.nn.update_continuous(dt=self.dt, sigma=noise[index_t, :])
            # Update the learning variables
            if train_network:
                self.nn.update_probabilities(dt=self.dt)
            # Update the weights
            if plasticity_on:
                self.nn.update_weights()

        # Concatenate with the past history and redefine dictionary
        for quantity, boolean in self.saving_dictionary.items():
            if boolean:
                self.history[quantity] = np.concatenate((self.history[quantity], step_history[quantity]))

        return self.history

    def run_network_protocol(self, protocol, plasticity_on=False, verbose=True,
                             values_to_save_epoch=None, reset=True, empty_history=True):

        if empty_history:
            self.empty_history()
            self.T_training_total = 0
        if reset:
            self.nn.reset_values(keep_connectivity=True)

        # Updated the stored patterns
        self.update_patterns(protocol.network_representation)

        # Unpack the protocol
        times = protocol.times_sequence
        patterns_sequence = protocol.patterns_sequence
        learning_constants = protocol.learning_constants_sequence  # The values of Kappa

        # Initialize dictionary for storage
        epoch_history = {}
        if values_to_save_epoch:
            saving_dictionary_epoch = self.get_saving_dictionary(values_to_save_epoch)
            # Create a list for the values that are in the saving dictionary
            for quantity, boolean in saving_dictionary_epoch.items():
                if boolean:
                    epoch_history[quantity] = []

        # Run the protocol
        epochs = 0
        start_time = 0.0
        n_aux = 0
        for time, pattern, k in zip(times, patterns_sequence, learning_constants):
            # End of the epoch
            if pattern == epoch_end_string:
                # Store the values at the end of the epoch
                if values_to_save_epoch:
                    self.append_history(epoch_history, saving_dictionary_epoch)

                if verbose:
                    print('epochs', epochs)
                    epochs += 1

            # Running step
            else:
                running_time = np.arange(start_time, start_time + time, self.dt)
                self.run_network(time=running_time, I=pattern, train_network=True, plasticity_on=plasticity_on)
                start_time += time
                n_aux += running_time.size

        # Get timings quantities
        self.T_training_total += start_time
        self.n_time_total += n_aux
        self.time = np.linspace(0, self.T_training_total, num=self.n_time_total)

        # Update weights
        if not plasticity_on:
            self.nn.update_weights()

        # Return the history if available
        if values_to_save_epoch:
            return epoch_history

    def run_network_protocol_offline(self, protocol, NMDA=True):
        # Build time input
        timed_input = TimedInput(protocol, self.dt)
        timed_input.build_timed_input()
        timed_input.build_filtered_input_pre(tau_z=self.nn.tau_z_pre)
        timed_input.build_filtered_input_post(tau_z=self.nn.tau_z_post)

        # Calculate probabilities
        self.nn.p_pre, self.nn.p_post, self.nn.P = timed_input.calculate_probabilities_from_time_signal()
        # Store the connectivity values
        self.nn.beta = get_beta(self.nn.p_post, self.nn.epsilon)
        self.nn.w = get_w_pre_post(self.nn.P, self.nn.p_pre, self.nn.p_post, self.nn.epsilon, diagonal_zero=False)

        if NMDA:
            timed_input_AMPA = TimedInput(protocol, self.dt)
            timed_input_AMPA.build_timed_input()
            timed_input_AMPA.build_filtered_input_pre(tau_z=self.nn.tau_z_pre_ampa)
            timed_input_AMPA.build_filtered_input_post(tau_z=self.nn.tau_z_post_ampa)
            # Calculate probabilities
            aux = timed_input_AMPA.calculate_probabilities_from_time_signal()
            self.nn.p_pre_AMPA, self.nn.p_post_AMPA, self.nn.P_AMPA = aux
            # Store the connectivity values
            self.nn.w_ampa = get_w_pre_post(self.nn.P_AMPA, self.nn.p_pre_AMPA, self.nn.p_post_AMPA,
                                            self.nn.epsilon, diagonal_zero=False)

        # Update the patterns
        self.update_patterns(protocol.network_representation)

        # Get timings quantities
        t_total, n_time_total, time = protocol.calculate_time_quantities(self.dt)
        self.T_training_total += t_total
        self.n_time_total += n_time_total
        self.time = np.linspace(0, self.T_training_total, num=self.n_time_total)

        return timed_input

    def run_artificial_protocol(self, ws=1.0, wn=0.25, wb=-3.0, alpha=0.5, alpha_back=None, cycle=False):
        """
        This creates an artificial matrix
        :return: w, the weight matrix that was created
        """
        minicolumns = self.nn.minicolumns
        extension = self.nn.minicolumns
        sequence = self.canonical_activity_representation
        if cycle:
            sequence = np.append(sequence, sequence[0]).reshape(self.nn.minicolumns + 1, self.nn.hypercolumns)

        w = create_weight_matrix(minicolumns, sequence, ws, wn, wb, alpha,
                                 alpha_back, extension, w=None)
        self.nn.w = w

        p = np.ones(self.nn.n_units) * (1.0 / len(sequence))
        self.nn.beta = get_beta(p, self.nn.epsilon)

        # Updated the patterns in the network
        nr = self.canonical_network_representation
        self.update_patterns(nr)

        return w

    def run_network_recall(self, T_recall=10.0, T_cue=0.0, I_cue=None, reset=True,
                           empty_history=True, plasticity_on=False, stable_start=True):
        """
        Run network free recall
        :param T_recall: The total time of recalling
        :param T_cue: the time that the cue is run
        :param I_cue: The current to give as the cue
        :param reset: Whether the state variables values should be returned
        :param empty_history: whether the history should be cleaned
        """
        if T_recall < 0.0:
            raise ValueError('T_recall = ' + str(T_recall) + ' has to be positive')
        time_recalling = np.arange(0, T_recall, self.dt)
        time_cue = np.arange(0, T_cue, self.dt)

        if plasticity_on:
            train_network = True
        else:
            train_network = False

        if empty_history:
            self.empty_history()
        if reset:
            # Never destroy connectivity while recalling
            self.nn.reset_values(keep_connectivity=True)
            # Recall times
            self.T_recall_total = 0
            self.n_time_total = 0

        if I_cue is None:
            pass
        elif isinstance(I_cue, int):
            self.nn.z_pre[I_cue] = 1.0
            self.nn.z_pre_ampa[I_cue] = 1.0
        else:
            self.nn.z_pre[np.where(I_cue)[0]] = 1.0
            self.nn.z_pre_ampa[np.where(I_cue)][0] = 1.0

        # Set initial conditions of the current to the clamping if available
        if stable_start:
            if I_cue is None:
                pass
            elif isinstance(I_cue, (float, int)):
                self.nn.s = self.nn.g_I * self.patterns_dic[I_cue].astype('float')
                self.nn.o = strict_max(self.nn.s, minicolumns=self.nn.minicolumns)
                self.nn.i = self.nn.w @ self.nn.o / self.nn.normalized_constant
                self.nn.s += self.nn.i + self.nn.beta - self.nn.g_a * self.nn.a
            else:
                self.nn.s = self.nn.g_I * I_cue  # The pattern is the input
                self.nn.o = strict_max(self.nn.s, minicolumns=self.nn.minicolumns)
                self.nn.i = self.nn.w @ self.nn.o / self.nn.normalized_constant
                self.nn.s += self.nn.i + self.nn.beta - self.nn.g_a * self.nn.a

        # Run the cue
        if T_cue > 0.001:
            self.run_network(time=time_cue, I=I_cue, train_network=train_network, plasticity_on=plasticity_on)

        # Run the recall
        self.run_network(time=time_recalling, train_network=train_network, plasticity_on=plasticity_on)

        # Calculate total time
        self.T_recall_total += T_recall + T_cue
        self.n_time_total += self.history['o'].shape[0]
        self.time = np.linspace(0, self.T_recall_total, num=self.n_time_total)

    def set_persistent_time_with_adaptation_gain(self, T_persistence, from_state=2, to_state=3, NMDA=False):
        """
        This formula adjusts the adpatation gain g_a so the network with the current weights lasts for T_persistence
        when passing from `from_state' to `to_state'
        :param T_persistence: The persistent time necessary
        :param from_state: the state tat will last T_persistent seconds activated
        :param to_state: the state that it will go to
        :return: g_a the new adaptation
        """

        delta_w = self.nn.w[from_state, from_state] - self.nn.w[to_state, from_state]
        if NMDA:
            delta_w += self.nn.w_ampa[from_state, from_state] - self.nn.w_ampa[to_state, from_state]

        delta_beta = self.nn.beta[from_state] - self.nn.beta[to_state]
        aux = 1 - np.exp(-T_persistence / self.nn.tau_a) / (1 - self.nn.r)
        g_a = (delta_w + delta_beta) / aux

        self.nn.g_a = g_a

        return g_a

    def calculate_persistence_time_matrix(self):

        self.w_diff = self.nn.w.diagonal() - self.nn.w
        self.beta_diff = (self.nn.beta[:, np.newaxis] - self.nn.beta[np.newaxis, :]).T

        self.B = (self.w_diff + self.beta_diff) / self.nn.g_a
        self.T = self.nn.tau_a * np.log(1 / (1 - self.B))
        if not self.nn.perfect:
            self.T += self.nn.tau_a * np.log(1 / (1 - self.nn.r))

        self.T[self.T < 0] = 0.0
        return self.T


class Protocol:
    def __init__(self):
        # Protocol input variables
        self.training_times = None
        self.inter_pulse_intervals = None
        self.inter_sequence_interval = None
        self.resting_time = None
        self.w = None

        # Protocol output variables
        self.times_sequence = None
        self.patterns_sequence = None
        self.learning_constants_sequence = None

        # Patterns to store variables
        self.n_patterns = None
        self.activity_representation = None
        self.network_representation = None

        # Time course related variables
        self.n_time_total = None
        self.dt = None
        self.T_protocol_total = None
        self.time = None

    def calculate_time_quantities(self, dt):
        self.dt = dt
        inter_sequence_interval_length = int(self.inter_sequence_interval / dt)
        resting_time_length = int(self.resting_time / dt)

        # Add the patterns and the pulse
        self.n_time_total = 0

        for epoch in range(self.epochs):
            for training_time, inter_pulse_interval in zip(self.training_times, self.inter_pulse_intervals):
                pattern_length = int(training_time / dt)
                inter_pulse_interval_length = int(inter_pulse_interval / dt)
                self.n_time_total += pattern_length + inter_pulse_interval_length

            # Add inter sequence interval or all but the last epoch
            if epoch < self.epochs - 1:
                self.n_time_total += inter_sequence_interval_length

        self.n_time_total += resting_time_length

        self.T_protocol_total = self.n_time_total * self.dt
        self.time = np.linspace(0, self.T_protocol_total, num=self.n_time_total)

        return self.T_protocol_total, self.n_time_total, self.time

    def simple_protocol(self, representation, training_times=1.0, inter_pulse_intervals=0.0,
                        inter_sequence_interval=1.0, epochs=1, resting_time=0.0):
        """
        The simples protocol to train a sequence

        :param patterns_indexes: All the indexes patterns that will be train
        :param training_time: How long to present the pattern
        :param inter_pulse_interval: Time between each pattern
        :param inter_sequence_interval: Time between each repetition of the sequence
        :param epochs: how many times to present the sequence
        """

        self.epochs = epochs
        self.activity_representation = representation.activity_representation
        self.network_representation = representation.build_network_representation()
        self.n_patterns = representation.n_patterns
        self.inter_sequence_interval = inter_sequence_interval
        self.resting_time = resting_time

        if isinstance(training_times, numbers.Number):
            self.training_times = [training_times for _ in range(self.n_patterns)]
        elif isinstance(training_times, (list, np.ndarray)):
            self.training_times = training_times
        else:
            raise TypeError('Type of training time not understood')

        if isinstance(inter_pulse_intervals, numbers.Number):
            self.inter_pulse_intervals = [inter_pulse_intervals for i in range(self.n_patterns)]
        elif isinstance(inter_pulse_intervals, (list, np.ndarray)):
            self.inter_pulse_intervals = inter_pulse_intervals
        else:
            raise TypeError('Type of inter-puse-interval not understood')

        patterns_sequence = []
        times_sequence = []
        learning_constants_sequence = []

        for i in range(epochs):
            # Let's fill the times
            for pattern, training_time, inter_pulse_interval in zip(self.network_representation,
                                                                    self.training_times,
                                                                    self.inter_pulse_intervals):
                # This is when the pattern is training
                patterns_sequence.append(pattern)
                times_sequence.append(training_time)
                learning_constants_sequence.append(1.0)

                # This happens when there is time between the patterns
                if inter_pulse_interval > 0.0:
                    patterns_sequence.append(None)
                    times_sequence.append(inter_pulse_interval)
                    learning_constants_sequence.append(0.0)

            # Remove the inter pulse interval at the end of the patterns
            if inter_pulse_interval > 0.0:
                patterns_sequence.pop()
                times_sequence.pop()
                learning_constants_sequence.pop()

            if inter_sequence_interval > 0.0 and i < epochs - 1:
                patterns_sequence.append(None)
                times_sequence.append(inter_sequence_interval)
                learning_constants_sequence.append(0.0)

            if resting_time > 0.0 and i == epochs - 1:
                patterns_sequence.append(None)
                times_sequence.append(resting_time)
                learning_constants_sequence.append(0.0)

            # End of epoch
            if epochs > 1:
                patterns_sequence.append(epoch_end_string)
                times_sequence.append(epoch_end_string)
                learning_constants_sequence.append(epoch_end_string)
        # Store
        self.patterns_sequence = patterns_sequence
        self.times_sequence = times_sequence
        self.learning_constants_sequence = learning_constants_sequence

    def cross_protocol(self, chain, training_time=1.0,  inter_sequence_interval=1.0, epochs=1):

        self.epochs = epochs
        self.patterns_indexes = {pattern for patterns in chain for pattern in patterns}  # Neat double iteration
        self.patterns_indexes = list(self.patterns_indexes)
        print(self.patterns_indexes)

        patterns_sequence = []
        times_sequence = []
        learning_constant_sequence = []

        for i in range(epochs):
            for patterns in chain:
                # Get the chains one by one
                for pattern in patterns:
                    patterns_sequence.append(pattern)
                    times_sequence.append(training_time)
                    learning_constant_sequence.append(1.0)

                # Get a space between the chains
                patterns_sequence.append(None)
                times_sequence.append(inter_sequence_interval)
                learning_constant_sequence.append(0.0)

            # Get the epoch if necessary
            if epochs > 1:
                patterns_sequence.append(epoch_end_string)
                times_sequence.append(epoch_end_string)
                learning_constant_sequence.append(epoch_end_string)

        # Store
        self.patterns_sequence = patterns_sequence
        self.times_sequence = times_sequence
        self.learning_constants_sequence = learning_constant_sequence


class TimedInput:
    def __init__(self, protocol, dt):

        self.protocol = protocol
        self.n_patterns = protocol.n_patterns
        self.n_units = protocol.network_representation.shape[1]
        self.dt = dt

        self.T_protocol_total, self.n_time_total, self.time = protocol.calculate_time_quantities(self.dt)

        self.network_representation = protocol.network_representation
        self.O = np.zeros((self.n_units, self.n_time_total))
        self.z_pre = np.zeros_like(self.O)
        self.z_post = np.zeros_like(self.O)
        self.tau_z_pre = None
        self.tau_z_post = None

    def build_timed_input(self):
        end = 0
        for epoch in range(self.protocol.epochs):
            for pattern, (training_time, inter_pulse_interval) in \
                    enumerate(zip(self.protocol.training_times, self.protocol.inter_pulse_intervals)):

                pattern_length = int(training_time / self.dt)
                inter_pulse_interval_length = int(inter_pulse_interval / self.dt)
                start = end
                end = start + pattern_length
                # Add the input
                indexes = np.where(self.network_representation[pattern])[0]
                self.O[indexes, start:end] = 1
                end += inter_pulse_interval_length

            inter_sequence_interval_length = int(self.protocol.inter_sequence_interval / self.dt)
            end += inter_sequence_interval_length

        return self.O

    def build_filtered_input_pre(self, tau_z):
        self.tau_z_pre = tau_z
        for index, o in enumerate(self.O.T):
            if index == 0:
                self.z_pre[:, index] = (self.dt / tau_z) * (o - 0)
            else:
                self.z_pre[:, index] = self.z_pre[:, index - 1] + (self.dt / tau_z) * (o - self.z_pre[:, index - 1])

        return self.z_pre

    def build_filtered_input_post(self, tau_z):
        self.tau_z_post = tau_z
        for index, o in enumerate(self.O.T):
            if index == 0:
                self.z_post[:, index] = (self.dt / tau_z) * (o - 0)
            else:
                self.z_post[:, index] = self.z_post[:, index - 1] + (self.dt / tau_z) * (o - self.z_post[:, index - 1])

        return self.z_post

    def calculate_probabilities_from_time_signal(self, filtered=True):
        if filtered:
            y_pre = self.z_pre
            y_post = self.z_post
        else:
            y_pre = self.O
            y_post = self.O

        n_units = self.n_units
        n_time_total = self.n_time_total

        p_pre = sp.integrate.simps(y=y_pre, x=self.time, axis=1) / self.T_protocol_total
        p_post = sp.integrate.simps(y=y_post, x=self.time, axis=1) / self.T_protocol_total

        outer_product = np.zeros((n_units, n_units, n_time_total))
        for index, (s_pre, s_post) in enumerate(zip(y_pre.T, y_post.T)):
            outer_product[:, :, index] = s_post[:, np.newaxis] @ s_pre[np.newaxis, :]

        P = sp.integrate.simps(y=outer_product, x=self.time, axis=2) / self.T_protocol_total

        return p_pre, p_post, P