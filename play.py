import IPython
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from connectivity_functions import fill_connection
from patterns_representation import create_canonical_activity_representation
from network import Protocol, NetworkManager, Network
from patterns_representation import PatternsRepresentation
from analysis_functions import calculate_persistence_time, calculate_recall_quantities
from plotting_functions import plot_weight_matrix, plot_network_activity_angle
from analysis_functions import calculate_probability_theo, calculate_joint_probabilities_theo
from analysis_functions import calculate_self_probability_theo, calculate_get_weights_theo


S = '())'

def solution(S):
    stack_counter = 0
    for s in S:
        print(s)
        if s=='(':
            stack_counter += 1
        if s==')':
            stack_counter -= 1
        print(stack_counter)
    if stack_counter == 0:
        return 1
    else:
        return 0


expect = solution(S)
print('solution', expect)
