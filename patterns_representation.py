import numpy as np
import IPython


class PatternsRepresentation:
    def __init__(self, activity_representation, minicolumns):
        self.activity_representation = activity_representation
        self.minicolumns = minicolumns
        self.n_patterns, self.hypercolumns = activity_representation.shape
        self.network_representation_available = False
        self.network_representation = None

    def build_network_representation(self):
        if self.network_representation_available:
            return self.network_representation
        else:
            self.network_representation = build_network_representation(self.activity_representation, self.minicolumns)
            self.network_representation_available = True
            return self.network_representation


def create_canonical_activity_representation(minicolumns, hypercolumns):
    aux = []
    for i in range(minicolumns):
        aux.append(i * np.ones(hypercolumns))

    return np.array(aux, dtype='int')


def build_network_representation(activity_representation, minicolumns):
    n_patterns, hypercolumns = activity_representation.shape
    network_representation = np.zeros((len(activity_representation), minicolumns * hypercolumns), dtype='int')

    for pattern, indexes in enumerate(activity_representation):
        for hypercolumn_index, minicolumn_index in enumerate(indexes):
            index = hypercolumn_index * minicolumns + minicolumn_index
            network_representation[pattern, index] = 1

    return network_representation





