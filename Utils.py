import json
from Network import *


def save(data):
    result = dict()
    if isinstance(data, Network):
        for i in range(0, len(data.layers)):
            current_layer = data.layers[i]
            weights = dict
            bias = dict
            for j in range(0, len(current_layer.synapses)):
                current_synapse = current_layer.synapses[j]
                weights = {"weights": current_synapse.weights}
                bias = {"bias": current_layer.neurons[j].bias}
            values = [weights, bias]
            result.update({"update": values})

    print(json.JSONEncoder().encode(result))


def max_index(inputs):
    max_i = 0
    max_v = inputs[0]
    for i in range(0, len(inputs)):
        if inputs[i] > max_v:
            max_i = i
            max_v = inputs[i]
    return max_i
