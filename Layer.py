from Cortex import LayerType
from Neuron import Neuron
from Synapse import Synapse
import Log


class Layer(object):

    def __init__(self, neurons=[], synapses=[], layer_type=LayerType.INPUT, neuron_count=-1,
                 synapse_count=-1, weight_per_synapse=-1):
        self.neurons = []
        self.synapses = []
        if len(neurons) > 0:
            Log.i("Using custom neurons.")
            self.neurons = neurons
        elif neuron_count > 0:
            Log.i("Creating Layer with %d neurons." % neuron_count)
            for i in range(0, neuron_count):
                self.neurons.append(Neuron())
            Log.d('debug neuron count %s' % len(self.neurons))
        self.layer_type = layer_type

        if len(synapses) > 0:
            self.synapses = synapses
        elif synapse_count > 0 and weight_per_synapse> 0:
            for i in range(0, synapse_count):
                self.synapses.append(Synapse(weights=[], weight_count=weight_per_synapse, random_weight=True))

    def calculate_outputs(self, previous_layer):
        Log.d('calculate output -> Layer type :: %s' % self.layer_type)
        for i in range(0, len(self.neurons)):
            inputs = []
            for j in range(0, len(previous_layer.neurons)):
                inputs.append(previous_layer.neurons[j].output)
            Log.d('input from previous layer output %d' % previous_layer.neurons[j].output)
            self.neurons[i].calculate_output(inputs, self.synapses[i].weights)

    def calculate_errors(self, expected_outputs):
        for i in range(0, len(self.neurons)):
            self.neurons[i].calculate_error(expected_outputs[i])

    def calculate_hidden_errors(self, next_layer):
        for i in range(0, len(self.neurons)):
            errors = next_layer.get_errors()
            weights = []
            for j in range(0, len(next_layer.synapses)):
                weights.append(next_layer.synapses[j].weights[i])
            self.neurons[i].calculate_hidden_error(errors, weights)

    def get_errors(self):
        errors = []
        for i in range(0, len(self.neurons)):
            errors.append(self.neurons[i].error)
        return errors
