from Cortex import LayerType
import Log


class Layer:
    neurons = []
    synapses = []
    layer_type = 0

    def __init__(self, neurons, synapses=[], layer_type=LayerType.INPUT):
        self.neurons = neurons
        self.layer_type = layer_type
        self.synapses = synapses

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
