from Cortex import *


class Network:
    layers = []
    learning_rate = 0.0

    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def train(self, expected_outputs):
        self.forward()
        self.print_outputs()
        self.backward(expected_outputs)

    def forward(self):

        for i in range(0, len(self.layers)):
            current_layer = self.layers[i]
            print('Layer ::  %i, Layer Type :: %s' % (i, current_layer.layer_type))
            if current_layer.layer_type != LayerType.INPUT:
                previous_layer = self.layers[i - 1]
                self.layers[i].calculate_outputs(previous_layer)

    def backward(self, expected_outputs):

        # calculate errors
        print('Calculating errors')
        for i in range(len(self.layers) - 1, -1, -1):
            if self.layers[i].layer_type == LayerType.OUTPUT:
                print('on output layer')
                self.layers[i].calculate_errors(expected_outputs)
            elif self.layers[i].layer_type == LayerType.HIDDEN:
                print('on hidden')
                next_layer = self.layers[i + 1]
                self.layers[i].calculate_hidden_errors(next_layer)

        # update weights and bias
        for i in range(1, len(self.layers)):
            for j in range(0, len(self.layers[i].synapses)):
                # j represents both synapse and neuron
                current_synapse = self.layers[i].synapses[j]
                current_neuron = self.layers[i].neurons[j]
                for k in range(0, len(current_synapse.weights)):
                    # k represents connections with previous
                    # delta weight = learning_rate * error of current neuron * output of previous neuron
                    delta = self.learning_rate * current_neuron.error * \
                            self.layers[i - 1].neurons[k].output
                    # update current weight
                    current_synapse.weights[k] = current_synapse.weights[k] + delta

                # update bias
                current_neuron.bias = current_neuron.bias + self.learning_rate * current_neuron.error

    def print_outputs(self):
        for i in range(0, len(self.layers)):
            current_layer = self.layers[i]
            print('Layer :: %s' % current_layer.layer_type)
            for j in range(0, len(current_layer.neurons)):
                current_neuron = current_layer.neurons[j]
                print('neuron %i  output %f' % (i, current_neuron.output))

    def print_errors(self):
        for i in range(0, len(self.layers)):
            current_layer = self.layers[i]
            for j in range(0, len(current_layer.synapses)):
                current_synapse = current_layer.synapses[j]
                print('layer :: %d , synapse :: %d errors' % (i, j))
                for k in range(0, len(current_layer.synapses)):
                    print(current_layer.neurons[j].error)
                print('weights')
                print(current_synapse.weights)
