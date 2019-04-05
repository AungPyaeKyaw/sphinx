from Cortex import *
import Log
import Utils


class Network:
    errors_history = []
    layers = []
    learning_rate = 0.0

    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def train(self, training_pattern, iteration=-1, min_error=0.003):
        if iteration == -1:
            Log.i('Training method is in minimum error mode.')
            i = 0
            while self.get_error() >= min_error or self.get_error() == 0:
                for j in range(0, len(training_pattern)):
                    self.forward(training_pattern[j].inputs)
                    self.backward(training_pattern[j].outputs)
                Log.i('Min Error Mode : Iteration %d. Error %f' % (i, self.get_error()))
                i += 1
        else:
            Log.i('Training method is in iteration mode.')
            for i in range(0, iteration):
                for j in range(0, len(training_pattern)):
                    self.forward(training_pattern[j].inputs)
                    self.backward(training_pattern[j].outputs)
                self.errors_history.append(self.get_error())
                Log.i('Iteration %d. Error %f' % (i, self.get_error()))

    def forward(self, inputs):

        for i in range(0, len(self.layers)):
            current_layer = self.layers[i]
            Log.d('Layer ::  %i, Layer Type :: %s' % (i, current_layer.layer_type))
            if current_layer.layer_type == LayerType.INPUT:
                if len(inputs) > 0 and len(inputs) == len(current_layer.neurons):
                    for j in range(0, len(current_layer.neurons)):
                        current_layer.neurons[i].inputs.append(inputs[j])
                else:
                    Log.w('No input to input layer neurons.')
            elif current_layer.layer_type != LayerType.INPUT:
                previous_layer = self.layers[i - 1]
                self.layers[i].calculate_outputs(previous_layer)

    def backward(self, expected_outputs):

        # calculate errors
        for i in range(len(self.layers) - 1, -1, -1):
            if self.layers[i].layer_type == LayerType.OUTPUT:
                self.layers[i].calculate_errors(expected_outputs)
            elif self.layers[i].layer_type == LayerType.HIDDEN:
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
            Log.d('Layer :: %s' % current_layer.layer_type)
            for j in range(0, len(current_layer.neurons)):
                current_neuron = current_layer.neurons[j]
                Log.d('neuron %i  output %f' % (i, current_neuron.output))

    def get_result(self):
        result = []
        output_layer = self.layers[len(self.layers) - 1]
        for j in range(0, len(output_layer.neurons)):
            result.append(output_layer.neurons[j].output)
        return result

    def print_result(self):
        output_layer = self.layers[len(self.layers) - 1]
        for j in range(0, len(output_layer.neurons)):
            current_neuron = output_layer.neurons[j]
            Log.i('neuron %i  output %f' % (j, current_neuron.output))

    def print_errors(self):
        Log.i('-------- Printing errors and weights --------')
        for i in range(0, len(self.layers)):
            current_layer = self.layers[i]
            for j in range(0, len(current_layer.synapses)):
                current_synapse = current_layer.synapses[j]
                Log.i('layer :: %d , synapse :: %d errors' % (i, j))
                for k in range(0, len(current_layer.synapses)):
                    Log.i(current_layer.neurons[j].error)
                Log.i('weights')
                Log.i(current_synapse.weights)

    def get_error(self):
        error = 0.0
        output_layer = self.layers[len(self.layers) - 1]
        for i in range(0, len(output_layer.neurons)):
            error += output_layer.neurons[i].error
        return error / len(output_layer.neurons)

    def predict(self, inputs):
        self.forward(inputs)
        self.print_result()
        return Utils.max_index(self.get_result())
