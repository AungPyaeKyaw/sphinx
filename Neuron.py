from Cortex import *
import math
import Log


class Neuron(object):

    def __init__(self, inputs=[], neuron_type=NeuronType.Sigmoid, bias=0.0):
        self.type = neuron_type
        self.inputs = inputs
        self.bias = bias
        self.error = 0.0
        self.output = 0.0
        if len(inputs) >= 1:
            self.output = inputs[0]

    def calculate_output(self, inputs, weights):
        Log.d('inputs and weights')
        Log.d(inputs)
        Log.d(weights)
        result = 0.0
        for i in range(0, len(inputs)):
            result += inputs[i] * float(weights[i])
        result += self.bias
        Log.d('before activation %f' % result)
        self.output = self.activation(result)
        Log.d('calculated output %f' % self.output)
        return self.output

    def calculate_error(self, expected_output):
        Log.d('expected output %s , actual output %s' %(expected_output,self.output))
        self.error = self.output * (1 - self.output) * (expected_output - self.output)
        Log.d('error at output layer %f' % self.error)
        return self.error

    def calculate_hidden_error(self, errors, weights):
        error_sum = 0
        for i in range(0, len(errors)):
            error_sum += errors[i] * weights[i]
        self.error = self.output * (1 - self.output) * error_sum
        Log.d('error at hidden layer %f' % self.error)
        return self.error

    def activation(self, value):
        return 1 / (1 + math.pow(math.e, (-1) * value))
