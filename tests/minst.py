import Log
import sys
import numpy as np
from Cortex import *
from Neuron import *
from TrainingPattern import *
from Layer import *
from Network import *
from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot as pyp


def main():
    Log.i('Sphinx is starting....')
    # Load MNIST data
    x, y = loadlocal_mnist(
        images_path='images',
        labels_path='label'
    )

    Log.i('Dimensions: %s  x %s' % (x.shape[0], x.shape[1]))

    input_layer = Layer(neuron_count=784, layer_type=LayerType.INPUT)
    hidden_layer = Layer(neuron_count=392, synapse_count=392, weight_per_synapse=784, layer_type=LayerType.HIDDEN)
    output_layer = Layer(neuron_count=10, synapse_count=10, weight_per_synapse=392, layer_type=LayerType.OUTPUT)

    layers = [input_layer, hidden_layer, output_layer]

    network = Network(layers, 0.3)
    if len(sys.argv) > 0:
        Log.debug = bool(sys.argv[0])
    else:
        Log.debug = False
    patterns = []
    Log.i('Loading data...')
    for i in range(0, len(x)):
        cl_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cl_label[y[i]] = 1
        c_x = np.divide(x[i], 255)
        patterns.append(TrainingPattern(c_x, cl_label))

    Log.i('Loading data finished')
    network.train(patterns, 10, save_weight_per_ite=1)
    network.predict(x[0])
    # network.print_errors()
    # pyp.plot(network.errors_history)
    # pyp.show()


if __name__ == "__main__":
    main()
