from Cortex import *
from Network import Network
from Layer import Layer
from Synapse import Synapse
from Neuron import Neuron
from TrainingPattern import TrainingPattern
import Log
import Utils
from matplotlib import pyplot as pyp


def main():

    Log.i('Sphinx is starting....')
    input_layer = Layer(neurons=[Neuron(), Neuron(), Neuron()],
                        layer_type=LayerType.INPUT)

    hidden_layer = Layer([Neuron(bias=-0.4), Neuron(bias=0.2)],
                         [Synapse([0.2, 0.4, -0.5]), Synapse([-0.3, 0.1, 0.2])],
                         LayerType.HIDDEN)

    output_layer = Layer([Neuron(bias=0.1)],
                         [Synapse([-0.3, -0.2])],
                         LayerType.OUTPUT)

    layers = [input_layer, hidden_layer, output_layer]

    network = Network(layers, 0.1)

    # Utils.save(network)
    network.print_outputs()
    network.train([TrainingPattern([1, 0, 1], [1])], 1000)
    network.print_errors()
    pyp.plot(network.errors_history)
    pyp.show()


if __name__ == '__main__':
    main()
