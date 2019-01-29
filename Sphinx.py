from Cortex import *
from Network import Network
from Layer import Layer
from Synapse import Synapse
from Neuron import Neuron


def main():
    print('Cortex starting....')
    input_layer = Layer(neurons=[Neuron([1]), Neuron([0]), Neuron([1])],
                        layer_type=LayerType.INPUT)

    hidden_layer = Layer([Neuron(bias=-0.4), Neuron(bias=0.2)],
                         [Synapse([0.2, 0.4, -0.5]), Synapse([-0.3, 0.1, 0.2])],
                         LayerType.HIDDEN)

    output_layer = Layer([Neuron(bias=0.1)],
                         [Synapse([-0.3, -0.2])],
                         LayerType.OUTPUT)

    layers = [input_layer, hidden_layer, output_layer]

    network = Network(layers, 0.9)

    network.print_outputs()
    network.train([1, 1, 0])
    network.print_errors()


if __name__ == '__main__':
    main()
