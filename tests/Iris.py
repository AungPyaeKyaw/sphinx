from Cortex import *
from Network import Network
from Layer import Layer
from Synapse import Synapse
from Neuron import Neuron
from TrainingPattern import TrainingPattern
import Log
import Utils
from matplotlib import pyplot as pyp
from csv import *


def main():
    Log.i('Sphinx is starting....')
    # in put_layer = Layer(neurons=[Neuron(), Neuron(), Neuron(), Neuron()],
    # layer_type = LayerType.INPUT)

    # hidden_layer = Layer([Neuron(bias=-0.4), Neuron(bias=0.2)],
    # [Synapse([0.2, 0.4, -0.5, -0.3]), Synapse([-0.3, 0.1, 0.2, -0.3])],
    # LayerType.HIDDEN)

    # output_layer = Layer([Neuron(), Neuron(), Neuron()],
    # [Synapse([-0.3, -0.2]), Synapse([-0.2, 0.1]), Synapse([0.1, -0.4])],
    # LayerType.OUTPUT)

    input_layer = Layer(neuron_count=4, layer_type=LayerType.INPUT)
    hidden_layer = Layer(neuron_count=2, synapse_count=2, weight_per_synapse=4, layer_type=LayerType.HIDDEN)
    output_layer = Layer(neuron_count=3, synapse_count=3, weight_per_synapse=2, layer_type=LayerType.OUTPUT)

    layers = [input_layer, hidden_layer, output_layer]

    network = Network(layers, 0.3)

    # Create training from csv file
    patterns = []
    csv_reader = DictReader(f=open("iris.csv"))
    class_label = ["Iris-virginica", "Iris-versicolor", "Iris-setosa"]
    for i in csv_reader:
        values = []
    for j in range(0, 4):
        values.append(float(list(i.values())[j]))

    c = list(i.values())[4]
    if c == "Iris-virginica":
        c_label = [0, 0, 1]
    elif c == "Iris-versicolor":
        c_label = [0, 1, 0]
    elif c == "Iris-setosa":
        c_label = [1, 0, 0]
    else:
        c_label = []
    t = TrainingPattern(values, c_label)
    patterns.append(t)

    # Utils.save(network)
    # network.print_outputs()
    # network.train(patterns, 1000)
    network.load_weight('weights_iter=999_iter_mode.txt')
    i = network.predict([5.4, 3.4, 1.7, 0.2])
    print(class_label[i])
    # network.print_errors()
    # pyp.plot(network.errors_history)
    # pyp.show()


if __name__ == '__main__':
    main()
