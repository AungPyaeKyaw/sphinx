class TrainingPattern:
    inputs = []
    outputs = []

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __str__(self):
        return str(self.inputs) + " :: " + str(self.outputs)
