import Utils
import Log


class Synapse(object):

    def __init__(self, weights, weight_count=-1, random_weight=False):
        self.weights = []
        if weight_count > 0 and random_weight is True:
            for i in range(0, weight_count):
                rnd = Utils.rnd()
                # Log.d("Adding random weight %f at index %d" % (rnd, i))
                self.weights.append(rnd)

        else:
            self.weights = weights
