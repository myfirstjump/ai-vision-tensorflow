import numpy as np

class DataProcessing(object):
    def __init__(self):
        pass

    def grey_scale_normalize(self, training_data, testing_data):
        x_train, y_train = training_data
        x_test, y_test = testing_data

        x_train = x_train.reshape(60000, 28, 28, 1) # A single tensor should contain everything while doing convNet
        x_test = x_test.reshape(10000, 28, 28, 1)

        x_train = x_train/255.0
        x_test = x_test/255.0

        return (x_train, y_train), (x_test, y_test)