import os
import time

class ObjectPrinter(object):
    def __init__(self):
        pass
    
    def data_dimension_descripition(self, training_data, testing_data):

        print('Training Data:', training_data.shape)
        print('Testing Data:', testing_data.shape)
        