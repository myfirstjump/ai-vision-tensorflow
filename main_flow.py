import argparse

import tensorflow as tf
from tensorflow import keras

from utils import mnist_reader
from object_printer import ObjectPrinter

class MainDataLoadingAndProcessing(object):

    def __init__(self):
        self.prt_obj = ObjectPrinter()
    

    def data_loading(self):

        ### import fashion_mnist data
        x_train, y_train = mnist_reader.load_mnist('/app/data/fashion', kind='train')
        x_test, y_test = mnist_reader.load_mnist('/app/data/fashion', kind='t10k')

        self.prt_obj.data_dimension_descripition(x_train, x_test)
        self.prt_obj.data_dimension_descripition(y_train, y_test)

        # otherwise, can import tensorflow build-in dataset by:
        # from tensorflow.examples.tutorials.mnist import input_data
        # data = input_data.read_data_set('data/fashion')
        # or data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
        # data.train.next_batch(BATCH_SIZE)

        # Keras build-in dataset by:
        # fashion_mnist = keras.dataset.fashion_mnist
        # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class MainTraining(object):
    def __init__(self):
        pass

class MainEvaluationAndPrediction(object):
    def __init__(self):
        pass


def ai_vision_main(input_dir, output_dir):
    data_obj = MainDataLoadingAndProcessing()
    # train_obj = MainTraining()
    # evaluate_obj = MainEvaluationAndPrediction()

    data_obj.data_loading()

if __name__ == "__main__":
    pass
    parser = argparse.ArgumentParser(description='AI-VISION-TF.')

    parser.add_argument('-i', '--input_dir', default='/app/data')
    parser.add_argument('-o', '--output_dir', default='/app/output')
    
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    ai_vision_main(input_dir, output_dir)
    
