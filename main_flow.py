import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

from utils import mnist_reader
from object_printer import ObjectPrinter
from data_training import DataTraining
from model_evaluation import ModelEvaluation
from data_preprocessing import DataProcessing
from data_generator import DataGenerator

class MainDataLoadingAndProcessing(object):

    def __init__(self):
        self.prt_obj = ObjectPrinter()
        self.processing_obj = DataProcessing()

    def train_test_data_loading(self):

        ###############################
        #  import fashion_mnist data  #
        ###############################
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

        return (x_train, y_train), (x_test, y_test)

    def mnist_dateset_obj_loading(self):
        mnist = input_data.read_data_sets('/app/data/fashion')
        return mnist

    def data_preprocessing(self, training_data, testing_data):
        x_train, y_train = training_data
        x_test, y_test = testing_data
        (x_train, y_train), (x_test, y_test) = self.processing_obj.grey_scale_normalize((x_train, y_train), (x_test, y_test))
        return (x_train, y_train), (x_test, y_test)

class MainTraining(object):
    def __init__(self):
        self.training_obj = DataTraining()
    
    def model_training(self, training_data):
        x_train, y_train = training_data
        model = self.training_obj.model_design(model_name='CNN')
        print(model.summary())
        model = self.training_obj.model_training(model, (x_train, y_train))
        return model
    
    def tf_model_training(self, data, output_dir):

        hyperparameters = {}
        hyperparameters['seed'] = 9487
        hyperparameters['batch_size'] = 32
        hyperparameters['X_dim'] = 784
        hyperparameters['z_dim'] = 10
        hyperparameters['h_dim'] = 128
        hyperparameters['lam'] = 10
        hyperparameters['n_disc'] = 1
        hyperparameters['lr'] = 1e-4

        model = self.training_obj.model_design(model_name='GAN', data=data, hyperparameters=hyperparameters) # or LSGAN
        self.training_obj.gan_model_training(data, output_dir, model, hyperparameters)



class MainEvaluationAndPrediction(object):
    def __init__(self):
        self.evaluation_obj = ModelEvaluation()
    
    def model_evaluation(self, model, testing_data):
        result = self.evaluation_obj.model_evaluation(model, testing_data)
        return result

def ai_vision_main(input_dir, output_dir):
    data_obj = MainDataLoadingAndProcessing()
    train_obj = MainTraining()
    evaluate_obj = MainEvaluationAndPrediction()

    (x_train, y_train), (x_test, y_test) = data_obj.train_test_data_loading()
    (x_train, y_train), (x_test, y_test) = data_obj.data_preprocessing((x_train, y_train), (x_test, y_test))
    model = train_obj.model_training((x_train, y_train))
    result = evaluate_obj.model_evaluation(model, (x_test, y_test))
    print(result)

def gan_training_main(input_dir, output_dir):
    data_obj = MainDataLoadingAndProcessing()
    train_obj = MainTraining()
    evaluate_obj = MainEvaluationAndPrediction()

    data = data_obj.mnist_dateset_obj_loading()
    train_obj.tf_model_training(data, output_dir)

if __name__ == "__main__":
    pass
    parser = argparse.ArgumentParser(description='AI-VISION-TF.')

    parser.add_argument('-i', '--input_dir', default='/app/data')
    parser.add_argument('-o', '--output_dir', default='/app/output/gan_output/')
    
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # ai_vision_main(input_dir, output_dir)
    gan_training_main(input_dir, output_dir)
    
