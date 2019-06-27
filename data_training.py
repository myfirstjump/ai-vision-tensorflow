import numpy as np

import tensorflow as tf
from tensorflow import keras

class DataTraining(object):
    def __init__(self):
        pass
    
    def model_design(self):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        return model

    def model_training(self, model, data):
        
        training_images, training_labels = data
        model.compile(optimizer = tf.compat.v1.train.AdamOptimizer(),
                      loss =  'sparse_categorical_crossentropy',
                      metrics = ['accuracy'])
        model.fit(training_images, training_labels, epochs=10)

        return model