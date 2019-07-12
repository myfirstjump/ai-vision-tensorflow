import time

import numpy as np

import tensorflow as tf
from tensorflow import keras

from neural_design import NeuralCalculation, LossDeisgn

class DataTraining(object):
    def __init__(self):
        self.neural_obj = NeuralCalculation()
        self.loss_obj = LossDesign()

    def sys_show_execution_time(method):
        def time_record(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            execution_time = np.round(end_time - start_time, 3)
            print('Running function:', method.__name__, ' cost time:', execution_time, 'seconds.')
            return result
        return time_record
        
    def model_design(self, model_name):

        if model_name == 'DNN':
            model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dense(256, activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
        
        if model_name == 'CNN':
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=64, input_shape=(28,28,1), kernel_size=(3,3), strides=1, padding='valid', activation=tf.nn.relu),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='valid', activation=tf.nn.relu),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])

        if model_name = 'GAN':
            tf.reset_default_graph()

            X = tf.placeholder(tf.float32, shape=[None, X_dim])
            X_target = tf.placeholder(tf.float32, shape=[None, X_dim])
            z = tf.placeholder(tf.float32, shape=[None, z_dim])

            G_sample, G_var = self.neural_obj.generator(z)
            D_real_logits, D_var = self.neural_obj.discriminator(X, spectral_normed=False)
            D_fake_logits, _ = self.neural_obj.discriminator(G_sample, spectral_normed=False)

            D_loss, G_loss = self.loss_obj.gan_loss(D_real_logits, D_fake_logits, gan_type='GAN', relativistic=False)
            D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)).minimize(D_loss, var_list=D_var)
            G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)).minimize(G_loss, var_list=G_var)

            # z search
            z_optimizer = tf.train.AdamOptimizer(0.0001)
            z_r = tf.get_variable('z_update', [batch_size, z_dim], tf.float32)
            G_z_r, _ = self.neural_obj.generator(z_r, reuse=True)

            z_r_loss = tf.reduce_mean(tf.abs(tf.reshape(X_target, [-1, 28, 28, 1]) - G_z_r))
            z_r_optim = z_optimizer.minimize(z_r_loss, var_list=[z_r])

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            

        return model



    @sys_show_execution_time
    def model_training(self, model, data):
        
        training_images, training_labels = data
        model.compile(optimizer = 'adam',#optimizer = tf.compat.v1.train.AdamOptimizer(),
                      loss =  'sparse_categorical_crossentropy',
                      metrics = ['accuracy'])
        callbacks = CallBack()
        model.fit(training_images, training_labels, epochs=15, callbacks=[callbacks])

        return model

class CallBack(tf.keras.callbacks.Callback):

    # Each epoch end, will call the method on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.98):
            print('Reached enough accuracy so stop training...')
            self.model.stop_training = True