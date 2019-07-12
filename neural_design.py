import os
import time
import math
import tensorflow as tf
import numpy as np

class NeuralCalculation(object):
    def __init__(self):
        pass
    
    def spectral_norm(self, w, iteration=1):
  
        def l2_norm(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = l2_norm(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = l2_norm(u_)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)
        return w_norm

    def linear(input_, output_size, name="linear", stddev=None, spectral_normed=False, reuse=False):
        shape = input_.get_shape().as_list()

        if stddev is None:
            stddev = np.sqrt(1. / (shape[1]))

        with tf.variable_scope(name, reuse=reuse) as scope:
            weight = tf.get_variable("w", [shape[1], output_size], tf.float32,
                tf.truncated_normal_initializer(stddev=stddev))

            bias = tf.get_variable("b", [output_size],
                initializer=tf.constant_initializer(0))

            if spectral_normed:
                mul = tf.matmul(input_, spectral_norm(weight))
            else:
                mul = tf.matmul(input_, weight)

        return mul + bias
  
    def conv2d(input_, output_dim, k_h=4, k_w=4, d_h=2, d_w=2, stddev=None, name="conv2d", spectral_normed=False, reuse=False, padding="SAME"):

        fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
        fan_out = k_h * k_w * output_dim
        if stddev is None:
            stddev = np.sqrt(2. / (fan_in))
        
        with tf.variable_scope(name, reuse=reuse) as scope:
            w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            if spectral_normed:
            conv = tf.nn.conv2d(input_, spectral_norm(w),
                                strides=[1, d_h, d_w, 1], padding=padding)
            else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

            biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
            
        return conv

    def deconv2d(input_, output_shape, k_h=4, k_w=4, d_h=2, d_w=2, stddev=None, name="deconv2d", spectral_normed=False, reuse=False, padding="SAME"):
        # Glorot initialization
        # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
        fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
        fan_out = k_h * k_w * output_shape[-1]
        if stddev is None:
            stddev = np.sqrt(2. / (fan_in))

        with tf.variable_scope(name, reuse=reuse) as scope:
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable("w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            if spectral_normed:
            deconv = tf.nn.conv2d_transpose(input_, spectral_norm(w),
                                            output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1], padding=padding)
            else:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1], padding=padding)

            biases = tf.get_variable("b", [output_shape[-1]], initializer=tf.constant_initializer(0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.shape(deconv))
            
        return deconv