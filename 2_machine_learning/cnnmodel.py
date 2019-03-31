# Copyright 2019 Cloudera, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # Function definition for convolutional neural network estimator

# This function conforms to the interface expected 
# by TensorFlow's Estimator API. It uses the
# high-level API defined in TensorFlow's `tf.layers`
# module.

# This example is derived from the example at
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py
# which is described at
# https://www.tensorflow.org/tutorials/estimators/cnn

import numpy as np
import tensorflow as tf

def cnn_model_fn(features, labels, mode, params):
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # Images are 128x128 pixels with one color channel (grayscale)
  #input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
  input_layer = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns']), [-1, 128, 128, 1])
  
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 128, 128, 1]
  # Output Tensor Shape: [batch_size, 128, 128, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 128, 128, 32]
  # Output Tensor Shape: [batch_size, 16, 16, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 16, 16, 32]
  # Output Tensor Shape: [batch_size, 16, 16, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 8, 8, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 4], strides=4)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 8, 8, 64]
  # Output Tensor Shape: [batch_size, 8 * 8 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

  # Dense Layer
  # Densely connected layer with 64 neurons
  # Input Tensor Shape: [batch_size, 8 * 8 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 6]
  logits = tf.layers.dense(inputs=dropout, units=6)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
