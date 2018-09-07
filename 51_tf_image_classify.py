# Copyright 2018 Cloudera, Inc.
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

# # TensorFlow DNNClassifier for image classification

# This example applies TensorFlow's
# [`DNNClassifier`](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier)
# pre-made estimator to a simple image classification 
# task.


# ## Preparation

# Import the required modules
import tensorflow as tf
import pandas as pd
import os, random, math
from IPython.display import Image, display

# Specify the unique labels (names of the chess pieces)
chess_pieces = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']

# Specify the root directory where the images are
img_root = 'data/chess/images'

# Make empty lists to hold image file paths (x) and 
# labels (y)
(x, y) = ([], [])

# There are images of pieces from four different chess
# sets (A, B, C, and D); specify which one use
chess_set = 'A'

# Fill the empty lists with the file paths and labels
for chess_piece in chess_pieces:
  img_dir = img_root + '/' + chess_set + '/' + chess_piece + '/'
  img_paths = [img_dir + d for d in os.listdir(img_dir)]
  img_labels = [chess_piece] * len(img_paths)
  x.extend(img_paths)
  y.extend(img_labels)

# View the image file paths and labels
for path, label in zip(x, y):
  print((path, label))

# Split the paths and labels into 80% training, 20% test
train_frac = 0.8
train_n = math.floor(train_frac * len(x))
indices = list(range(0, len(x)))
random.shuffle(indices)
train_indices = indices[0:train_n]
test_indices = indices[train_n:]
train_x = [x[i] for i in train_indices]
train_y = [y[i] for i in train_indices]
test_x = [x[i] for i in test_indices]
test_y = [y[i] for i in test_indices]


# ## TensorFlow setup

# Set constants for TensorFlow
BATCH_SIZE = 100
TRAIN_STEPS = 300

# Define a function that reads an image from a file,
# decodes it to numbers, and returns a two-element tuple 
# `(features, labels)` where `features` is a dictionary
# containing the image pixel data
def _parse_function(path, label):
    image = tf.image.decode_png(tf.read_file(path))
    return ({'image':image}, label)

# Define input functions to supply data for training
# and evaulating the model

# These functions apply `_parse_function`
def train_input_fn():
  dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
  dataset = dataset.map(_parse_function)
  dataset = dataset.shuffle(len(train_x)).repeat().batch(BATCH_SIZE)
  return dataset

def test_input_fn():
  dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(BATCH_SIZE)
  return dataset


# ## Specifying the model

# Create a list with the feature column
my_feature_columns = [
  tf.feature_column.numeric_column('image', shape=[128, 128])
]

# Instantiate a `DNNClassifier` estimator

# In this example, the [optimizer](https://www.tensorflow.org/api_guides/python/train#Optimizers) 
# used to train the model is specified, because the default
# Adagrad optimizer yielded a model with poor accuracy.
# The optimizer's learning rate is also specified, because
# the default value of 0.001 caused the algorithm to
# converge to a local minimum.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[1024, 128],
    optimizer=tf.train.AdamOptimizer(
      learning_rate=0.0001
    ),
    label_vocabulary=chess_pieces,
    n_classes=6
)


# ## Training and evaluating the model

# Call the `train` method to train the model
model.train(
  input_fn=train_input_fn,
  steps=TRAIN_STEPS
)

# Call the `evaluate` method to evaluate (test) the
# trained model
eval_result = model.evaluate(
  input_fn=test_input_fn
)

# Print the result to examine the accuracy
print(eval_result)


# ## Making predictions

# Use the trained model to generate predictions
# on unlabeled images

# Some of these images are of pieces from other
# chess sets (not from set A)
img_dir = img_root + '/unknown/'
img_paths = [img_dir + d for d in os.listdir(img_dir)]
pred_x = img_paths

# Define a function that reads an image from a file

# This is similar to the `_parse_function` function
# defined above, but without labels
def _predict_parse_function(path):
    image = tf.image.decode_png(tf.read_file(path))
    return ({'image':image})

# Define an input function to supply data for generating
# predictions
def predict_input_fn():
  dataset = tf.data.Dataset.from_tensor_slices(pred_x)
  dataset = dataset.map(_predict_parse_function)
  dataset = dataset.batch(BATCH_SIZE)
  return dataset

# Call the `predict` method to use the trained model to
# make predictions
predictions = model.predict(
    input_fn=predict_input_fn
)

# Print the predictions and display the images
template = ('\n\n\n\nPrediction is "{}" ({:.1f}%) from image:"')
for (prediction, image) in zip(predictions, pred_x):
    class_name = prediction['classes'][0].decode()
    class_id = prediction['class_ids'][0]
    probability = prediction['probabilities'][class_id]
    print(
      template.format(
        class_name,
        100 * probability
      )
    )
    display(Image(image))


# ## Exercises

# 1. This code trains the model using only images of
#    pieces from chess set A. The resulting trained 
#    model is poor at generalizing to images of pieces
#    from other chess sets. Modify the code in the
#    **Preparation** section to train the model using
#    the images of pieces from all four sets (A, B, 
#    C, and D). How does this affect the accuracy
#    of the model on the test (evaluation) set?
#
# 2. In the **TensorFlow setup** section and the
#    **Specifying the model** section, modify 
#    `BATCH_SIZE`, `TRAIN_STEPS`, the number of
#    hidden layers, and the number of nodes in the
#    hidden layers to try to improve the accuracy of
#    the model.
#
# 3. After making these changes, does the model do 
#    a better job of generating predictions on the
#    unlabeled images?
#
# 4. Modify the code in the **Making predictions**
#    section to use images from the `weird` directory
#    instead of the `unknown` directory. How well
#    does the model predict on these images?


# ## Next steps

# Dense neural networks (with all layers fully connected)
# are not well-suited to image classification tasks except
# in relatively simple cases like this one. Predictions 
# are sensitive to the size and position of the objects 
# in the image; they cannot robustly generalize.

# Accuracy can be improved by increasing the number of
# hidden layers and nodes, increasing the number of
# training steps, and using larger amounts of more
# diverse training data, but this is inefficient.

# Convolutional neural networks (CNNs) provide a solution.
# In addition to dense (fully connected) layers, they use
# - Convolutional layers (for filtering and weighting)
# - Pooling layers (for downsampling)

# These types of layers allow DNNs to differentiate between
# images based on subregions, and efficiently learn what
# visual features are most important for predicting labels.

# TensorFlow does not provide pre-made estimators for CNNs
# but you can use the estimator API to build your own.

# To learn more, see the TensorFlow tutorial:
# [Build a Convolutional Neural Network using Estimators](https://www.tensorflow.org/tutorials/estimators/cnn).

# Alternatively, you could build a CNN for image 
# classification using TensorFlow's 
# [Keras API](https://www.tensorflow.org/guide/keras)
# which offers a higher level of abstraction.
