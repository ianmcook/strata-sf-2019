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

# # Confusion matrix of TF model predictions vs. labels

# This example computes and displays a confusion matrix
# comparing a TensorFlow classification model's
# predictions to the true labels, using data from the
# test (evaluation) set.

# Run the code in `50_tf_classify.py` or 
# `51_tf_image_classify.py` before running the code below.


# ## Generate predicted labels for the test set

predictions = model.predict(
    input_fn = test_input_fn
)
test_pred = []
for prediction in predictions:
  test_pred.append(prediction['classes'][0].decode())


# ## Compute the confusion matrix

# TensorFlow has a function 
# [`tf.confusion_matrix`](https://www.tensorflow.org/api_docs/python/tf/confusion_matrix)
# for creating a confusion matrix, but it's often easier
# to use scikit-learn's 
# [`sklearn.metrics.confusion_matrix`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
# function

# Import the required module
from sklearn import metrics

# Compute the confusion matrix
confusion = metrics.confusion_matrix(
  y_true = test_y,
  y_pred = test_pred,
  labels = chess_pieces
)


# ## Display and interpret the confusion matrix

# Print the confusion matrix
print(confusion)

# - Rows represent the true labels
# - Columns represent predicted labels.


# ## Exercise

# The scikit-learn website provides
# [sample code](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
# to visualize a confusion matrix in a plot. Use the
# function defined in that code to plot the confusion
# matrix computed here.
