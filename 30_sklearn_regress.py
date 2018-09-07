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

# # scikit-learn linear regression model

# This example demonstrates a simple linear regression
# modeling task using the
# [`LinearRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
# class in the 
# [`sklearn.linear_model`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
# module.


# ## Preparation

# Import the required modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data representing one brand of chess set ("set A")
chess = pd.read_csv('data/chess/one_chess_set.csv')

# View the data
chess

# Split the data into an 80% training set and a 20%
# evaluation (test) set, using scikit-learn's 
# [`train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
# function
train, test = train_test_split(chess, test_size=0.2)

# Separate the features (x) and targets (y) in the 
# training and test datasets
train_x = train.filter(['base_diameter'])
train_y = train.weight
test_x = test.filter(['base_diameter'])
test_y = test.weight


# ## scikit-learn setup

# Create the linear regression model object ("estimator")
# by calling the `LinearRegression` function
model = LinearRegression()


# ## Training and evaluating the model

# Call the `fit` method to train the model
model.fit(train_x, train_y)

# Call the `score` method to compute the coefficient of
# of determination (R-squared) on the test set. This is 
# the proportion of the variation in the features that
# can be explained by the model
model.score(test_x, test_y)


# ## Interpreting the model

# Print the coefficient (slope) of the linear regression
# model
model.coef_

# Print the intercept  of the linear regression model
model.intercept_


# ## Making predictions

# Call the `predict` method to use the trained model to
# make predictions (on the test set)
predictions = model.predict(test_x)


# ## Visualizing the model

# Display a scatterplot of the actual feature values (x)
# and target (y) values in the test set, with the 
# regression line overlaid
plt.scatter(test_x, test_y); plt.plot(test_x, predictions)
