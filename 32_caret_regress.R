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

# # caret regression models

# This example demonstrates a simple linear regression
# modeling task using the 
# [caret](http://topepo.github.io/caret/index.html)
# package, first for a simple linear regression model,
# then for a random forest model.


# ## Preparation

# Load the required packages
library(readr)
library(dplyr)
library(caret)

# Load data representing one brand of chess set ("set A")
chess <- read_csv("data/chess/one_chess_set.csv")

# Split the data into an 80% training set and a 20%
# evaluation (test) set
train_frac <- 0.8
indices <- sample.int(
  n = nrow(chess),
  size = floor(train_frac * nrow(chess))
)
chess_train <- chess[indices, ]
chess_test  <- chess[-indices, ]


# ## Specifying and training the model

# To create and train a model using the caret package,
# you call the `train()` function. The basic syntax and
# the formula notation are the same as for the `lm()` 
# function. The specific model you want to create is
# specified by the `method` argument. For example, to 
# create a linear regression model, you use 
# `method = "lm"`.
model <- train(
  weight ~ base_diameter,
  data = chess_train,
  method = "lm"
)


# ## Evaluating the trained model

# To evaluate the model, first you use the model to 
# generate predictions for the test (evaulation) set

# To generate predictions from the trained model, call
# caret's `predict()` function, passing the trained model
# object as the first argument, and the data to predict
# on as the `newdata` argument
test_pred <- predict(model, newdata = chess_test)

# To see the R help page for the `predict()` function, 
# run the command `?predict.train`.

# Use the `R2` function method to compute the 
# coefficient of determination (R-squared) on the test 
# set. This is the proportion of the variation in the
# target that can be explained by the model. Specify the
# predictions as the first argument, and the actual 
# values of the respose variable as the second argument.
R2(test_pred, chess_test$weight)


# ## Making predictions on new data

# See what predictions the trained model generates for
# six new rows of data (predictor variables only)
new_data <- tibble(
  base_diameter = c(27.3, 32.7, 30.1, 32.1, 35.9, 37.4),
  height = c(45.7, 58.1, 65.2, 46.3, 75.6, 95.4)
)

# Call the `predict` function to use the trained model to
# make predictions on this new data
predictions = predict(model, new_data)

# Print the predictions
predictions


# ## Other available regression models

# To see the list of all models that can be trained
# using the caret package, call the `modelLookup()`
# function, or see the 
# [list of supported models](http://topepo.github.io/caret/train-models-by-tag.html)
# on the caret website.

# You can filter the data frame returned by 
# `modelLookup()` to show only the models that can be
# used for regression tasks.
modelLookup() %>% filter(forReg)


# Let's see if we can get a better R-squared value by
# using a different model, and also by using a second
# predictor variable in the model.


# ## Specifying and training the model

# This time, specify `method = "rf"` to train a 
# random forest model. Internally, the `train()`
# function calls the `randomForest()` function in the
# randomForest package, so that package must be 
# installed, but you do not need to load it with a 
# `library()` command.

# The formula for a model with two predictor variables
# (`x1` and `y1`) and one response variable (`y`) is
# `y ~ x1 + x2`
model2 <- train(
  weight ~ base_diameter + height,
  data = chess_train,
  method = "rf"
)


# ## Evaluating the trained model

# Generate predictions from the trained model
test_pred <- predict(model2, newdata = chess_test)

# Compute the R-squared
R2(test_pred, chess_test$weight)


# ## Hyperparameter tuning

# To look for an indication of overfitting, see whether
# the R-squared on the training set is higher than the
# R-squared on the test set:
train_pred <- predict(model2, newdata = chess_train)
R2(train_pred, chess_train$weight)

# The model does not seem to be overfitting. If it were,
# you could adjust the hyperparameters. For example, to 
# reduce the maximum complexity of the model, you could
# set low values for the `ntree` and `maxnodes` 
# arguments:
#```r
# model3 <- train(
#   weight ~ base_diameter + height,
#   data = chess_train,
#   method = "rf",
#   ntree = 100,
#   maxnodes = 5
# )
#```

# The caret package passes these arguments to the 
# underlying `randomForest()` function.

# To see the full list of hyperparameters that you can
# use when the underlying model function is 
#`randomForest()`, see the R help page for that function:
?randomForest
