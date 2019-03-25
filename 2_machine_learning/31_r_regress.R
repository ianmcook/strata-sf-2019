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

# # R regression models

# This example demonstrates a simple linear regression
# modeling task, first using R's built-in `lm()` function
# then using the `rpart()` function in the 
# [rpart](https://cran.r-project.org/package=randomForest)
# package.


# ## Preparation

# Load the required packages
library(readr)
library(rpart)

# Load data representing one brand of chess set ("set A")
chess <- read_csv("data/chess/one_chess_set.csv")

# View the data
chess

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

# In R, you do not need to instantiate an estimator 
# object before training a model. Instead, you train
# the model directly by calling the appropriate
# function. For a simple linear regression model, the
# function is `lm()`.

# R uses a formula notation to specify models. The 
# formula for a simple model with one predictor variable
# (`x`) and one response variable (`y`) is `y ~ x`
model <- lm(
  weight ~ base_diameter,
  data = chess_train
)

# Other models in R (including those in contributed R
# packages) typically use the same basic syntax and the
# same formula notation as the `lm()` function.

# For example, you could use the `rpart()` function
# in the rpart package simply by replacing `lm` with 
# `rpart`:
model2 <- rpart(
  weight ~ base_diameter,
  data = chess_train
)

# However, the objects that these different model training 
# functions return are very different, and the steps to
# evaluate the model are different depending on which 
# function you use.


# ## Next steps

# Instead of calling these different model training 
# functions directly, many R users instead use the
# [caret](http://topepo.github.io/caret/index.html)
# package, which was designed to help wrestle the 
# numerous different machine learning functions in
# different R packages into a more uniform interface, 
# and to standardize common tasks such as training 
# and evaluating models.
