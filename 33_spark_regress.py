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

# # Spark MLlib regression model


# ## Preparation

# Import the required modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor

# Start a Spark session
spark = SparkSession.builder.master('local').getOrCreate()

# Load the data
chess = spark.table('chess.one_chess_set')

# Display a subset of rows from the Spark DataFrame
chess.show()

# Return the data as a pandas DataFrame
chess.toPandas()


# ## Preparing features

# Create a new DataFrame named `selected` with only the
# columns that will be used in the model
selected = chess.select('base_diameter', 'height', 'weight')

# Specify the names of the feature columns
feature_columns = ['base_diameter', 'height']

# Assemble the features into a single column of vectors
# in a DataFrame named `assembled`
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled = assembler.transform(selected)

# Split the `assembled` DataFrame into training and test 
# sets
(train, test) = assembled.randomSplit([0.8, 0.2], 12345)


# ## Specifying and training the model

# instantiate the Spark MLlib linear regression estimator
lr = LinearRegression(featuresCol="features", labelCol="weight")

# Call the `fit` method to fit (train) the linear regression
# model
lr_model = lr.fit(train)


# ## Evaluating the trained model

# Generate predictions on the test set
test_with_predictions = lr_model.transform(test)

# Create an instance of `RegressionEvaluator` class
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="weight", metricName="r2")

# Compute the R-squared
evaluator.evaluate(test_with_predictions)


# ## Interpreting the model

# Print the coefficient (slope) of the linear regression
# model
lr_model.coefficients

# Print the intercept  of the linear regression model
lr_model.intercept


# ## Other available regression models

# To see the list of all the regression and classification 
# models that Spark MLlib supports, see the 
# [Classification and regression](https://spark.apache.org/docs/latest/ml-classification-regression.html)
# page of the Spark MLlib guide.

# This time, let's also use the dataset representing four 
# different brands of chess sets, and use the column named
# `set` specifying which set each piece is from as one of  
# the features.

chess = spark.table('chess.four_chess_sets')

# Use `StringIndexer` to convert `set` from string codes to
# numeric codes
indexer = StringIndexer(inputCol="set", outputCol="set_ix")
indexer_model = indexer.fit(chess)
list(enumerate(indexer_model.labels))
indexed = indexer_model.transform(chess)

# Depending on the model, we might also need to apply another
# like the `OneHotEncoder` to generate a set of dummy variables
encoder = OneHotEncoder(inputCol="set_ix", outputCol="set_cd")
encoded = encoder.transform(indexed)

selected = encoded.select('base_diameter', 'height', 'set_cd', 'weight')
feature_columns = ['base_diameter', 'height', 'set_cd']

# we must assemble the features into a single column of vectors:
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled = assembler.transform(selected)

(train, test) = assembled.randomSplit([0.8, 0.2])
lr = RandomForestRegressor(featuresCol="features", labelCol="weight")

lr_model = lr.fit(train)

test_with_predictions = lr_model.transform(test)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="weight", metricName="r2")
evaluator.evaluate(test_with_predictions)


# End the Spark session
spark.stop()
