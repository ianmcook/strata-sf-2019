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

# # Grouping and aggregating data in a pandas DataFrame

# Import modules and read games data
import numpy as np
import pandas as pd
games = pd.read_table('data/games/games.csv', sep=',')
games


# ## Aggregation without grouping

# Use the DataFrame method `agg` to reduce all rows of 
# a table down to a single row by aggregating the
# specified column by the specified function

# As the argument to `agg`, pass a dictionary in the form  
# `{'column_name': ['aggregate_function']}`
games \
  .agg({'list_price': ['mean']})

# You can also specify multiple aggregate functions in 
# the list
games \
  .agg({'min_age': ['count', 'nunique']})

# However, these results are transposed: the DataFrame 
# returned by the `agg` method has one _row_ for each
# of the aggregates. (Compare this to aggregation with 
# dplyr and SQL, which returns one _column_ for each
# aggregate.) To resolve this, transpose the result 
# using the `transpose` method
games \
  .agg({'min_age': ['count', 'nunique']}) \
  .transpose()

# The `agg` method does not give you control over the
# column names in the aggregated result, but you can use
# `rename` to rename them
games \
  .agg({'list_price': ['mean']}) \
  .transpose() \
  .rename(columns = {'mean': 'avg_list_price'})

games \
  .agg({'min_age': ['count', 'nunique']}) \
  .transpose() \
  .rename(
    columns = {
      'count': 'count_min_age',
      'nunique': 'unique_count_min_age'
    }
  )
  

# ## Aggregation with grouping

# Use the DataFrame method `groupby` immediately before
# `agg` to aggregate by groups
games \
  .groupby('min_age') \
  .agg({'list_price': ['mean']})
  
# When you use grouping, the result is not transposed.
  
# Load the flights dataset to demonstrate on larger data
flights = pd.read_csv('data/flights/flights.csv')

# You can specify muliple grouping columns in a list
flights \
  .groupby(['origin', 'month']) \
  .agg({'arr_delay': ['count', 'min', 'max', 'mean']})


# ## Grouping and aggregating missing values

# Load the inventory data (since the games data has no
# missing values)
inventory = pd.read_table('data/inventory/data.txt')
inventory

# The function `count` does not count missing values. To 
# count missing values, use `len`. Because `len` is a 
# funcion defined in the Python language, not in the 
# pandas package, do not enclose it in quotes:
inventory \
  .agg({'aisle': [len, 'count']}) \
  .transpose()

inventory \
  .groupby('shop') \
  .agg({'aisle': [len, 'count']}) 

# Aggregate functions ignore missing values
inventory \
  .groupby('shop') \
  .agg({'price': ['mean']})

# With pandas, missing values in grouping columns are not
# included in the results
inventory \
  .groupby('aisle') \
  .agg({'aisle': [len]}) 


# ## Method chaining after `groupby` and `agg`

# When you apply `groupby` and `agg`, the resulting 
# DataFrame has what's called a _MultiIndex_ (a 
# hierarchical index). You can see this by looking at
# the header of the DataFrame returned by the above
# examples; notice how there are three header rows
# instead of the usual one.

# To continue chaining DataFrame methods after 
# `groupby` and `agg`, you typically must flatten
# this MultiIndex. Then the result will appear with
# just one header row.

# If you do not flatten the MultiIndex, some 
# DataFrame methods later in the chain will fail. 
# For example, this will fail because `sort_values`
# does not expect a DataFrame with a MultiIndex:

#```python
#flights \
#  .groupby(['origin', 'month']) \
#  .agg({'arr_delay': ['count', 'max']}) \
#  .sort_values('max')
#```

# To flatten the multiindex, use square brackets 
# to flatten the first level (which has only one 
# value in it, named according to the column whose 
# values were aggregated) then use the `reset_index`
# method to flatten the second level. Then you can
# apply other methods like `sort_values` to the 
# result:

flights \
  .groupby(['origin', 'month']) \
  .agg({'arr_delay': ['count', 'max']}) \
  ['arr_delay'] \
  .reset_index() \
  .sort_values('max')

# However, this method only works if the `agg` 
# method aggregates the values from a single column.
# If the `agg` method aggregates values from 
# multiple columns, then it is necessary to redefine
# the columns while removing the levels of the 
# MultiIndex to give them unique, informative names.
# You can define a function that does this, then 
# use the DataFrame method `pipe` to apply this
# function in the chain after `groupby` and `agg`.
# Then you can apply other methods like 
# `sort_values` to the result:

def flatten_index(df):
  df_copy = df.copy()
  df_copy.columns = ['_'.join(col).rstrip('_') for col in df_copy.columns.values]
  return df_copy.reset_index()

flights \
  .groupby(['origin', 'month']) \
  .agg({
    'dep_delay': ['count', 'max'], \
    'arr_delay': ['count', 'max'] \
  }) \
  .pipe(flatten_index) \
  .sort_values('arr_delay_max')
