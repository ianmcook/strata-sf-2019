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

# # Ordering rows in a pandas DataFrame

# Import modules and read games data
import numpy as np
import pandas as pd
games = pd.read_table('data/games/games.csv', sep=',')
games

# Use the DataFrame method `sort_values` to sort the rows of
# a DataFrame
games.sort_values('min_age')

# The default sort order is ascending. Use the the parameter
# `ascending` to control the sort order
games.sort_values('min_age', ascending=False)

# You can specify multiple columns to sort by, in a list
# with the sort orders also specified in a list
games \
  .sort_values(
    ['max_players','min_age'],
    ascending=[False,True]
  )

# After ordering rows, use the `head` method to limit
# the number of rows returned, to get the "top N" results
# For example: What are the two least expensive games?
games \
  .filter(['name', 'list_price']) \
  .sort_values('list_price') \
  .head(2)

# Alternatively, use the `.iloc` indexer to limit or
# paginate results at the end of the chain of
# operations. Use slice notation: `start:(end+1)`
games \
  .filter(['name', 'list_price']) \
  .sort_values('list_price') \
  .iloc[0:2, :]
