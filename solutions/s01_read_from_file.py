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

# # Read data from a delimited text file into a pandas DataFrame

# Load modules
import numpy as np
import pandas as pd

# Read data using the `read_table` function
games = pd.read_table('data/games/games.csv', sep=',')

# The default delimiter for `read_table` is tab (`'\t'`).
# Use the `sep` parameter to specify a different
# delimiter

# For details about the `read_table` function, run the
# help command:
pd.read_table?

# Alternatively, use the `read_csv` function which uses
# a comma delimiter by default
games = pd.read_csv('data/games/games.csv')

# pandas provides other functions for reading other
# formats of file-based data. See the 
# [Input/Output section of the pandas API reference](https://pandas.pydata.org/pandas-docs/stable/api.html#input-output)
# for details.

# These functions return a `DataFrame` object
type(games)

# View the data
games

# To display the pandas DataFrame in a scrollable
# grid, set the pandas option
# `display.html.table_schema` to `True`:

pd.set_option("display.html.table_schema", True)

# View the data in a scrollable grid
games

# Change to previous setting
pd.set_option("display.html.table_schema", False)
