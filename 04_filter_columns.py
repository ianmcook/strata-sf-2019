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

# # Select specific columns from a pandas DataFrame

# Import modules and read games data
import numpy as np
import pandas as pd
games = pd.read_table('data/games/games.csv', sep=',')
games


# Use the DataFrame method `filter` method to return a 
# DataFrame containing only some of the columns from the 
# `games` DataFrame
games.filter(['name', 'min_players', 'max_players'])


# Write the expression on multiple lines
games \
  .filter(['name', 'min_players', 'max_players'])

# or

(games
  .filter(['name', 'min_players', 'max_players']))
