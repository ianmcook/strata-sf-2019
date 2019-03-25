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

# # Getting the rows with the top _n_ values in a column of a pandas DataFrame

# Import modules and read inventory data
import numpy as np
import pandas as pd
inventory = pd.read_table('data/inventory/data.txt', sep="\t")
inventory

# Example: What's the least expensive game in each shop?

inventory \
  .sort_values('price', ascending=True) \
  .groupby('shop') \
  .head(1)

# To return find the _most_ expensive game in each shop,
# set `ascending=False`.
  
# Optional:
# Add `.sort_values('shop')` to arrange by shop
# Add `.reset_index(drop=True)` to reset the indices
