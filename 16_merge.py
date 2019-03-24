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

# # Joining pandas DataFrames

# Import modules
import numpy as np
import pandas as pd

# Read employees data
employees = pd.read_table('data/employees/employees.txt')
employees

# Read offices data
offices = pd.read_table("data/offices/offices.txt")
offices


# The DataFrame method `merge` can be used to join two
# DataFrames together. The parameter `how` specifies
# what type of join to perform:
# - `how='inner'` for inner joins
# - `how='left'` for left outer joins
# - `how='right'` for right outer joins
# - `how='outer'` for full outer joins

# For example, use `merge` with `how='left'` to perform
# a left outer join on the `employees` and `offices` 
# DataFrames
employees.merge(offices, how='left')

# pandas automatically identifies common column names in
# the two DataFrames and joins on them. To manually
# specify the join key columns, use the `on` parameter
employees.merge(offices, how='left', on='office_id')

# For more details, see the documentation for
# [`pandas.DataFrame.merge`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html).
