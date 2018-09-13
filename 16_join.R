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

# # Joining R data frames

# Load packages
library(readr)
library(dplyr)

# Read employees data
employees <- read_tsv("data/employees/employees.txt")
employees

# Read offices data
offices <- read_tsv("data/offices/offices.txt")
offices


# dplyr provides several _two-table verbs_ that can be 
# used to join two data frames together. These include:
# - `inner_join()` for inner joins
# - `left_join()` for left outer joins
# - `right_join()` for right outer joins
# - `full_join()` for full outer joins

# For example, use `left_join()` to perform a left outer
# join on the `employees` and `offices` data frames
employees %>% left_join(offices)

# dplyr automatically identifies common column names in
# the two data frames and joins on them. To manually
# specify the join key columns, use the `by` argument
employees %>% left_join(offices, by = "office_id")

# For more details, see the dplyr `join` help page:
?join
