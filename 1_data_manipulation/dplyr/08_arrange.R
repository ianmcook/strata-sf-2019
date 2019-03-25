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

# # Ordering rows in an R data frame

# Load packages and read games data
library(readr)
library(dplyr)
games <- read_csv("data/games/games.csv")
games


# Use the dplyr verb `arrange` to sort the rows of a data 
# frame
games %>% arrange(min_age)

# The default sort order is ascending. Use the helper 
# function `desc` to sort in descending order
games %>% arrange(desc(min_age))

# You can specify multiple columns to sort by
games %>% arrange(desc(max_players), min_age)

# After ordering rows, use the `head` function to limit
# the number of rows returned, to get the "top N" results
# For example: What are the two least expensive games?
games %>% 
  select(name, list_price) %>%
  arrange(list_price) %>%
  head(2)
