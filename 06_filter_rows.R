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

# # Returning a subset of rows from an R data frame

# Load packages and read games data
library(readr)
library(dplyr)
games <- read_csv("data/games/games.csv")
games


# Use the dplyr verb method `filter()` method to return the
# subset of rows that satisfy some conditions
games %>% filter(min_age < 5)

# You can use the pipe `%>%` _inside_ dplyr verbs as well as
# between them
games %>% filter(max_players %>% between(5, 6))

# To use a local variable in the dplyr verb, use `!!` before
# it to ensure no conflicts between variable names and 
# column names
list_price <- 10.00
games %>% filter(list_price <= !!list_price)

# You can use symbolic logical operators like `&` and `|`
# in `filter()`
games %>% filter(list_price < 10 & max_players >= 6)

# You can use other R operators and functions in `filter()`
games %>% filter(name %in% c("Clue", "Risk"))
games %>% filter(is.na(name))
games %>% filter(name %>% startsWith("C"))
games %>% filter(round(list_price, 0) == 20)

# You can negate the truth of an expression using `!`:
games %>% filter(!(name %in% c("Clue", "Risk")))
games %>% filter(!is.na(name))
