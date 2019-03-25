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

# # Combining R data frames (adding rows)

# Load packages and read games data
library(readr)
library(dplyr)
games <- read_csv("data/games/games.csv")
games

# Create a second data frame describing more games
more_games <- tribble(
  ~id, ~name, ~inventor, ~year, ~min_age, ~min_players, ~max_players, ~list_price,
  6, 'Checkers', NA, -3000, 6, 2, 2,  8.99,
  7, 'Chess',    NA,   500, 8, 2, 2, 12.99
)
more_games


# Use the dplyr function `bind_rows()` to combine two
# data frames vertically, adding the rows of the second
# at the bottom of the first. This is equivalent to
# what the SQL operator `UNION ALL` does
games %>% bind_rows(more_games)


# To remove duplicates from the combined result, like 
# the SQL operator `UNION DISTINCT` does, use the dplyr
# function `distinct()` after combining the data frames.
# For example, the following series of operations
# combines the `games` and `more_games` data frames, 
# selects only the `min_players` and `max_players` 
# columns, and returns only the distinct (unique) rows 
# (the rows with unique combinations of `min_players`
# and `max_players`).
games %>% 
  bind_rows(more_games) %>%
  select(min_players, max_players) %>%
  distinct()
