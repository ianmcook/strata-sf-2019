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

# # Inspect an R data frame or tibble

# Load packages and read data
library(readr)
games <- read_csv("data/games/games.csv")
games


# How many rows and columns does the data have?
dim(games)
ncol(games)
nrow(games)


# What are the names and data types of the columns?
games # Read from top rows of tibble
colnames(games)
sapply(games, class)
