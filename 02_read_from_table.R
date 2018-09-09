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

# # Read data from a Hive table into an R data frame

# ## Using odbc and DBI

# Load packages
library(odbc)
library(DBI)

# Connect to Impala with ODBC
impala <- dbConnect(
  drv = odbc(),
  dsn = "Impala DSN"
)

# Read data into a data frame
games <- dbGetQuery(impala, "SELECT * FROM games")

# Disconnect
dbDisconnect(impala)

# View the data as a data frame
games

# View the data as a tibble
library(tibble)
as.tibble(games)
