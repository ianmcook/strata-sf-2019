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

# # sparklyr example with dplyr verbs

# Load required packages
library(sparklyr)
library(dplyr)

# Start a Spark session
spark <- spark_connect(master = "local")

# Load the data
flights <- tbl(spark, "flights")

# Use dplyr verbs to perform operations on the Spark
# DataFrame and return a pointer to the result DataFrame
flights %>%
  filter(dest == "LAS") %>%
  group_by(origin) %>%
  summarise(
    num_departures = n(),
    avg_dep_delay = mean(dep_delay, na.rm = TRUE)
  ) %>%
  arrange(avg_dep_delay)

# In this case, the full result DataFrame is printed to the
# screen because it's so small

# End the Spark session
spark_disconnect(spark)
