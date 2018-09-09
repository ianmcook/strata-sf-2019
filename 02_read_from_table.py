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

# # Read data from a Hive table into a pandas DataFrame

# Import modules
import numpy as np
import pandas as pd


# ## Using impyla

import impala.dbapi
con = impala.dbapi.connect(host='worker-1', port=21050)
sql = 'SELECT * FROM games'
games = pd.read_sql(sql, con)
games


# ## Using pyodbc

import pyodbc
con = pyodbc.connect('DSN=Impala DSN', autocommit=True)
sql = 'SELECT * FROM games'
games = pd.read_sql(sql, con)
games
