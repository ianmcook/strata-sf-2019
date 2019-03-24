#!/bin/bash

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

# Bash script to create and populate tables for Strata NY 2018 training: Expand your data science and machine learning skills

# Change the values of these variables before running
DEFAULT_DATABASE_NAME="default"
DEFAULT_DATABASE_LOCATION="hdfs:///user/hive/warehouse"
CHESS_DATABASE_NAME="chess"
CHESS_DATABASE_LOCATION="hdfs:///user/hive/warehouse/chess.db"
IMPALAD_HOST_PORT="hostname:21000"


put_file_if_dir_is_empty(){ # args: $1 = directory of local file, $2 = file name, $3 HDFS directory
    dir_empty=$(hdfs dfs -count $3 | awk '{print $2}')
  if [[ $dir_empty -eq 0 ]]; then
    hdfs dfs -put $1/$2 $3/$2
  else
    echo ERROR: $3 is not empty
    exit 1
  fi
}

put_file_no_headers_if_dir_is_empty(){ # args: $1 = directory of local file, $2 = file name, $3 HDFS directory
    dir_empty=$(hdfs dfs -count $3 | awk '{print $2}')
  if [[ $dir_empty -eq 0 ]]; then
    tail -n +2 $1/$2 | hdfs dfs -put - $3/$2
  else
    echo ERROR: $3 is not empty
    exit 1
  fi
}


impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE DATABASE IF NOT EXISTS ${var:dbname} LOCATION "${var:location}";
' || { echo ERROR: Failed to create database $DEFAULT_DATABASE_NAME ; exit 1; }

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.daylight
     (month TINYINT,
      day TINYINT,
      sunrise TIMESTAMP,
      sunset TIMESTAMP,
      light DECIMAL(6,4))
     ROW FORMAT DELIMITED
        FIELDS TERMINATED BY "\t"
        LINES TERMINATED BY "\n" 
     STORED AS TEXTFILE
     LOCATION "${var:location}/daylight/";
' || { echo ERROR: Failed to create table daylight in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/daylight daylight.tsv $DEFAULT_DATABASE_LOCATION/daylight

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.customers 
     (cust_id STRING,
      name STRING,
      country STRING)
     ROW FORMAT DELIMITED 
        FIELDS TERMINATED BY "\t"
        LINES TERMINATED BY "\n" 
     STORED AS TEXTFILE
     LOCATION "${var:location}/customers/";
' || { echo ERROR: Failed to create table customers in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/customers customers.txt $DEFAULT_DATABASE_LOCATION/customers

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.orders 
     (order_id INT,
      cust_id STRING,
      empl_id INT,
      total DECIMAL(5,2))
     ROW FORMAT DELIMITED 
        FIELDS TERMINATED BY "\t"
        LINES TERMINATED BY "\n" 
     STORED AS TEXTFILE
     LOCATION "${var:location}/orders/";
' || { echo ERROR: Failed to create table orders in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/orders orders.txt $DEFAULT_DATABASE_LOCATION/orders

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.offices
     (office_id STRING,
      city STRING,
      state_province STRING,
      country STRING)
     ROW FORMAT DELIMITED
        FIELDS TERMINATED BY "\t"
        LINES TERMINATED BY "\n" 
     STORED AS TEXTFILE
     LOCATION "${var:location}/offices/"
     TBLPROPERTIES("serialization.null.format"="");
' || { echo ERROR: Failed to create table offices in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/offices offices.txt $DEFAULT_DATABASE_LOCATION/offices

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.employees
     (empl_id INT,
      first_name STRING,
      last_name STRING,
      salary INT,
      office_id STRING)
     ROW FORMAT DELIMITED
        FIELDS TERMINATED BY "\t"
        LINES TERMINATED BY "\n" 
     STORED AS TEXTFILE
     LOCATION "${var:location}/employees/";
' || { echo ERROR: Failed to create table employees in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/employees employees.txt $DEFAULT_DATABASE_LOCATION/employees

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.salary_grades
     (grade TINYINT,
      min_salary INT,
      max_salary INT)
     ROW FORMAT DELIMITED
        FIELDS TERMINATED BY "\t"
        LINES TERMINATED BY "\n" 
     STORED AS TEXTFILE
     LOCATION "${var:location}/salary_grades/";
' || { echo ERROR: Failed to create table salary_grades in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/salary_grades salary_grades.tsv $DEFAULT_DATABASE_LOCATION/salary_grades

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.games
     (id INT,
      name STRING,
      inventor STRING,
      year STRING,
      min_age TINYINT,
      min_players TINYINT,
      max_players TINYINT,
      list_price DECIMAL(5,2))
     ROW FORMAT DELIMITED
        FIELDS TERMINATED BY ","
        LINES TERMINATED BY "\n" 
     STORED AS TEXTFILE
     LOCATION "${var:location}/games/";
' || { echo ERROR: Failed to create table games in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/games games.csv $DEFAULT_DATABASE_LOCATION/games

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.inventory
     (shop STRING,
     game STRING,
     qty INT,
     aisle TINYINT,
     price DECIMAL(5,2))
     ROW FORMAT DELIMITED
        FIELDS TERMINATED BY "\t"
        LINES TERMINATED BY "\n" 
     STORED AS TEXTFILE
     LOCATION "${var:location}/inventory/";
' || { echo ERROR: Failed to create table inventory in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/inventory data.txt $DEFAULT_DATABASE_LOCATION/inventory

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.crayons
     (color VARCHAR(25),
      hex CHAR(6),
      red SMALLINT,
      green SMALLINT,
      blue SMALLINT,
      pack TINYINT)
     ROW FORMAT DELIMITED
        FIELDS TERMINATED BY ","
        LINES TERMINATED BY "\n" 
     STORED AS TEXTFILE
     LOCATION "${var:location}/crayons/";
' || { echo ERROR: Failed to create table crayons in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/crayons crayons.csv $DEFAULT_DATABASE_LOCATION/crayons

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.flights
     (year SMALLINT,
      month TINYINT,
      day TINYINT,
      dep_time SMALLINT,
      sched_dep_time SMALLINT,
      dep_delay SMALLINT,
      arr_time SMALLINT,
      sched_arr_time SMALLINT,
      arr_delay SMALLINT,
      carrier STRING,
      flight SMALLINT,
      tailnum STRING,
      origin STRING,
      dest STRING,
      air_time SMALLINT,
      distance SMALLINT,
      hour TINYINT,
      minute TINYINT,
      time_hour TIMESTAMP)
    STORED AS PARQUET
    LOCATION "${var:location}/flights/";
' || { echo ERROR: Failed to create table flights in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_if_dir_is_empty data/flights flights.parquet $DEFAULT_DATABASE_LOCATION/flights

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME --var=location=$DEFAULT_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.airlines
     (carrier STRING,
      name STRING)
    ROW FORMAT DELIMITED
      FIELDS TERMINATED BY ","
      LINES TERMINATED BY "\n" 
    STORED AS TEXTFILE
    LOCATION "${var:location}/airlines/";
' || { echo ERROR: Failed to create table airlines in database $DEFAULT_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/airlines airlines.csv $DEFAULT_DATABASE_LOCATION/airlines

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$DEFAULT_DATABASE_NAME -q '
  REFRESH ${var:dbname}.daylight;
  REFRESH ${var:dbname}.customers;
  REFRESH ${var:dbname}.orders;
  REFRESH ${var:dbname}.offices;
  REFRESH ${var:dbname}.employees;
  REFRESH ${var:dbname}.salary_grades;
  REFRESH ${var:dbname}.games;
  REFRESH ${var:dbname}.inventory;
  REFRESH ${var:dbname}.crayons;
  REFRESH ${var:dbname}.flights;
  REFRESH ${var:dbname}.airlines;
' || { echo ERROR: Failed to refresh tables in database $DEFAULT_DATABASE_NAME ; exit 1; }

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$CHESS_DATABASE_NAME --var=location=$CHESS_DATABASE_LOCATION -q '
  CREATE DATABASE IF NOT EXISTS ${var:dbname} LOCATION "${var:location}";
' || { echo ERROR: Failed to create database $CHESS_DATABASE_NAME ; exit 1; }

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$CHESS_DATABASE_NAME --var=location=$CHESS_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.one_chess_set
     (piece STRING,
      base_diameter DECIMAL(6,4),
      height  DECIMAL(6,4),
      weight DECIMAL(6,4))
    ROW FORMAT DELIMITED
      FIELDS TERMINATED BY ","
      LINES TERMINATED BY "\n" 
    STORED AS TEXTFILE
    LOCATION "${var:location}/one_chess_set/";
' || { echo ERROR: Failed to create table one_chess_set in database $CHESS_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/chess one_chess_set.csv $CHESS_DATABASE_LOCATION/one_chess_set

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$CHESS_DATABASE_NAME --var=location=$CHESS_DATABASE_LOCATION -q '
  CREATE TABLE ${var:dbname}.four_chess_sets
     (`set` STRING,
      piece STRING,
      base_diameter DECIMAL(6,4),
      height  DECIMAL(6,4),
      weight DECIMAL(6,4))
    ROW FORMAT DELIMITED
      FIELDS TERMINATED BY ","
      LINES TERMINATED BY "\n" 
    STORED AS TEXTFILE
    LOCATION "${var:location}/four_chess_sets/";
' || { echo ERROR: Failed to create table four_chess_sets in database $CHESS_DATABASE_NAME ; exit 1; }

put_file_no_headers_if_dir_is_empty data/chess four_chess_sets.csv $CHESS_DATABASE_LOCATION/four_chess_sets

impala-shell -i $IMPALAD_HOST_PORT --var=dbname=$CHESS_DATABASE_NAME -q '
  REFRESH ${var:dbname}.one_chess_set;
  REFRESH ${var:dbname}.four_chess_sets;
' || { echo ERROR: Failed to refresh tables in database $CHESS_DATABASE_NAME ; exit 1; }
