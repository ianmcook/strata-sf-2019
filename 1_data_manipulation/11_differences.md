# Watch out for unexpected differences between between 
# dplyr and pandas

# For example, with dplyr, the command

#```r
#inventory %>% group_by(shop) %>% head(1)
#```

# returns just one row:

#| shop      | game     | qty | aisle | price |
#|-----------|----------|-----|-------|-------|
#| Dicey     | Monopoly | 7   | 3     | 17.99 |


# but with pandas, the command

#```python
#inventory.groupby('shop').head(1)
#```

# returns two rows, one for each group:

#| shop      | game     | qty | aisle | price |
#|-----------|----------|-----|-------|-------|
#| Dicey     | Monopoly | 7   | 3     | 17.99 |
#| Board 'Em | Monopoly | 11  | 2     | 25.00 |
