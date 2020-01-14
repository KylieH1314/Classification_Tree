# Classification Tree
This project is impletmenting classification tree and classifify the data based on maximum impurity reduction. This project support building single tree, pruning tree, cross validation for pruning, random forest and prediction. 

## Included files
* [Classification_tree.py] https://github.com/36-750/assignments-KylieH1314/blob/master/classification-tree/classification_tree.py
* [test_classification_tree.py] https://github.com/36-750/assignments-KylieH1314/blob/master/classification-tree/test_classification_tree.py
* [SQL.py] https://github.com/36-750/assignments-KylieH1314/blob/master/classification-tree/SQL.py
* [SQL_test.py] https://github.com/36-750/assignments-KylieH1314/blob/master/classification-tree/SQL_test.py
* [Benchmark.ipynb] https://github.com/36-750/assignments-KylieH1314/blob/master/classification-tree/Benchmark.ipynb

## Structure
* Build a single tree by splitting dataset at maxiumum reduction point
* Prune the tree by the pruning factor from cross validation 
* Build random forest with random sampled dataset
* Return prediction with new dataset
* Is_valid function for tree validation
### Special note for SQL supportance
* Prediction feature not support SQL database as input
* Random forest not support the trees based on SQL database

## Getting Started

### Input for single tree builder
* SQL database 
* pandas dataframe
#### Pandas dataframe
* dataset (Pandas dataframe)
* respond_variable (String)
* impurity_function (Function)
#### SQL database
* SQL table name (String)
* predictors (List of string)
* repond_variable (String)
* impurity_function (SQL_Function)
* cursor (Psycogy cursor)
* max_depth (Positive integer)

## Running the tests
There are two testing files in the package. One is for pandas dataframe tree and the other is for SQL dataframe tree. 
To run the tests:
```
pytest test_classification_tree.py
```
or 
```
pytest SQL_test.py
```

## Performance

[pd_tree_building_proformance]classification-tree/result_pd_tree
The average time to build a tree with 50*6 pd dataset is 30.7s

[sql_tree_building_proformance]classification-tree/result_sql_tree
The average time to build a tre with 50*6 sql database is 16s

[result_pd_rf]classification-tree/result_pd_rf
The average time to build a ramdom forest with 50*6 pd dataset is 19.7s, whhile sklearn random forest use 0.052s

## Authors

* **Zhiyan (Kylie) Huang** - *Initial work* 
## Acknowledgments
* 36-650 Statistical Computing at Carnegie Mellon University
* Alex Reinhart
* Frank Kovacs
* Shamindra Shrotriya
* [Classification_tree]https://github.com/36-750/problem-bank/blob/master/All/classification-tree.pdf
