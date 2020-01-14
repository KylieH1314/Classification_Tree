import psycopg2
import classification_tree
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database="zhiyanh",
                user="zhiyanh", password="ahY0ieMou")
cur = conn.cursor()
class SQL_node(classification_tree.Node):
    def __init__(self, table, predictors, respond_variable,impurity_function, cur, max_depth, prev_split = []):
        """
        Build classification trees by recursively initailize
        and split each node.

        Parameters
        -----------
        table: String
        The SQL table name
        
        predictors: String List
        List of the predictors that will be used for tree building
        
        repond_variable: String
        name of reponse variable
        
        impurity_function: Function
        SQL function to calculate impurity
        
        cur:psycogy cursor
        
        max_depth:integer
        The maximum level that tree can reach
        
        prev_split: String
        The list store the split record
        
        Returns
        --------
        None
        """
        
        self.predictors = predictors
        self.respond_variable = respond_variable
        self.table = table
        self.split_variable = None
        self.split_value = None
        self.right_child = None
        self.left_child = None   
        
        cur.execute("SELECT COUNT(*) FROM "+table + ";")
        self.whole_data_length = cur.fetchone()[0]
        
        self.leaf = True
        self.prev_split = prev_split
        
        # If curent node is not root node
        if len(self.prev_split) != 0:
            cur.execute("SELECT COUNT(*) FROM " + self.table + " WHERE " + " AND ".join(self.prev_split) + ";")
            self.own_data_point = cur.fetchone()[0]
            
            # If there is nothing in the current dataset
            if self.own_data_point != 0:
                cur.execute("SELECT fraction_ones("+respond_variable+") FROM "+table +" WHERE " + " AND ".join(prev_split) + ";")
                self.pA1 = cur.fetchone()[0]
            else:
                self.pA1 = 0
        
        # If current node is a root node
        else:
            # Check missingness
            cur.execute("SELECT * FROM " + self.table + " WHERE " + " IS NULL OR ".join(predictors) + " IS NULL;")
            missing_count = cur.fetchone()
            assert missing_count is None,"There are some missing values in the dataset"
            
            # Check binary reponse
            cur.execute("SELECT DISTINCT "+respond_variable+" FROM "+self.table+" ;")
            y_col = cur.fetchone()
            assert len(y_col) == 2 or len(y_col) == 1, "The response variable is not binary."
            assert 1 in y_col or 0 in y_col, "The response variable is not binary."
            
            cur.execute("SELECT COUNT(*) FROM "+table+" WHERE "+self.respond_variable + " = 1;")
            self.pA1 = cur.fetchone()[0]/self.whole_data_length
            self.own_data_point = self.whole_data_length
        
        # Take the majority of the reponse as the result of the current node
        self.predict = 1 if self.pA1 >= 0.5 else 0

        # If current node is pure (namely p=1 or p=0) or the level of the tree reach maximum depth, the tree sould stop splitting.
        # Otherwise the tree spliting keeps going
        if self.pA1 != 1 and len(prev_split)<max_depth and self.pA1 != 0 :
            self.leaf = False
            self.split_variable, self.split_value=self.find_best_splitting_point(prev_split, impurity_function,cur)
            
            # Filter out the corner case that dataset on the current node is pure or only has one line
            if self.split_variable is not None and self.split_value is not None:
                
                # Save the past split for left node
                prev_split_left = self.prev_split.copy()
                prev_split_left.append((str(self.split_variable)+" < "+str(self.split_value)))
                self.left_child = SQL_node(self.table, self.predictors, self.respond_variable,impurity_function, cur, max_depth, prev_split_left)

                # Save the past split for right node
                prev_split_right = self.prev_split.copy()
                prev_split_right.append((str(self.split_variable)+" >= "+str(self.split_value)))
                self.right_child = SQL_node(self.table, self.predictors, self.respond_variable,impurity_function, cur, max_depth,  prev_split_right)

    def find_best_splitting_point(self, prev_split, impurity_function,cur ):
        """
        This function is used to iterate each potential
        splitting value for each column and return the
        splitting point and value that
        has max impurity reduction.
    
        Iterate though the rows, find the maximum p value
        for each varaible, then
        find the variable that has the max p value.
        Then we need to access the p value
        of the children to find the maximum delta IA
    
        Unsolved question: Whether the p will be same,
        if we do same step size or inversal for each variable.
    
        Parameters
        ----------
        prev_split: List 
            the list of previous pslit
        
        impurity_function: function with p
            A function with p to get impurity score
            and impurity reduction.
    
        Returns
        --------
        split_variable: string
            A string of the column name of
            splitting point.
    
        split_point_value: index
            A index of the row that first contain
            the splitting value.
        """
        
        impurity_reduction = []
        max_redu = (0,0)
        
        # Loops over the predictors
        for col in self.predictors:
            # In case the user put a column name as both predictors and repond variable
            if col != self.respond_variable:
                cur.execute("SELECT DISTINCT "+col+" FROM "+self.table+" ;")
                target_col = cur.fetchall()
                
                # Filter out the corner case that there is no dataset
                if len(target_col) != 1:
                    impurity_reduction_each_col = dict()
                    
                    # Loop over the distinct values in the target column
                    for row in target_col:
                        row = float(row[0])
                        
                        # If the current value is not the samllest value in the target column
                        if row != float(min(target_col)[0]):
                            row = str(row)
                            current_right_split = col + " >= " + row
                            current_left_split = col + " < " + row
                            
                            # If the node is root node
                            if len(self.prev_split) == 0:
                                cur.execute("SELECT " + impurity_function + " (fraction_ones(" + self.respond_variable + ")) FROM " + self.table + " WHERE " + current_right_split + ";")
                                right_impurity = cur.fetchone()[0]
                                
                                cur.execute("SELECT COUNT(*) FROM " + self.table + " WHERE " + current_right_split + ";")
                                right_data_point = cur.fetchone()[0]
                            
                                right_fraction = right_data_point/self.own_data_point
                                
                                cur.execute("SELECT " + impurity_function + "(fraction_ones(" + self.respond_variable + ")) FROM " + self.table + " WHERE "+current_left_split + ";")
                                left_impurity = cur.fetchone()[0]
                                
                                cur.execute("SELECT COUNT(*) FROM " + self.table + " WHERE " + current_left_split + ";")
                                left_data_point = cur.fetchone()[0]
                                
                                left_fraction = left_data_point/self.own_data_point
                                
                                own_impurity = self.pA1
                            
                            # If the node is not a root node
                            else:
                                
                                # If the current split point has been used, set all values into default
                                if current_right_split in self.prev_split or current_left_split in self.prev_split:
                                    own_impurity = 0
                                    right_fraction = 0
                                    right_impurity = 0
                                    left_fraction = 0
                                    left_impurity = 0
                                    
                                else:
                                    cur.execute("SELECT COUNT(*) FROM " + self.table + " WHERE " + col + ">=" + row + " AND " + " AND ".join(self.prev_split) + ";")
                                    right_data_point = cur.fetchone()[0]
                                    
                                    cur.execute("SELECT COUNT(*) FROM " + self.table + " WHERE " + col + "<" + row + " AND "+ " AND ".join(self.prev_split) + ";")
                                    left_data_point = cur.fetchone()[0]
                                    
                                    # If the trial dataset is not empty
                                    if right_data_point != 0 and left_data_point != 0:
                                        
                                        cur.execute("SELECT " + impurity_function + " (fraction_ones(" + self.respond_variable + ")) FROM " + self.table + " WHERE " + col + ">= " + row +  " AND " + " AND ".join(self.prev_split) + ";")
                                        right_impurity = cur.fetchone()[0]
                                    
                                        right_fraction = right_data_point/self.own_data_point
                                        
                                        cur.execute("SELECT " + impurity_function + "(fraction_ones(" + self.respond_variable + ")) FROM " + self.table + " WHERE "+col + "<" + row + " AND "+" AND ".join(self.prev_split) + ";")
                                        left_impurity = cur.fetchone()[0]
                                        
                                        left_fraction = left_data_point/self.own_data_point
                                        
                                        own_impurity = self.pA1
                                    
                                    # If the trial dataset is empty, then set everything into default
                                    else:
                                        own_impurity = 0
                                        right_fraction = 0
                                        right_impurity = 0
                                        left_fraction = 0
                                        left_impurity = 0
                            
                            impurity_reduction_score = own_impurity - right_fraction*right_impurity - left_fraction*left_impurity
                            
                            # If the maximum reduction impurity is the same as pervious column, then skip
                            if impurity_reduction_score in impurity_reduction_each_col:
                                pass
                            else:
                                impurity_reduction_each_col[impurity_reduction_score] = float(row)
                            
                best_value_for_col = max(impurity_reduction_each_col)
                impurity_reduction.append((best_value_for_col, impurity_reduction_each_col.get(best_value_for_col)))
                
            # If the number in this column is the same, then skip this column 
            elif col != self.respond_variable and len(target_col.unique()) == 1:
                impurity_reduction.append((0,0))
        
        # Find the maximum reduction column
        for i in impurity_reduction:
            if i[0]>max_redu[0]:
                max_redu=i
    
        # If the split can reduce the impurity, then return the split variables and split values
        if max_redu[0] != 0:
            split_variable = self.predictors[impurity_reduction.index(max_redu)]
            # Change this line if change jump interval
            split_value = max_redu[1]
        
            self.split_variable = split_variable
            self.split_value = split_value
            return split_variable, split_value
        
        # Otherwise, dont split
        else:
            return None, None
    
    def show(self, level=0):
        """Print out the tree in an appealing way."""
    
        print(" " * level, self.split_variable, ": ", self.split_value, sep="")
    
        if self.left_child is not None:
            print(" " * level, "left:", sep="")
            self.left_child.show(level + 2)
    
        if self.right_child is not None:
            print(" " * level, "right:", sep="")
            self.right_child.show(level + 2)
            
 
    def is_valid_helper(self, dataset):
        """
        This function is a helper that used for testing some basic requirements of the tree.
        - No nodes are empty (no dataset)
        - If a node is split on xj, all data points in the left
          child should have xj<s and
          all the points in the right child should be xj >=s
        
        Parameters:
        ----------
        dataset: The dataset used for tree building 
    
        Returns:
        --------
        The length of cummulating all the length in the leaves
        """
        if self.leaf == True:
            return self.get_data_length(dataset)[0]
        else:
            cur.execute("SELECT MAX(" + self.split_variable +") FROM "+ dataset + " WHERE "+ " AND ".join(self.left_child.prev_split) + ";")
            max_left = cur.fetchone()[0]
            assert float(max_left) <self.split_value,"The max left bigger than the split value"
            
            cur.execute("SELECT MIN(" + self.split_variable +") FROM "+ dataset + " WHERE "+ " AND ".join(self.right_child.prev_split) + ";")
            min_right = cur.fetchone()[0]
            assert float(min_right)>=self.split_value,"The min right smaller than the split value"
            
            return self.left_child.is_valid_helper(dataset) + self.right_child.is_valid_helper(dataset)
        
    def is_valid(self, dataset):
        """
        Combine all the dataset in the leaf nodes shoud yield the original  dataset
        
        Parameters:
        -----------
        dataset: The dataset used for tree building
        
        Returns:
        --------
        Wether the tree is valid or not
        """
        total_leaf_length = self.is_valid_helper(dataset)
        if total_leaf_length == self.whole_data_length:
            return True
        else:
            return False

    def get_data_length(self,table):
        """
        Get the length of dataset for is_valid method
        
        Prameters:
        ---------
        table: String
            The name of SQL table
            
        Return: Dataset_length
        """
        cur.execute("SELECT COUNT(*) FROM " + table +" WHERE " + " AND ".join(self.prev_split) + ";")
        dataset_length = cur.fetchone()

        return dataset_length
    
 
 
class SQL_Tree(classification_tree.Tree):

    def __init__(self, table , predictors, respond_variable,impurity_function, cur, max_depth, pruning_factor, prev_split = []):
        """
        Build and prun the tree
        
        Parameters:
        -----------
        table: String
            The name of SQL table
            
        predictors: List
            The list of predictors
            
        respond_variable: String
            The name of respond variable
            
        impurity_function: Function
            The function used for impurity calculation
            
        purning_factor: Number
            The factors that used for deicision whether to prun the tree
            
        prev_split: List
            The record of past split
        
        Returns:
        -------
        None
        """
        self.tnode = SQL_node(table, predictors, respond_variable, impurity_function, cur, max_depth, prev_split = [])
        self.tnode = self.tnode.pruning(pruning_factor)
    
    
    def query_tree (self, test_dataset):
        """
        Make prediction based on tree
        
        Parameters:
        -----------
        test_dataset: pd dataframe
            The new dataset used for prediction
            
        Returns:
        ----------
        result: List
            The list of predicting results
        """
        result = []
        for index in range(len(test_dataset)):
            sd = test_dataset.iloc[index,:]
            tree_predict = self.tnode.query_node(self.tnode, sd)
            result.append(tree_predict)
        return result
    
