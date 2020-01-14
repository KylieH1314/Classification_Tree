import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_breast_cancer

class Node:
    
    """
    This program can classify the dataset successfully with gini, cross entropy
    and bayes.
    However, there are still some bugs exsit in the tree class and random forest,
    which includes pruning and cross validation.
    """
    
    # add y value, max 
    def __init__(self, dataset, respond_variable,impurity_function, whole_data_length = 0, prev_split = []):
        """
        Build classification trees by recursively initailize
        and split each node.

        Parameters
        -----------
        dataset: pandas dataframe
            The dataset used for building the tree
        
        respond_variable: String
            The responde variable in dataframe
        
        impurity_function: Function
            The function used for impurity calculation
            
        whole_data_length: Integer
            The length of the dataset
            
        prev_split: List
            The list of pass split

        Returns
        --------
        None
        """

        self.respond_variable = respond_variable
        self.split_variable = None
        self.split_value = None
        self.right_child = None
        self.left_child = None
        self.pA1 = self.get_pA1(dataset)
        self.leaf = True
        self.prev_split = prev_split
        self.own_data_point = len(dataset)
        
        # Use the majority of the response as the result of the node
        if len(dataset) > 0:
            self.predict = dataset[respond_variable].mode()[0]
        else:
            self.predict = None
        
        # If current node is root node
        if len(self.prev_split) == 0:
            self.whole_data_length = len(dataset)
            
            # Check the type of the variables
            for y in dataset.columns:
                assert dataset[y].dtype == np.float64 or dataset[y].dtype == np.int32 or dataset[y].dtype == np.int64,"The items in the dataframe should be numbers"
        else:
            self.whole_data_length = whole_data_length
        # If the current node is pure and the current dataset is not 1, then keeps splitting
        if self.pA1 != 1 and len(dataset)>1 and self.pA1 != 0:
            self.leaf = False
            self.split_variable, self.split_value=self.find_best_splitting_point(dataset, impurity_function)
            
            # If the split variable and split value are valid
            if self.split_variable is not None and self.split_value is not None:
                left_dataset, right_dataset = self.get_data(dataset,self.split_variable, self.split_value)

                
                prev_split_left = self.prev_split.copy()
                prev_split_left.append((self.split_variable, self.split_value, "L"))
                self.left_child = Node(left_dataset, self.respond_variable, impurity_function,self.whole_data_length,prev_split_left)
                
                prev_split_right = self.prev_split.copy()
                prev_split_right.append((self.split_variable, self.split_value, "R&E"))
                self.right_child = Node(right_dataset,self.respond_variable, impurity_function,self.whole_data_length,prev_split_right)
                
                
    def show(self, level=0):
        """Print out the tree in an appealing way."""

        print(" " * level, self.split_variable, ": ", self.split_value, sep="")

        if self.left_child is not None:
            print(" " * level, "left:", sep="")
            self.left_child.show(level + 2)

        if self.right_child is not None:
            print(" " * level, "right:", sep="")
            self.right_child.show(level + 2)

    def get_data(self, dataset, split_variable, split_value):
        """
        This is function used for splitting the dataset from parent dataset,
        based on the splitting point and splitting value of parent node.

        The right dataset includes the split_varaible column and
        split_point_value row

        Parameters
        -----------

        dataset: pandas dataframe
            Dataframe from parent node

        split_variable: string
            The column name of splitting point

        split_point_value: int
            The index of the first row the splitting value appears

        Returns
        --------
        left_dataset: pandas dataframe
            The dataset for left child

        right dataset: pandas dataframe
            The dataset for right child
        """
        left_dataset = dataset[dataset[split_variable]<split_value]
        right_dataset = dataset[dataset[split_variable]>=split_value]

        return left_dataset, right_dataset

    def get_pA1(self, dataset):
        """
        This function is used to calculate p for each dataset.

        p is the probability that Y = 1 in the dataset.

        Parameters
        -----------
        dataset: pandas dataframe
            Dataframe from parent node

        Returns
        --------
        pA1: The p for this dataset
        """
        if len(dataset.index) !=0:
            y_value = dataset[self.respond_variable]
            count_ones = 0
            for row in y_value:
                assert row == 1 or row == 0, "Y should be either 0 or 1"
                if row == 1:
                    count_ones = count_ones + 1
            pA1 = round(count_ones/y_value.size,2)
            return pA1
        else:
            return 0

    def find_best_splitting_point(self, dataset, impurity_function):
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
        for col in dataset.columns:
            if col != self.respond_variable and len(dataset[col].unique()) != 1:
                impurity_reduction_each_col = dict()
                for row in dataset[col].unique():
                    if row != min(dataset[col].unique()):
                        left, right = self.get_data(dataset,col,row)
                        impurity_reduction_score = self.get_impurity_reduction\
                        (dataset,left,right,impurity_function)
                        impurity_reduction_each_col[impurity_reduction_score] = row
                best_value_for_col = max(impurity_reduction_each_col)
                impurity_reduction.append((best_value_for_col, impurity_reduction_each_col.get(best_value_for_col)))
                
            # If the number in this column is the same, then skip this column 
            elif col != self.respond_variable and len(dataset[col].unique()) == 1:
                impurity_reduction.append((0,0))

        for i in impurity_reduction:
            if i[0]>=max_redu[0]:
                max_redu=i

        if max_redu[0] != 0:
            if dataset.columns[0] != self.respond_variable:
                split_variable = dataset.columns[impurity_reduction.index(max_redu)]
            else:
                split_variable = dataset.columns[1+impurity_reduction.index(max_redu)]
            split_value = max_redu[1]
        
            self.split_variable = split_variable
            self.split_value = split_value
            return split_variable, split_value
        else:
            return None, None

    def get_impurity_reduction(self,dataset, left_possible_dataset,
    right_possible_dataset, impurity_function):
        """
        Use the givien impurity_function to calcualte
        the impurity reduction score IA

        Parameters:
        ----------
        left_possible_dataset: pandas dataframe
            The dataset used for calculate new pA1
             and impurity score

        right_possible_dataset: pandas dataframe
            The dataset used for calcualte new pA1
            and impurity scores

        impurtiy_function: funtion with p
            A funtion to get impurtiy score

        return
        ------
        impurity_reduction score: double
            A number which calculated by
             I(A)- pL*I(AL) - pR*I(AR)
        """
        own_impurity_score = self.get_impurity(dataset, impurity_function)

        left_impurity_score =\
        self.get_impurity(left_possible_dataset, impurity_function)

        right_impurity_score =\
        self.get_impurity(right_possible_dataset, impurity_function)

        left_part_ratio = len(left_possible_dataset)/(len(left_possible_dataset)+len(right_possible_dataset))
        right_part_ratio = len(right_possible_dataset)/(len(left_possible_dataset)+len(right_possible_dataset))
        impurity_reduction_score = \
        own_impurity_score-left_part_ratio*left_impurity_score-right_part_ratio*right_impurity_score

        return impurity_reduction_score

    def get_impurity(self, dataset, impurtiy_function):
        
        """
        This function use the dataset and
        imputiry_function to get I(A)

        Paramters:
        ---------
        dataset: pandas dataframe
            The data used for pA1

        impurity_function: function with p
            A funtion to get impurtiy score

        returns
        ------
        impurity score: number
            I(A)
        """
        pA1 = self.get_pA1(dataset)
        if pA1 == 1 or pA1 == 0:
            impurity_score = 0
        else:
            impurity_score = impurtiy_function(pA1)
            
        return impurity_score

    
    def pruning(self, alpha):
        """
        This function recursively goes down to one of the
        leaves and compare the alpha got from cross validation.
        use search method to get the parent node and do purning.

        Parameters:
        ----------
        node: tree node
            Thie node for replacing current node with children nodes.

        alpha: number
            The number used for decision whether this
            node need to be pruned or not.

        Returns:
        -------
        None
        """
        if self.left_child is not None and self.right_child is not None:
            if self.left_child.leaf == False:
                
                self.left_child.pruning(alpha)
    
            if self.right_child.leaf == False:
    
                self.right_child.pruning(alpha)
    
            if self.right_child.leaf == True and self.left_child.leaf == True:
    
                alpha_star = self.calc()
                if alpha_star <= alpha:
                    self.leaf = True
                    self.right_child = None 
                    self.left_child = None
                else:
                    pass
                
                return
        

    def calc(self):
        '''
        This function caculate the alpha star for the decision of pruning.

        Parameters:
        ----------
        node: tree node
            Current node.

        Returns:
        -------
        alpha_start:
            scores for deciding whether to prune or not.

        '''
        alpha_star = 0
        if self.right_child is not None:
            alpha_star = min(self.right_child.pA1,
            1-self.right_child.pA1) * (self.right_child.own_data_point/self.whole_data_length)
        if self.left_child is not None:
            alpha_star = alpha_star + min(self.left_child.pA1,
            1-self.left_child.pA1) * (self.left_child.own_data_point/self.whole_data_length)
        alpha_star = min(self.pA1,1-self.pA1)*(self.own_data_point/self.whole_data_length)-alpha_star
        return alpha_star

    def query_node(self, data_point):
        """
        This function take a new row which has same column as dataset and
        get the y after following the splitting point and
        splitting value in the tree.

        Parameters:
        ----------
        tree: tree
            The tree that we are trying to query for

        data_point: pandas dataframe with one row
            A new row of dataframe that has same columns as the old dataset

        Return:
        ------
        pA1:
            The outcome of this data point
        """
        if self.right_child is not None and self.left_child is not None:
            if data_point[self.split_variable] < self.split_value:
                return self.left_child.query_node(data_point)
            if data_point[self.split_variable] >= self.split_value:
                return self.right_child.query_node(data_point)
        else:
            return self.predict


    def is_valid_helper(self, dataset):
        """
        Not sure whether this function should be in tree class or not.
    
        This function is used for testing some basic requirements of the tree.
        - No nodes are empty (no dataset)
        - If a node is split on xj, all data points in the left
          child should have xj<s and
          all the points in the right child should be xj >=s
        - Combine all the dataset in the leaf nodes shoud yield
          the original  dataset
        
        Parameters:
        ----------
        tree first classification tree node
    
        Returns:
        --------
        The whole original dataset
        """
        if self.leaf == True:
            return len(self.get_data_1(dataset))
        else:
            assert max(self.left_child.get_data_1(dataset)[self.split_variable])<self.split_value,\
            "The max left bigger than the split value"
            assert min(self.right_child.get_data_1(dataset)[self.split_variable])>=self.split_value,\
            "The max right smaller than the split value"
            return self.left_child.is_valid_helper(dataset) + self.right_child.is_valid_helper(dataset)
        
    def is_valid(self, dataset):
        total_leaf_length = self.is_valid_helper(dataset)
        if total_leaf_length == len(dataset):
            return True
        else:
            return False
            
    def get_data_1(self,dataset):
        own_dataset = pd.DataFrame()
        for variable,value, direction in self.prev_split:
            if direction == "L":
                own_dataset = dataset[dataset[variable]<value]
            elif direction == "R&E":
                own_dataset = dataset[dataset[variable]>=value]
            dataset = own_dataset
        return dataset
            
class Tree:

    def __init__(self, dataset, respond_variable, impurity_function, prev_split = []):
        """
        Build and purn the tree
        
        Parameters
        -----------
        dataset: pandas dataframe
            The dataset used for building the tree
        
        respond_variable: String
            The responde variable in dataframe
        
        impurity_function: Function
            The function used for impurity calculation
            
        whole_data_length: Integer
            The length of the dataset
            
        prev_split: List
            The list of pass split

        Returns
        --------
        None
        """
        self.tnode = Node(dataset, respond_variable, impurity_function, prev_split = [])
        self.pruning(self.tnode,dataset, impurity_function,respond_variable)


    def pruning(self, node, dataset, impurity_function,respond_variable):
        
        """
        Use the pruning factors from cross validation to prun the tree
        
        Parameters:
            node:tree node
                The tree that will be pruned
            
            dataset:pd dataframe
                The dataset that the tree based on
                
            impurity_function: function
                The function used for calculated impurity
        """
        alpha = self.cross_validation(dataset, impurity_function,respond_variable)
        self.tnode.pruning(alpha)

    def cross_validation(self,raw_dataset, impurity_function,respond_variable):
         '''
         This function is used to find the best alpha for pruning.
    
         Parameters:
         ----------
         raw_dataset: pandas dataframe
             orginal dataset without any slicing
    
         Return:
         ------
         alpha: number
             Best alpha for pruning decision
         '''
         step_size = round((len(raw_dataset)+1)/5)
         dataset_1 = raw_dataset[0:step_size]
         dataset_2 = raw_dataset[step_size:(2*step_size)]
         dataset_3 = raw_dataset[(2*step_size):(3*step_size)]
         dataset_4 = raw_dataset[(3*step_size):(4*step_size)]
         dataset_5 = raw_dataset[(4*step_size):]
         dataset_list = [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]
    
         alpha_range = [0.00001, 0.001, 0.1, 0.3, 0.5, 0.8, 0.95]
         error_for_alpha = []
         for a in (alpha_range):
             error_for_tree = []
             for k in range(5):
                 error_count = 0
                 testing = dataset_list[k]
                 mask = raw_dataset.iloc[:,1].isin(testing.iloc[:,1])
                 training = raw_dataset[~mask]
                 training_tree = Node(training,respond_variable, impurity_function)
                 training_tree.pruning(a)
                 for index in range(len(testing)) :
                     sd = testing.iloc[index,:]
                     prediction = training_tree.query_node(sd)
                     if prediction != sd[self.tnode.respond_variable]:
                         error_count = error_count+1
                 error_for_tree.append(error_count)
             error_for_alpha.append(sum(error_for_tree)/len(error_for_tree))
         alpha = alpha_range[error_for_alpha.index(min(error_for_alpha))]
         return alpha
    
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
        
 
    
class tree_for_forest(Node):
    def __init__(self, k, dataset, respond_variable,impurity_function, whole_data_length = 0, prev_split = []):
        self.k = k
        """
        This function overwrite the init for Node class and
        change the way call fin_best_splitting_point(),
        because the parameters  are changed.

        Parameters:
        ----------
        k: int
        The number used for number of columns that
        used for impurity scores

        dataset: pandas dataframe
        The dataset used for the calssification tree
        in forest, which means the dataset
        after random sampled rows.

        Returs:
        -------
        None
        """
        super().__init__(dataset, respond_variable,impurity_function, whole_data_length = 0, prev_split = [])


    def find_best_splitting_point(self, dataset, impurity_function):
        """
        This function overwrite the find_best_splitting_point
        function in Node class.

        This function is used to iterate each potential splitting value
        for each column and return the splitting point and value that
        has max impurity reduction. The only difference between this one and
        the one in Node class is that it use the random sampled subset to
        find the relatively best splitting value and splitting variables.

        Iterate though the rows, find the maximum p value for each varaible,
        then find the variable that has the max p value.
        Then we need to access the p value
        of the children to find the maximum delta IA

        Parameters
        ----------
        impurity_function: function with p
            A function with p to get impurity score and impurity reduction.

        k: int
        The number of columns that needed for impurity calculation.

        Returns
        --------
        split_variable: string
            A string of the column name of splitting point.

        split_point_value: index
            A index of the row that first contain the splitting value.
        """
        sub_dataset=dataset.loc[:,dataset.columns != self.respond_variable]
        sub_dataset = sub_dataset.sample(self.k, axis=1)
        sub_dataset[self.respond_variable] = dataset[self.respond_variable]
        impurity_reduction = []
        max_redu = (0,0)
        for col in dataset.columns:
            if col != self.respond_variable and len(dataset[col].unique()) != 1:
                impurity_reduction_each_col = dict()
                for row in dataset[col].unique():
                    if row != min(dataset[col].unique()):
                        left, right = self.get_data(dataset,col,row)
                        impurity_reduction_score = self.get_impurity_reduction\
                        (dataset,left,right,impurity_function)
                        impurity_reduction_each_col[impurity_reduction_score] = row
                best_value_for_col = max(impurity_reduction_each_col)
                impurity_reduction.append((best_value_for_col, impurity_reduction_each_col.get(best_value_for_col)))
            elif col != self.respond_variable and len(dataset[col].unique()) == 1:
                impurity_reduction.append((0,0))

        for i in impurity_reduction:
            if i[0]>=max_redu[0]:
                max_redu=i

        if max_redu[0] != 0:
            if dataset.columns[0] != self.respond_variable:
                split_variable = dataset.columns[impurity_reduction.index(max_redu)]
            else:
                split_variable = dataset.columns[1+impurity_reduction.index(max_redu)]
            split_value = max_redu[1]
        
            self.split_variable = split_variable
            self.split_value = split_value
            return split_variable, split_value
        else:
            return None, None
    
class random_forest():
    def __init__(self, train_dataset, k , n, respond_variable, impurity_function):
        """
        This function is used for generating forest.

        This function build n trees with random sample rows,
        by calling tree_for_forest class.

        Parameters:
        ----------
        k: int
        Number of column the user wants for each tree to do impurity calcultion

        n: int
        Number of tree the user wants for the forest

        Returns:
        --------
        forest: list
        A list of tree nodes.
        """
        nrow, ncol = train_dataset.shape
        self.forest = []
        for i in range(n):
            random_size=random.randint(1, nrow)
            random_sample=train_dataset.sample(n=random_size)
            tree = tree_for_forest(k, random_sample, respond_variable,impurity_function)
            self.forest.append(tree)
            
    def predict(self, test_dataset):
        result_in_forest = []
        for i in range(len(self.forest)):
            result_each_tree = []
            for j in range(len(test_dataset)):
                prediction = self.forest[i].query_node(self.forest[i],test_dataset.iloc[j])
                result_each_tree.append(prediction)
            result_in_forest.append(result_each_tree)
        result_in_forest = pd.DataFrame(result_in_forest)
        result_avg_forest = result_in_forest.mode(axis = 0)
        print(result_avg_forest)
        return result_avg_forest
        
  
# Impurity function 
def gini(pA1):
    gini_score = pA1*(1-pA1)
    return gini_score

def cross_entropy(pA1):
    cross_entropy_score = -pA1*np.log(1-pA1)-(1-pA1)*np.log(1-pA1)
    return cross_entropy_score

def bayers_error(pA1):
    bayers_score = min(pA1, 1-pA1)
    return bayers_score
'''
data = load_breast_cancer()

data_d = pd.DataFrame(data.data[:,[0,1,2,3,4]])
data_d = data_d.head(50)
data_t = pd.DataFrame(data.target)
data_t = data_t.head(50)
data_1 = pd.concat((data_d,data_t), axis = 1)

feature_names = ["col1","col2","col3","col4","col5","col6"]
data_1.columns = np.asarray(feature_names)

#tree=Tree(data_1,'col6',gini)
rf = random_forest(data_1, 2 , 50, 'col6', gini)
'''