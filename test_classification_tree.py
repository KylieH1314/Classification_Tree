import pytest
import classification_tree
import pandas as pd
import numpy as np
"""
I got comment about the test is not detailed enough. Can you specify on that?
"""

def test_impurity_function():
    """
    This function check how the impurity function handle
    the dataset with p that equals to 0.
    """
    data=[[1,2,3,4,0],[2,3,4,6,0],[3,5,6,7,0]]
    test_case=pd.DataFrame(data, columns=['Name', 'Age', 'Ticket Fee', 'Shirt Size', 'Y'])
    tree=classification_tree.Node(test_case,'Y', classification_tree.gini)
    assert tree.get_impurity(test_case,classification_tree.gini) == 0, "The impurity score should be zero"


def test_impurity_R1():
    """
    This test is making a one column case,
    and check its splitting value and splitting point
    """
    data=[[1,1],[2,1],[3,1],[4,0]]
    test_case=pd.DataFrame(data, columns=['Age', 'Y'])
    tree=classification_tree.Node(test_case,'Y', classification_tree.bayers_error)
    assert tree.split_value == 4 and tree.split_variable == 'Age', "The splitting point and variables are wrong"

def test_impurity_R2():
    """
    This test is making a two columns case,
    and check its splitting value and splitting point
    """
    data=[[1,2,1],[2,3,0],[3,4,1],[4,5,1]]
    test_case=pd.DataFrame(data, columns=['Age', 'Ticket Fee', 'Y'])
    tree=classification_tree.Node(test_case,'Y', classification_tree.cross_entropy)
    assert tree.split_value == 4 and tree.split_variable == 'Ticket Fee', "The splitting point and variables are wrong"
    assert tree.left_child.split_value == 3 and tree.left_child.split_variable == 'Ticket Fee', "The splitting point and variables are wrong"

def test_impurity_R3():
    """
    This test is making a three columns case,
    and check its splitting value and splitting point
    """
    data=[[1,2,3,1],[2,3,4,1],[3,4,5,1],[4,5,6,0],[5,6,7,1],[6,7,8,0]]
    test_case=pd.DataFrame(data, columns=['Age', 'Tciket Fee', 'Shirt Size', 'Y'])
    tree=classification_tree.Node(test_case,'Y',classification_tree.gini)
    assert tree.split_value == 6 and tree.split_variable == 'Shirt Size', "The splitting point and variables are wrong"
    assert tree.right_child.split_value == 8 and tree.right_child.split_variable == 'Shirt Size', "The splitting point and variables are wrong"
    assert tree.right_child.left_child.split_value == 7 and tree.right_child.left_child.split_variable == 'Shirt Size',\
     "The splitting point and variables are wrong"

def test_wrong_input_1():
    """
    This function is a test to ensure the import format is as expected
    """
    data=[["Alice", 10, 30, 1], ["Ben", 10, 100, 0], ["Charles", 10, 10, 1],["Don", 10, 15, 1]]
    test_case=pd.DataFrame(data, columns=['Name', 'Age', 'Ticket Fee', 'Y'])
    with pytest.raises(Exception):
        classification_tree.Node(test_case,'Y', classification_tree.gini)

def test_wrong_input_2():
    """
    This test is to ensure the import format is as expected,
    by having a not binary y value.
    """
    data=[[1, 10, 30, 1], [2, 10, 100, 3], [3, 10, 10,1],[4, 10, 15, 1]]
    test_case=pd.DataFrame(data, columns=['Name', 'Age', 'Ticket Fee', 'Y'])
    with pytest.raises(AssertionError):
        classification_tree.Node(test_case,'Y', classification_tree.gini)


def test_same_value():
    """
    This function is to test how the program handle the same age value.
    """
    data=[[1, 10, 30, 1], [2, 10, 100, 0], [3, 10, 10, 1],[4, 10, 15, 1]]
    test_case=pd.DataFrame(data, columns=['Name', 'Age', 'Ticket Fee', 'Y'])
    tree=classification_tree.Node(test_case,'Y', classification_tree.gini)
    assert tree.split_variable == "Ticket Fee",\
    " The Age column shoud be ignored"

def test_same_respond():
    """
    This function is to test how the tree handle the same y values.
    """
    test_case=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    test_case["Y"]=[0] * 100
    tree=classification_tree.Node(test_case,'Y', classification_tree.gini)
    assert tree.left_child is None and tree.right_child is None,\
    "The tree should only have one node."

    test_case["Y"]=[1] * 100
    tree = classification_tree.Node(test_case,'Y', classification_tree.gini)
    assert tree.left_child is None and tree.right_child is None,\
    "The tree should only have one node."

def test_missing_value():
    """
    This function is to test how the tree handle the missing value.
    My assumption is the program will throw an error if it contains
    missing value.
    """
    data=[[1,2,3,1],[2,3,4,1],[3,4,5,1],[4,5,6,0],[5,6,7,1],[6,7,8,0]]
    test_case=pd.DataFrame(data, columns=['Age', 'Tciket Fee', 'Shirt Size', 'Y'])
    test_case[2, 4]=None
    test_case[3, 3]=None
    with pytest.raises(AssertionError):
        classification_tree.Node(test_case,'Y', classification_tree.gini)

        
    

def test_valid_tree ():
    """
    This function is testing whether the tree is built correctly with random dataset.
    And we use is_valid function to test that.
    """
    data=[[1,2,3,1],[2,3,4,1],[3,4,5,1],[4,5,6,0],[5,6,7,1],[6,7,8,0]]
    test_case=pd.DataFrame(data, columns=['Age', 'Tciket Fee', 'Shirt Size', 'Y'])
    tree=classification_tree.Node(test_case,'Y',classification_tree.gini)
    assert tree.is_valid(test_case) == True, "The tree should be valid"

