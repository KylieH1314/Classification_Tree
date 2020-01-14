import pytest
import SQL
import psycopg2
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

# the fixture test function is not working for now
@pytest.fixture(scope = "module")
def psycogy_connection():
    
    conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database="zhiyanh",
                user="zhiyanh", password="ahY0ieMou")
    cur = conn.cursor()
    return cur,conn

def test_SQL_realistic_example_build_and_prune(psycogy_connection):
    cur,conn = psycogy_connection
    
    data = load_breast_cancer()
    
    data_d = pd.DataFrame(data.data[:,[0,1,2,3,4]])
    data_d = data_d.head(50)
    data_t = pd.DataFrame(data.target)
    data_t = data_t.head(50)
    data_1 = pd.concat((data_d,data_t), axis = 1)
    
    feature_names = ["col1","col2","col3","col4","col5","col6"]
    data_1.columns = np.asarray(feature_names)
    cols = ",".join([str(i) for i in data_1.columns.tolist()])
    
    cur.execute("CREATE TEMP TABLE breast_cancer_1 (col1 numeric, col2 numeric, col3 numeric, col4 numeric,col5 numeric, col6 integer) ON COMMIT DELETE ROWS;")
    cur.execute("BEGIN TRANSACTION;")
    for i,row in data_1.iterrows():
        sql = "INSERT INTO breast_cancer_1 (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cur.execute(sql, tuple(row))
        
    node = SQL.SQL_node("breast_cancer_1", ["col1", "col2","col3","col4","col5"], "col6","gini_index", cur, 5, prev_split = [])
    conn.commit()
    assert node.split_variable == "col3" and node.split_value == 87.5,"The split variable or split value is wrong"
    assert node.left_child.split_variable == "col2" and node.left_child.split_value == 18.66,"The split variable or split value for left child is wrong"
    assert node.right_child.split_variable is None and node.right_child.split_value is None, "The split variable or split value for right child is wrong"
    
    node.pruning(0.8)
    assert node.left_child is None and node.right_child is None, "The pruning is not working right"
    

def test_SQL_is_valid(psycogy_connection):
    cur,conn = psycogy_connection
    
    data=[[1,2,3,1],[2,3,4,1],[3,4,5,1],[4,5,6,0],[5,6,7,1],[6,7,8,0]]
    test_case=pd.DataFrame(data, columns=['Age', 'Tciket', 'Shirt', 'Y'])
    cols = ",".join([str(i) for i in test_case.columns.tolist()])
    
    cur.execute("CREATE TEMP TABLE test_case (Age numeric, Tciket numeric, Shirt numeric, Y integer) ON COMMIT DELETE ROWS;")
    cur.execute("BEGIN TRANSACTION;")
    for i,row in test_case.iterrows():
        sql = "INSERT INTO test_case (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cur.execute(sql, tuple(row))

    node = SQL.SQL_node("test_case", ['Age', 'Tciket', 'Shirt'], 'Y',"gini_index", cur, 10, prev_split = [])
    conn.commit()
    assert node.is_valid("test_case") == True, "The tree should be valid"

    
def test_SQL_impurity_R1(psycogy_connection):
    """
    This test is making a one column case,
    and check its splitting value and splitting point
    """
    cur,conn = psycogy_connection
    
    data=[[1,1],[2,1],[3,1],[4,0]]
    test_case=pd.DataFrame(data, columns=['Age', 'Y'])
    cols = ",".join([str(i) for i in test_case.columns.tolist()])
    
    cur.execute("CREATE TEMP TABLE breast_cancer_3 (Age numeric,  Y integer) ON COMMIT DELETE ROWS;")
    cur.execute("BEGIN TRANSACTION;")
    for i,row in test_case.iterrows():
        sql = "INSERT INTO breast_cancer_3 (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cur.execute(sql, tuple(row))
    
    node = SQL.SQL_node("breast_cancer_3", ['Age'], 'Y',"gini_index", cur, 10, prev_split = [])
    conn.commit()
    
    assert node.split_value == 4 and node.split_variable == 'Age', "The splitting point and variables are wrong"
    
def test_SQL_impurity_R2(psycogy_connection):
    """
    This test is making a two columns case,
    and check its splitting value and splitting point
    """
    cur,conn = psycogy_connection
    
    data=[[1,2,1],[2,3,0],[3,4,1],[4,5,1]]
    test_case=pd.DataFrame(data, columns=['Age', 'Ticket', 'Y'])
    cols = ",".join([str(i) for i in test_case.columns.tolist()])
    
    cur.execute("CREATE TEMP TABLE breast_cancer_4 (Age numeric, Ticket numeric, Y integer) ON COMMIT DELETE ROWS;")
    cur.execute("BEGIN TRANSACTION;")
    for i,row in test_case.iterrows():
        sql = "INSERT INTO breast_cancer_4 (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cur.execute(sql, tuple(row))
    
    node = SQL.SQL_node("breast_cancer_4", ['Age', 'Ticket'], 'Y',"gini_index", cur, 10, prev_split = [])
    conn.commit()
    
    assert node.split_value == 3 and node.split_variable == 'Age', "The splitting point and variables are wrong"
    assert node.left_child.split_value == 2 and node.left_child.split_variable == 'Age', "The splitting point and variables are wrong"
    

def test_SQL_impurity_R3(psycogy_connection):
    """
    This test is making a three columns case,
    and check its splitting value and splitting point
    """
    cur,conn = psycogy_connection
    
    data=[[1,2,3,1],[2,3,4,1],[3,4,5,1],[4,5,6,0],[5,6,7,1],[6,7,8,0]]
    test_case=pd.DataFrame(data, columns=['Age', 'Ticket', 'Shirt', 'Y'])
    cols = ",".join([str(i) for i in test_case.columns.tolist()])
    
    cur.execute("CREATE TEMP TABLE breast_cancer_5 (Age numeric, Ticket numeric,Shirt numeric, Y integer) ON COMMIT DELETE ROWS;")
    cur.execute("BEGIN TRANSACTION;")
    for i,row in test_case.iterrows():
        sql = "INSERT INTO breast_cancer_5 (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cur.execute(sql, tuple(row))
    
    node = SQL.SQL_node("breast_cancer_5", ['Age', 'Ticket','Shirt'], 'Y',"gini_index", cur, 10, prev_split = [])
    conn.commit()
    
    assert node.split_value == 4 and node.split_variable == 'Age', "The splitting point and variables are wrong"
    assert node.right_child.split_value == 6 and node.right_child.split_variable == 'Age', "The splitting point and variables are wrong"
    assert node.right_child.left_child.split_value == 5 and node.right_child.left_child.split_variable == 'Age',\
     "The splitting point and variables are wrong"
     

def test_same_value(psycogy_connection):
    """
    This function is to test how the program handle the same age value.
    """
    cur,conn = psycogy_connection
    
    data=[[1, 10, 30, 1], [2, 10, 100, 0], [3, 10, 10, 1],[4, 10, 15, 1]]
    test_case=pd.DataFrame(data, columns=['Name', 'Age', 'Ticket', 'Y'])
    cols = ",".join([str(i) for i in test_case.columns.tolist()])
    
    cur.execute("CREATE TEMP TABLE breast_cancer_6 (Name numeric, Age numeric,Ticket numeric, Y integer) ON COMMIT DELETE ROWS;")
    cur.execute("BEGIN TRANSACTION;")
    for i,row in test_case.iterrows():
        sql = "INSERT INTO breast_cancer_6 (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cur.execute(sql, tuple(row))
    
    node = SQL.SQL_node("breast_cancer_6", ['Name', 'Age', 'Ticket'], 'Y',"gini_index", cur, 10, prev_split = [])
    conn.commit()
    
    assert node.split_variable == "Ticket"," The Age column shoud be ignored"
    

def test_same_respond(psycogy_connection):
    """
    This function is to test how the tree handle the same y values.
    """
    cur,conn = psycogy_connection
    
    test_case=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    test_case["Y"]=[0] * 100
    cols = ",".join([str(i) for i in test_case.columns.tolist()])
    
    cur.execute("CREATE TEMP TABLE breast_cancer_7 (A numeric, B numeric,C numeric, D numeric, Y integer) ON COMMIT DELETE ROWS;")
    cur.execute("BEGIN TRANSACTION;")
    for i,row in test_case.iterrows():
        sql = "INSERT INTO breast_cancer_7 (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cur.execute(sql, tuple(row))
        
    node = SQL.SQL_node("breast_cancer_7", ['A', 'B', 'C', 'D'], 'Y',"gini_index", cur, 10, prev_split = [])
    conn.commit()
    assert node.left_child is None and node.right_child is None,"The tree should only have one node."


def test_missing_value(psycogy_connection):
    """
    This function is to test how the tree handle the missing value.
    My assumption is the program will throw an error if it contains
    missing value.
    """
    cur,conn = psycogy_connection
    data=[[1,2,3,1],[2,3,4,1],[3,4,5,1],[4,5,6,0],[5,6,7,1],[6,7,8,0]]
    test_case=pd.DataFrame(data, columns=['Age', 'Tciket', 'Shirt', 'Y'])

    cols = ",".join([str(i) for i in test_case.columns.tolist()])
    
    cur.execute("CREATE TEMP TABLE breast_cancer_8 (Age numeric, Tciket numeric,Shirt numeric, Y integer) ON COMMIT DELETE ROWS;")
    cur.execute("BEGIN TRANSACTION;")
    for i,row in test_case.iterrows():
        sql = "INSERT INTO breast_cancer_8 (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cur.execute(sql, tuple(row))
    
    cur.execute("UPDATE breast_cancer_8 SET Age = NULL WHERE Age =1;")
    
    with pytest.raises(AssertionError):
        SQL.SQL_node("breast_cancer_8", ['Age', 'Tciket', 'Shirt'], 'Y',"gini_index", cur, 10, prev_split = [])
    conn.commit()

    
def test_SQL_binary(psycogy_connection):
    """
    This test is to ensure the import format is as expected,
    by having a not binary y value.
    """
    cur,conn = psycogy_connection
    data=[[1,2,3,1],[2,3,4,1],[3,4,5,1],[4,5,6,3],[5,6,7,1],[6,7,8,0]]
    test_case=pd.DataFrame(data, columns=['Age', 'Tciket', 'Shirt', 'Y'])
    cols = ",".join([str(i) for i in test_case.columns.tolist()])
    
    cur.execute("CREATE TEMP TABLE breast_cancer_9 (Age numeric, Tciket numeric,Shirt numeric, Y integer) ON COMMIT DELETE ROWS;")
    cur.execute("BEGIN TRANSACTION;")
    
    for i,row in test_case.iterrows():
        sql = "INSERT INTO breast_cancer_9 (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cur.execute(sql, tuple(row))
    
    with pytest.raises(AssertionError):
        SQL.SQL_node("breast_cancer_9", ['Age', 'Tciket', 'Shirt'], 'Y',"gini_index", cur, 10, prev_split = [])
    conn.commit()
