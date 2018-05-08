
# coding: utf-8

# # Importing Libraries and Datasets

# In[1]:


from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import operator


# In[2]:


df = pd.read_csv('/Users/sidverma/Documents/GitHub/SurvivalRF/scikit-learn/examples/tree/iris.csv')
df.head()


# In[3]:


df.Species = pd.Categorical(df.Species)


# In[4]:


df.shape


# # Class and Function Definitions

# In[5]:


def class_counts(df, column):
    """Counts the number of each type of example in a dataset."""
    counts = collections.Counter(df[column])
    return counts


# In[6]:


class_counts(df, 'Species')


# In[7]:


def isNumeric(value):
    return isinstance(value, int) or isinstance(value, float)

def isCategorical(value):
    return isinstance(value, object) and not isNumeric(value)


# In[8]:


class Splitter(object):
    """A splitting criterion for dividing the dataset based on the outcome
    
    This class forms a new node of a tree where a two outcome Question is asked
    depending on the response the current data is divided into two parts which form the
    remaining data for the left and right subtrees.
    """
    def __init__(self, attribute, operation, target):
        self.attribute = attribute
        self.target = target
        self.operation = operation
        
    def details(self):
        return 'The splitter condition is: '+ self.attribute + " " + self.operation + " " + str(self.target)


# In[9]:


def partition(df, splitter):
    """Partitions a dataset.
    
    @input: DataFrame <pandas.DataFrame>, splitter <object>
    @returns: positive, negative dataframes <pd.DataFrame>
    
    Check whether the row value matches the splitter condition. If it does,
    add it to the matched rows else to the unmatched rows
    """
    comparison = splitter.operation
    attr = splitter.attribute
    target = splitter.target
    
    # For handling Numeric Data
    if isNumeric(target):
        if comparison == '<':
            true_rows = df[df[attr] < target]
            false_rows = df[df[attr] >= target]
        elif comparison == '>':
            true_rows = df[df[attr] > target]
            false_rows = df[df[attr] <= target]
        elif comparison == '==':
            true_rows = df[df[attr] == target]
            false_rows = df[df[attr] != target]
        elif comparison == '!=':
            true_rows = df[df[attr] != target]
            false_rows = df[df[attr] == target]
        elif comparison == '<=':
            true_rows = df[df[attr] <= target]
            false_rows = df[df[attr] > target]
        elif comparison == '>=':
            true_rows = df[df[attr] >= target]
            false_rows = df[df[attr] < target]
        else:
            raise SyntaxError
        return true_rows, false_rows
    
    # For handling Categorical Data
    elif isCategorical(target):
        if comparison == '==':
            true_rows = df[df[attr] == target]
            false_rows = df[df[attr] != target]
        elif comparison == '!=':
            true_rows = df[df[attr] != target]
            false_rows = df[df[attr] == target]
        else:
            raise TypeError
        return true_rows, false_rows
    
    # Erroneous Datatype
    else:
        raise TypeError


# In[10]:


def gini(df, column):
    """Calculate the Gini Impurity for a list of rows.
    
    @input: DataFrame <pandas.DataFrame>, columnLabel <string>
    @returns: gini inpurity <float>
    
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(df, column)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(df.shape[0])
        impurity -= prob_of_lbl**2
    return impurity


# In[11]:


def info_gain(left, right, current_uncertainty, column):
    """Information Gain.
    @input: left, right dataframes <pd.DataFrame>, current uncertainty <float>, class Column name <string>
    @returns: infogain <float>
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(left.shape[0]) / (left.shape[0] + right.shape[0])
    return current_uncertainty - p * gini(left, column) - (1 - p) * gini(right, column)


# In[12]:


def find_best_split(df, column):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain.
    @input: dataframe <pd.DataFrame>, class Column name <string>
    @returns: best_gain <float>, best_splitter <object>
    """
    best_gain = 0  # Keep track of the best information gain
    best_splitter = None  # Keep train of the feature / value that produced it
    current_uncertainty = gini(df, column) # Current gini index of the (parent) node
    
    for attr in df.columns[:-1]:    # For each attribute in the dataset
        values = df[attr].unique()  # List of unique values for each attribute
        if isNumeric(values[0]):
            setOfOperations = ('>', '>=', '<', '<=', '==', '!=')
        if isCategorical(values[0]):
            setOfOperations = ('==', '!=')
            
        # For each unqiue value in list
        for val in values:        
            for operation in setOfOperations:
                
                # Creating new splitter condition
                splitter = Splitter(attr, operation, val) 
                #print("Checking Split Condition: ", splitter.attribute, splitter.operation, splitter.target)
                # Partitioning dataset using splitter
                true_branch, false_branch = partition(df, splitter)
                
                # Skip this split if it doesn't divide the dataset.
                if len(true_branch) == 0 or len(false_branch) == 0: 
                    continue

                # Calculate the information gain from this split
                gain = info_gain(true_branch, false_branch, current_uncertainty, column)
                
                # Selecting the best gain
                if gain >= best_gain:
                    best_gain, best_splitter = gain, splitter

    return best_gain, best_splitter


# In[13]:


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, df, column):
        self.predictions = class_counts(df, column)


# In[14]:


class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the splitter object, and to the two child nodes.
    """

    def __init__(self,
                 splitter,
                 true_branch,
                 false_branch):
        self.splitter = splitter
        self.true_branch = true_branch
        self.false_branch = false_branch


# In[15]:


def build_tree(df, column):
    """Builds the tree.
    @input: dataframe <pd.DataFrame>, class Column name <string>
    @returns: Decision Node <object>
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, splitter = find_best_split(df, column)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(df, column)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(df, splitter)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, column)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, column)

    # Return a Splitter node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(splitter, true_branch, false_branch)


# In[16]:


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the splitter at this node
    print (spacing + str(node.splitter.details()))

    # Call this function recursively on the true branch
    print (spacing + '|--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '|--> False:')
    print_tree(node.false_branch, spacing + "  ")


# In[17]:


# %%time
# myTree = build_tree(df, 'Species')


# In[18]:


# print_tree(myTree)


# # Prediction

# In[19]:


operatorMap = {'>': operator.gt,
               '>=': operator.ge,
               '<': operator.lt,
               '<=': operator.le,
               '==': operator.eq,
               '!=': operator.ne}


# In[20]:


def classify(observation, node):
    """See the 'rules of recursion' above."""
    
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
#     print("Operation: ", node.splitter.operation)
#     print('Row cell val: ', observation[node.splitter.attribute])
#     print("Target val: ", node.splitter.target)
    if operatorMap[node.splitter.operation](observation[node.splitter.attribute], node.splitter.target):
        return classify(observation, node.true_branch)
    else:
        return classify(observation, node.false_branch)


# In[21]:


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = int(counts[lbl] / total * 100)
    return probs


# In[22]:


# for index, obs in df.iterrows():
#     print(obs, index)
#     print('Actual: %s |-|-| Predicted: %s' %(obs[-1], print_leaf(classify(obs, myTree))))

