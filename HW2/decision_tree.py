import os
import graphviz
import numpy as np
import pandas as pd



def partition(x):
        """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector x.
    """
    partitions = {}
    uniqueX = np.unique(x)
    for ux in uniqueX:
        partitions[ux] = np.where(x == ux)[0]
    return partitions


def entropy(y):
    ENT = 0
    classes = np.unique(y)
    for c in classes:
        targetClassCount = 0
        for index in range(0, len(y)):
            if y[index] == c:
                targetClassCount += 1
        p = targetClassCount / len(y)
        ENT += ((-1)*p)*np.log2(p)
    return ENT

def information_gain(x, y):
    pENT = entropy(y)
    partitions = partition(x)
    wcENT = 0
    
    for part in partitions.values():
        cENT = entropy(y[part])
        partW = len(part)/len(y)
        wcENT += partW/cENT
    IG = pENT - wcENT
    return IG


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    # first stopping condition
    if len(np.unique(y)) = 1:
        return np.unique(y)[0]
    
    # second stopping condition
    if attribute_value_pairs = None:
        mostCommonLabel
        mclCount = 0
        for label in np.unique(y):
            currLabelCount = 0
            for obj in y:
                if obj == label:
                    currLabelCount += 1
            if currLabelCount > mclCount:
                mostCommonLabel = label
                mclCount = currLabelCount
        return mostCommonLabel
    
    # third stopping condition
    if depth == max_depth:
        mostCommonLabel
        mclCount = 0
        for label in np.unique(y):
            currLabelCount = 0
            for obj in y:
                if obj == label:
                    currLabelCount += 1
            if currLabelCount > mclCount:
                mostCommonLabel = label
                mclCount = currLabelCount
        return mostCommonLabel
    
    # selecting the next-best attribute-value pair using information_gain()
    bestAVP = []
    maxIG = 0
    for avp in attribute_value_pairs:
        currIG = information_gain(x[:, avp[0]], y)
        if currIG > maxIG:
            bestAVP = avp
            maxIG = currIG
    
    # getting all remaining attribute-value pairs
    otherAVP = []
    for attribute, value in attribute_value_pairs:
        if attribute != bestAVP[0]:
            otherAVP += (attribute, value)
   
    # partitioning on bestAVP
    partitions = partition(x[:, bestAVP[0]])
    
    # recursive call to ID3
    tree = {}
    for value, indices in partition.items():
        if len(indices) == 0:
            mostCommonLabel
            mclCount = 0
            for label in np.unique(y):
                currLabelCount = 0
                for obj in y:
                    if obj == label:
                        currLabelCount += 1
                if currLabelCount > mclCount:
                    mostCommonLabel = label
                    mclCount = currLabelCount
            tree[(bestAVP[0], value, True)] = mostCommonLabel
        else:
            tree[(bestAVP[0], value, True)] = id3(x[indices], y[indices], otherAVP, depth + 1, max_depth)
            
    return tree


def predict_example(x, tree):
    if type(tree) == int:
        return tree
    else:
        attribute = list(tree.keys())[0][0]
        value = list(tree.keys())[0][1]
        if x[attribute] == value:
            return predict_example(x, tree[(attribute, value, True)])
        else:
            return predict_example(x, tree[(attribute, value, False)])


def compute_error(y_true, y_pred):
    incorrect = 0
    n = len(y_pred)
    for true in y_true, pred in y_pred:
        if true != pred:
            incorrect += 1
    error = (1/n)*incorrect
    return error




# END OF ASSIGNMENT




def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    Modify Path to your GraphViz executable if needed. DO NOT MODIFY OTHER PART OF THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    
    #You may modify the following parts as needed
    
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
