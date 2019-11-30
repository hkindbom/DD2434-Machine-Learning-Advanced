""" This file is created as the solution template for question 2.3 in DD2434 - Assignment 2.

    Please keep the fixed parameters in the function templates as is (in 2_3.py file).
    However if you need, you can add parameters as default parameters.
    i.e.
    Function template: def calculate_likelihood(tree_topology, theta, beta):
    You can change it to: def calculate_likelihood(tree_topology, theta, beta, new_param_1=[], new_param_2=123):

    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_3_small_tree, q_2_3_medium_tree, q_2_3_large_tree).
    Each tree have 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.
"""

import numpy as np
from Tree import Tree
from Tree import Node
import math

def calculate_s(theta, beta, tree, node):
    nr_categories = theta[0].size
    s = np.zeros(nr_categories)

    # if leaf
    if not math.isnan(beta[node]):
        beta_value = beta[node].astype(int)
        s[beta_value] = 1
        return s

    # if node is parent
    nodes_children = tree.get_children_array_of(node)
    factors = []
    for child in np.nditer(nodes_children):
        #converting to suitable format
        child_CPD = get_matrix(theta[child])

        s_child = calculate_s(theta, beta, tree, child)
        factors.append(child_CPD.dot(s_child))

    if len(factors) > 1:
        s = np.multiply(factors[0], factors[1])  # elementwise multiplication
    else:
        s = factors[0]
    return s

def calculate_likelihood(theta, beta, tree):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    You can change the function signature and add new parameters. Add them as parameters with some default values.
    i.e.
    Function template: def calculate_likelihood(tree_topology, theta, beta):
    You can change it to: def calculate_likelihood(tree_topology, theta, beta, new_param_1=[], new_param_2=123):
    """

    print("Calculating the likelihood...")
    s_root = calculate_s(theta, beta, tree, 0)

    likelihood = theta[0].dot(s_root)

    return likelihood

#converting to format which we can calculate with
def get_matrix(mat):
    matrix = np.zeros((mat.size, mat.size))
    for index, array in enumerate(mat):
        matrix[index] = array
    return matrix

def main():

    print("\n1. Load tree data from file and print it\n")

    filename = {"small":"data/q2_3_small_tree.pkl", "medium": "data/q2_3_medium_tree.pkl", "large": "data/q2_3_large_tree.pkl"}
    tree = Tree()
    tree.load_tree(filename["small"])
    tree.print()

    print("tree topology: ", tree.get_topology_array())

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(tree.num_samples):
        # beta is an assignment of values to all the leaves of the tree.
        beta = tree.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(tree.get_theta_array(), beta, tree)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":

    main()
