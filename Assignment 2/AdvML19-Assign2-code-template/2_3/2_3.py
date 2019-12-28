#This file was built upon the the solution template for question 2.3 in DD2434 - Assignment 2.

import numpy as np
from Tree import Tree
import math, time

def calculate_s(theta, beta, tree, node):
    """
    This function calculates "smaller" part in the recursive formula (see report).
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :param: tree: The tree object
    :param: node: The current node to calculate for. Type: int.
    :return: s: The smaller value for the node. Type: numpy array. Dimensions: (K, )
    """
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

def calculate_likelihood_recursive(theta, beta, tree):
    """
    This function calculates the likelihood of a sample of leaves with pure recursion.
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :param: tree: The tree object
    :return: likelihood: The likelihood of beta. Type: float.
    """

    print("Calculating the likelihood...")
    s_root = calculate_s(theta, beta, tree, 0)

    likelihood = theta[0].dot(s_root)

    return likelihood

def calculate_likelihood(theta, beta, tree):
    """
    This function calculates the likelihood of a sample of leaves with Dynamic programming.
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :param: tree: The tree object
    :return: likelihood: The likelihood of beta. Type: float.
    """

    nr_categories = theta[0].size
    tree_topology = tree.get_topology_array()
    print("Calculating the likelihood...")

    # where all s values are stored
    s = np.zeros((tree_topology.size, nr_categories))  # (nodes, categories)

    # all leaves in tree
    nodes_to_compute = np.argwhere(np.isfinite(beta)).flatten()

    while nodes_to_compute.size > 0:
        node = nodes_to_compute[-1]  # last element in array
        nodes_parent = tree_topology[node].astype(int)

        # if leaf
        if not math.isnan(beta[node]):
            beta_value = beta[node].astype(int)

            temp_s = np.zeros(nr_categories)
            temp_s[beta_value] = 1
            s[node] = temp_s
        # if is parent
        else:
            nodes_children = tree.get_children_array_of(node)
            factors = []

            for child in np.nditer(nodes_children):
                child_CPD = get_matrix(theta[child])
                factors.append(child_CPD.dot(s[child]))

            if len(factors) > 1:
                # elementwise multiplication
                s[node] = np.multiply(factors[0], factors[1])
            else:
                s[node] = factors[0]

        # adding parent to queue if not already there and if not current node is root
        if nodes_parent not in nodes_to_compute and nodes_parent >= 0:
            nodes_to_compute = np.insert(nodes_to_compute, 0, nodes_parent)
        # remove computed node from queue
        nodes_to_compute = nodes_to_compute[:-1]

    likelihood = theta[0].dot(s[0])
    return likelihood


def get_matrix(mat):
    """
    This function converts a np.array of arrays to a format which we can calculate with.
    :param: mat: Type: numpy array. Dimensions: (K, K)

    :return: matrix: A non-nested matrix of "standard format". Type: numpy array. Dimensions: (K, K)
    """
    matrix = np.zeros((mat.size, mat.size))
    for index, array in enumerate(mat):
        matrix[index] = array
    return matrix

def test_prob_sum(tree):
    """
    This function calculates the sum of the probabilities for all possible node assignments (betas) and prints it.
    :param: tree: Type: object.
    """
    all_possible_betas = tree.generate_all_possible_betas()

    prob_sum = 0
    for beta_sample in all_possible_betas:
        sample_likelihood = calculate_likelihood(tree.get_theta_array(), beta_sample, tree)
        prob_sum += sample_likelihood

    print("Probability sum: ", prob_sum)

def generate_tree(filename):
    """
    This function generates a probabilistic binary tree, some filtered samples and then saves it.
    :param: filename: Type: string.
    """
    tree = Tree()
    tree.create_random_binary_tree(10, 6, 6)
    tree.sample_tree(2)
    tree.save_tree(filename)

def main():

    print("\n1. Load tree data from file and print it\n")

    filename = {"small_test":"data/q2_3_small_test_tree.pkl", "small":"data/q2_3_small_tree.pkl", "medium": "data/q2_3_medium_tree.pkl", "large": "data/q2_3_large_tree.pkl"}
    tree = Tree()
    tree.load_tree(filename["large"])
    tree.print()

    print("tree topology: ", tree.get_topology_array())

    print("\n2. Calculate likelihood of each FILTERED sample\n")

    #Testing if probability of all possible beta values sums to 1
    test_prob_sum(tree)

    for sample_idx in range(tree.num_samples):
        beta = tree.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(tree.get_theta_array(), beta, tree)
        print("\tLikelihood: ", sample_likelihood)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))