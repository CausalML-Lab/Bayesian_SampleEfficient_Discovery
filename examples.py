import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# Some example DAGs for testing posteriors
def get_diamond(plot = False):
    demo_D = nx.DiGraph()
    demo_D.add_nodes_from([1, 2, 3, 4])
    demo_D.add_edges_from([(2, 1), (2, 3), (3, 4), (3, 1), (2, 4)])
    if plot:
        nx.draw(demo_D, with_labels=True)
        plt.show()
    return demo_D

def get_K_1_n(n_nodes = 4, plot = True):
    demo_D = nx.DiGraph()
    demo_D.add_nodes_from( list( np.arange(n_nodes) + 1 ) )
    for i in range(2, n_nodes+1):
       demo_D.add_edge(1, i)
    if plot:
        nx.draw(demo_D, with_labels=True)
        plt.show()
    return demo_D

def K5():
    demo_D = nx.DiGraph()
    demo_D.add_nodes_from([1, 2, 3, 4, 5])
    demo_D.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)])
    nx.draw(demo_D, with_labels=True)
    plt.show()
    return demo_D

def Tree():
    demo_D = nx.DiGraph()
    demo_D.add_nodes_from([1, 2, 3, 4, 5, 6])
    demo_D.add_edges_from([(1, 2), (1, 3), (1, 4), (4, 5), (4, 6)])
    nx.draw(demo_D, with_labels=True)
    plt.show()
    return demo_D

def K4():
    demo_D = nx.DiGraph()
    demo_D.add_nodes_from([1, 2, 3, 4])
    demo_D.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
    nx.draw(demo_D, with_labels=True)
    plt.show()
    return demo_D

def get_clique(n_nodes = 10, plot = True):
    demo_D = nx.DiGraph()
    demo_D.add_nodes_from( list( np.arange(n_nodes) + 1 ) )
    for i in range(1, n_nodes):
        for j in range(i+1, n_nodes + 1):
            demo_D.add_edge(i, j)
    if plot:
        nx.draw(demo_D, with_labels=True)
        plt.show()
    return demo_D

def ex_in_paper():
    demo_D = nx.DiGraph()
    demo_D.add_nodes_from([1, 2, 3, 4, 5, 6])
    demo_D.add_edges_from([(2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (3, 1), (3, 4), (3, 5), (3, 6), (5, 4), (5, 6)])
    nx.draw(demo_D, with_labels=True)
    plt.show()
    return demo_D