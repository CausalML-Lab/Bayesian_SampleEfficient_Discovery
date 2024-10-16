# Run sample effecient causal discovery

import numpy as np
import networkx as nx
from juliacall import Main as jl
import pyAgrum
import graphical_models
import matplotlib.pyplot as plt
import subprocess
import copy
import random
import pickle
import argparse
import itertools as itr

from enumerator import *
from examples import *
from Utils import *
from graph_structure import *
from algorithms import sample_efficient

def shanmugam_random_chordal(nnodes, density):
    while True:
        d = nx.DiGraph()
        d.add_nodes_from(set(range(nnodes)))
        order = list(range(1, nnodes))
        for i in order:
            num_parents_i = max(1, np.random.binomial(i, density))
            parents_i = random.sample(list(range(i)), num_parents_i)
            d.add_edges_from({(p, i) for p in parents_i})
        for i in reversed(order):
            for j, k in itr.combinations(d.predecessors(i), 2):
                d.add_edge(min(j, k), max(j, k))

        perm = np.random.permutation(list(range(nnodes)))
        d = nx.relabel.relabel_nodes(d, dict(enumerate(perm)))

        return d

# Load your Julia script
jl.seval('include("MEC_utils.jl")')

# Fix random seed
seed = 42
pyAgrum.initRandom(seed=seed)
np.random.seed(seed)
random.seed(seed)

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Sample efficient')

# Add arguments
parser.add_argument('--n', type=int, default=5, help='n of nodes')
parser.add_argument('--den', type=float, default=0.1, help='density')
parser.add_argument('--n_dag', type=int, default=50, help='n of DAGs')
parser.add_argument('--n_sample', type=int, default=100000, help='n of samples')
parser.add_argument('--style', type=str, default='sep', help='atomic or sep sys')

# Parse arguments
args = parser.parse_args()

# Access the value of the argument
N_node = args.n
density = args.den
n_data_samples = args.n_sample
n_dag = args.n_dag
style = args.style

shd_mat = np.zeros((n_dag, n_data_samples))
i = 0
while i < n_dag:
    d = shanmugam_random_chordal(N_node, density)
    if len(nx.degree_histogram(d)) > 5:
        continue
    print('**************')
    print('{} out of {}'.format(i+1, n_dag))
    # Convert the DAG to BN
    BN = nxDAG_to_BN(d)
    BN.generateCPTs()
    name_id_map = BN_names_to_id_map(BN)
    id_name_map = BN_id_to_names_map(BN)
    shd = sample_efficient(d, BN, n_data_samples, style = style)
    shd_mat[i, :] = shd
    i = i + 1

with open('.\ours\sampleE_mini_{}_{}_data.pickle'.format(N_node, density), 'wb') as f:
    # Serialize and save the variables
    pickle.dump(shd_mat, f)
