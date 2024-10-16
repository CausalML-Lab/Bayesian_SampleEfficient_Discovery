# Check the posterior with different DAG sample numbers

import numpy as np
import networkx as nx
import pyAgrum
import graphical_models
from juliacall import Main as jl
import matplotlib.pyplot as plt
import subprocess
import copy

from enumerator import *
from examples import *
from Utils import *

# Fix random seed
pyAgrum.initRandom(seed=42)

# Load your Julia script
jl.seval('include("MEC_utils.jl")')

# Create a demo DAG
demo_D = shanmugam_random_chordal(5, 1)
print(demo_D.edges())
#demo_D = get_clique(10, plot = False)
#demo_D = get_diamond(plot=False)
#demo_D = Tree()
#demo_D = get_K_1_n(n_nodes=3, plot=True)
#demo_D = nx.DiGraph()
#demo_D.add_nodes_from([1,2,3])
#demo_D.add_edges_from([(3,1), (3,2), (1,2)])

# Convert the DAG to BN
BN = nxDAG_to_BN(demo_D)
BN.generateCPTs()
name_id_map = BN_names_to_id_map(BN)
id_name_map = BN_id_to_names_map(BN)

# Create a skeleton (PDAG) which we orient later
D_skeleton = skeleton_from_DAG(demo_D)
UCCG = D_skeleton.to_nx()
sep_sys = chordal_graph_separating_sys(UCCG)

# Get the joint prob of BN
joint_prob_obs = get_joint_prob(BN)

sep_class = []
# Compare posterior for each sep set
for sep_id in list(sep_sys.keys()):
    sep_set = sep_sys[sep_id]
    P_prior, P_posterior_class_do_v, P_likelihood_class_do_v = get_Prior_of_sep_set(jl,
                                                                UCCG, sep_set)

    # Get the modified BN
    sep_set_names = nodeset_to_nameset(sep_set)
    BN_v_int = get_intervened_BN(BN, sep_set_names)

    for i_x in range(n_data_samples):
        # Generate a sample from BN_v_int
        x = gum.generateSample(bn=BN_v_int, n=1)

        # For each class, calculate the likelihood
        P_likelihood_class_do_v = get_sep_set_class_likelihood(jl, x[0], BN, joint_prob_obs,
                                                               UCCG, sep_set,
                                                               P_likelihood_class_do_v)

        # Normalize posterior, update prior
        P_posterior_class_do_v, P_prior = Normalize_posterior(P_posterior_class_do_v,
                                                              P_likelihood_class_do_v, P_prior, False)

    # Record the class with highest posterior
    max_class = max(P_posterior_class_do_v, key=P_posterior_class_do_v.get)
    sep_class.append(max_class)
    print(max_class)

# Combine sep sets to orient the graph