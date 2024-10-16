# Run sample efficient causal discovery

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

def get_non_adjacent_pair(dag):
    nodes = list(dag.nodes())
    non_adjacent_pairs = [(u, v) for u in nodes for v in nodes if
                          u != v and not dag.has_edge(u, v) and not dag.has_edge(v, u)]

    if not non_adjacent_pairs:
        raise ValueError("No non-adjacent pairs found.")

    return random.choice(non_adjacent_pairs)

def estimate_causal_effect(UCCG:nx.DiGraph, BN:pyAgrum.BayesNet, x, y, n_data_samples):
    # Get ground truth
    x_name= nodeset_to_nameset( set([x]) )
    y_name = nodeset_to_nameset( set([y]) )
    ie = gum.LazyPropagation(BN)
    ie.setEvidence({str(x): 1})
    ie.makeInference()
    # Ground truth of p(y|do(x))
    p_y_do_x = ie.posterior(str(y))

    # Initial div array
    div = np.zeros( n_data_samples )

    # Intervention target
    int_tgt = set( UCCG.predecessors(x) ).union( set(UCCG.successors(x)) )

    # Create a skeleton (PDAG) which we orient later
    D_skeleton = skeleton_from_DAG(UCCG)
    UCCG = D_skeleton.to_nx()

    # Get the joint prob of BN
    joint_prob_obs = get_joint_prob(BN)

    # Counter
    i_total_sample = 0
    step = 10

    # Initialize the configurations' prior, posterior, likelihood
    int_tgt_names = nodeset_to_nameset(int_tgt)
    P_prior, P_likelihood_class_do_v, P_posterior_class_do_v, P_int = get_Prior_of_sep_set(jl, BN,
                                                                                           joint_prob_obs, UCCG, int_tgt)
    BN_v_int = get_intervened_BN(BN, int_tgt_names)

    # Iterate through samples
    while i_total_sample < n_data_samples:
        i_total_sample += 1
        # Generate a sample from BN_v_int
        sample_int = gum.generateSample(bn=BN_v_int, n=1)

        # For each class, calculate the likelihood
        P_likelihood_class_do_v = get_sample_likelihood_from_P_int(sample_int[0], P_int, P_likelihood_class_do_v)

        # Normalize posterior, update prior
        P_posterior_class_do_v, P_prior = Normalize_posterior(P_posterior_class_do_v,
                                                              P_likelihood_class_do_v, P_prior, False)

        # Find the DAG and calculate div
        if np.mod(i_total_sample - 1, step) == 0:
            div_avg = average_divergence(UCCG, P_posterior_class_do_v, BN, joint_prob_obs, x, y, p_y_do_x)
            div[i_total_sample - 1 : -1] = div_avg

        # Show progress
        if np.mod(i_total_sample - 1, 10) == 0:
            print('{} out of {}'.format(i_total_sample, n_data_samples))

    return div

# Calculate the average divergence of the posterior given each configuration
def average_divergence(UCCG, P_posterior_class_do_v, BN, joint_prob, x, y, p_y_do_x):
    div_avg = 0
    for configure in list( P_posterior_class_do_v.keys() ):
        PDAG = orient_configure(UCCG, configure)
        PDAG.to_complete_pdag()
        sample_DAG = DAG_sample_from_CPDAG(jl, PDAG)
        p_y_do_x_pred = causal_effect_from_DAG( sample_DAG, BN, joint_prob, x, y )
        div_avg += KL_divergence(p_y_do_x_pred, p_y_do_x)*P_posterior_class_do_v[configure]

    return div_avg

def KL_divergence(p, q):
    KL_div = p[0]*np.log(p[0]/q[0]) + p[1]*np.log(p[1]/q[1])
    return KL_div

def decision_rule(UCCG:nx.Graph, posterior_set: dict, BN, x_name):
    PDAG = graphical_models.PDAG.from_nx(UCCG)
    max_configure = max(posterior_set, key=posterior_set.get)
    PDAG = orient_configure(PDAG, max_configure)
    CPDAG = PDAG.to_complete_pdag()
    sample_DAG = DAG_sample_from_CPDAG(jl, CPDAG)

    return sample_DAG

def causal_effect_from_DAG( sample_DAG, BN, joint_prob, x, y ):
    # Calculate BN from DAG
    sample_DAG = PDAG_to_nx(sample_DAG)
    BN_sample_DAG = nxDAG_to_BN(sample_DAG)
    # Fill in the CPTs
    for v in list( BN_sample_DAG.nodes() ):
        pa_v = set( sample_DAG.predecessors(v) )
        p_v_give_pa_v = get_conditional_prob(joint_prob,
                                             nodeset_to_nameset(set( [str(v)] )), nodeset_to_nameset(pa_v) )
        BN_sample_DAG.cpt(v).fillWith( p_v_give_pa_v )

    ie_sample = gum.LazyPropagation(BN_sample_DAG)
    ie_sample.setEvidence({str(x): 1})
    ie_sample.makeInference()
    # Ground truth of p(y|do(x))
    p_y_do_x_pred = ie_sample.posterior(str(y))
    return p_y_do_x_pred

def orient_configure(PDAG, configure):
    PDAG = graphical_models.PDAG.from_nx(PDAG)
    nodes_l = configure.split(';')
    for node_conf in nodes_l:
        node = int(node_conf.split('|')[0])
        for i, neighbor in enumerate(list(PDAG.neighbors_of(node))):
            if (node, neighbor) in PDAG.arcs or (neighbor, node) in PDAG.arcs:
                continue
            elif node_conf.split('|')[1][i] == '1':
                PDAG.replace_edge_with_arc((node, neighbor))
            elif node_conf.split('|')[1][i] == '0':
                PDAG.replace_edge_with_arc((neighbor, node))

    return PDAG

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

# Parse arguments
args = parser.parse_args()

# Access the value of the argument
N_node = args.n
density = args.den
n_data_samples = args.n_sample
n_dag = args.n_dag

div_mat = np.zeros((n_dag, n_data_samples))
i = 0
while i < n_dag:
    d = shanmugam_random_chordal(N_node, density)
    if len(nx.degree_histogram(d)) > 5:
        continue
    print('**************')
    x, y = get_non_adjacent_pair(d)
    print('{} out of {}'.format(i+1, n_dag))
    # Convert the DAG to BN
    BN = nxDAG_to_BN(d)
    BN.generateCPTs()
    name_id_map = BN_names_to_id_map(BN)
    id_name_map = BN_id_to_names_map(BN)
    div = estimate_causal_effect(d, BN,x, y, n_data_samples)
    div_mat[i, :] = div
    i = i + 1

with open('.\ours\causal_{}_{}_data.pickle'.format(N_node, density), 'wb') as f:
    # Serialize and save the variables
    pickle.dump(div_mat, f)

err_scale = 10
ours_m = np.mean(div_mat, axis=0)
ours_e = np.std(div_mat, axis=0)/err_scale
xaxis = np.linspace(1, n_data_samples, 1000)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.xlabel('Interventional samples')
plt.ylabel('KL Divergence')
plt.plot(xaxis, ours_m, 'g', label = 'Ours')
plt.fill_between(xaxis, ours_m-ours_e, ours_m+ours_e, alpha = 0.5, color = 'g')
plt.ylim(0, 1)
plt.xlim(0, 100000)
plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
plt.gca().ticklabel_format(useMathText=True)
plt.legend()
plt.grid(True)
plt.savefig('.\ours\case_study_{}_{}.png'.format(N_node, density))








