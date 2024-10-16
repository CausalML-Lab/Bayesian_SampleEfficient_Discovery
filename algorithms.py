# Algorithms for sample effecient causal discovery

import numpy as np
import networkx as nx
import pyAgrum
import graphical_models
from juliacall import Main as jl
import matplotlib.pyplot as plt
import subprocess
import copy
import random
import itertools as itr

from enumerator import *
from examples import *
from Utils import *
from graph_structure import *

# Load your Julia script
jl.seval('include("MEC_utils.jl")')

def sample_efficient(UCCG:nx.DiGraph, BN:gum.BayesNet, n_data_samples, style = 'sep'):
    # Initial shd array
    shd = np.zeros((n_data_samples))
    eps = 0.1

    # Create a skeleton (PDAG) which we orient later
    D_skeleton = skeleton_from_DAG(UCCG)
    UCCG = D_skeleton.to_nx()
    if style == 'sep':
        sep_sys = chordal_graph_separating_sys(UCCG)
    elif style == 'atomic':
        sep_sys = atomic_graph_sep(UCCG)

    # Get the joint prob of BN
    joint_prob_obs = get_joint_prob(BN)

    # Get the initial prior, posterior, likelihood for each sep set
    sep_set_tracking = initialize_sep_set(jl, BN, joint_prob_obs, UCCG, sep_sys)

    # Get interventional samples
    target_sets = sep_sys.keys()
    for i in range(n_data_samples):
        # Choose a random target set from sep sys
        rand_set_id = random.choice( list(target_sets) )
        sep_set = sep_sys[rand_set_id]
        P_prior, P_likelihood_class_do_v, P_posterior_class_do_v, P_int = sep_set_tracking[frozenset(sep_set)]

        # Get the modified BN
        sep_set_names = nodeset_to_nameset(sep_set)
        BN_v_int = get_intervened_BN(BN, sep_set_names)

        # Generate a sample from BN_v_int
        x = gum.generateSample(bn=BN_v_int, n=1)

        # For each class, calculate the likelihood
        P_likelihood_class_do_v = get_sample_likelihood_from_P_int(x[0], P_int, P_likelihood_class_do_v)
        #P_likelihood_class_do_v = get_sep_set_class_likelihood(jl, x[0], BN, joint_prob_obs,
        #                                                       UCCG, sep_set,
        #                                                       P_likelihood_class_do_v)

        # Normalize posterior, update prior
        P_posterior_class_do_v, P_prior = Normalize_posterior(P_posterior_class_do_v,
                                                              P_likelihood_class_do_v, P_prior, False)

        # Update tracking
        sep_set_tracking[frozenset(sep_set)] = (P_prior, P_likelihood_class_do_v, P_posterior_class_do_v, P_int)

        # Find the DAG and calculate shd
        step = 50
        if np.mod(i, step) == 0:
            DAG_pred = decision_rule(UCCG, sep_set_tracking)
            shd[i:i + step] = len( BN.arcs() - DAG_pred.arcs )
        # Stop early
        if max(P_posterior_class_do_v.values()) > 1 - eps:
            target_sets = target_sets - set([rand_set_id])
        if len(target_sets) == 0:
            break
        # Show progress
        if np.mod(i, 5000) == 0:
            print('{} out of {}'.format(i+1, n_data_samples))

    return shd


def sample_efficient_on_large(UCCG:nx.DiGraph, BN:gum.BayesNet, n_data_samples, style = 'sep'):
    # Initial shd array
    shd = np.zeros((n_data_samples))
    eps = 0.1

    # Create a skeleton (PDAG) which we orient later
    D_skeleton = skeleton_from_DAG(UCCG)
    UCCG = D_skeleton.to_nx()
    if style == 'sep':
        sep_sys = chordal_graph_separating_sys(UCCG)
    elif style == 'atomic':
        sep_sys = atomic_graph_sep(UCCG)

    # Get the joint prob of BN
    joint_prob_obs = get_joint_prob(BN)

    i_total_sample = 0
    n_sample_per_node = n_data_samples/len(sep_sys)
    step = 10

    # For each node initialize with prior, posterior, likelihood
    for v_set in list(sep_sys.values()):
        i_total_sample += 1
        P_prior, P_likelihood_class_do_v, P_posterior_class_do_v, P_int = get_Prior_of_sep_set(jl, BN,
                                                                                joint_prob, UCCG, v_set)
        j_set_sample = 0
        # Get the modified BN
        sep_set_names = nodeset_to_nameset(sep_set)
        BN_v_int = get_intervened_BN(BN, sep_set_names)

        while j_set_sample <= n_sample_per_node:
            j_set_sample += 1
            # Generate a sample from BN_v_int
            x = gum.generateSample(bn=BN_v_int, n=1)

            # For each class, calculate the likelihood
            P_likelihood_class_do_v = get_sample_likelihood_from_P_int(x[0], P_int, P_likelihood_class_do_v)

            # Normalize posterior, update prior
            P_posterior_class_do_v, P_prior = Normalize_posterior(P_posterior_class_do_v,
                                                                  P_likelihood_class_do_v, P_prior, False)

            # Find the DAG and calculate shd
            if np.mod(i_total_sample, step) == 0:
                DAG_pred = decision_rule(UCCG, sep_set_tracking)
                shd[i: -1] = len( BN.arcs() - DAG_pred.arcs )
            # Stop early
            if max(P_posterior_class_do_v.values()) > 1 - eps:
                continue
            if len(target_sets) == 0:
                break
            # Show progress
            if np.mod(i, 500) == 0:
                print('{} out of {}'.format(i_total_sample+1, n_data_samples))

    return shd

def initialize_sep_set(jl, BN, joint_prob, UCCG, sep_sys):
    sep_set_tracking = {}
    for set_key in list(sep_sys.keys()):
        P_prior, P_likelihood, P_posterior, P_int = get_Prior_of_sep_set(jl, BN, joint_prob, UCCG, sep_sys[set_key])
        sep_set_tracking[frozenset(sep_sys[set_key])] = (P_prior, P_likelihood, P_posterior, P_int)

    return sep_set_tracking

def decision_rule(UCCG:nx.Graph, sep_set_tracking: dict):
    PDAG = graphical_models.PDAG.from_nx(UCCG)
    # Collect all posteriors of each configure
    all_post = {}
    for sep_set in list(sep_set_tracking.keys() ):
        all_post.update( sep_set_tracking[sep_set][2] )
    # While PDAG is not fully oriented, find the highest configure
    while PDAG.edges != set():
        max_configure = max(all_post, key=all_post.get)
        # If the configure is valid, remove the set from all posteriors, orient the arcs
        if check_valid(PDAG, max_configure):
            del all_post[max_configure]
            PDAG = orient_configure(PDAG, max_configure)
        # Else Remove this configure from all post
        else:
            del all_post[max_configure]

        # If no valid ocnfigure, randomly orient
        if len(all_post) == 0:
            for edge in list(PDAG.edges):
                PDAG.replace_edge_with_arc(edge)

    return PDAG


def check_valid(PDAG:graphical_models.PDAG, configure:str):
    nodes_l = configure.split(';')
    flag = True
    for node_conf in nodes_l:
        node = int(node_conf.split('|')[0])
        for i, neighbor in enumerate( list( PDAG.neighbors_of(node) ) ):
            if node_conf.split('|')[1][i] == '0' and (node, neighbor) in PDAG.arcs:
                flag = False
                return flag
            elif node_conf.split('|')[1][i] == '1' and (neighbor, node) in PDAG.arcs:
                flag = False
                return flag

    return flag

def orient_configure(PDAG, configure):
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

