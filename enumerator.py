import graphical_models
import networkx as nx

from Utils import *
import pyAgrum as gum
from juliacall import Main as jl
import copy
import itertools

# Calculate likelihood
def get_P_likelihood(jl, x, u_rooted_CPDAG:graphical_models.PDAG, BN:gum.BayesNet,
                     int_V:set, joint_prob:gum.Potential,
                     estimator = 'enumerator', n_samples = 50):
    P_likelihood = 0
    if estimator == 'sampler':
        for i in range(n_samples):
            DAG = DAG_sample_from_CPDAG(jl, u_rooted_CPDAG)
            P_likehood +=  P_likelihood_given_DAG(x, DAG, BN,
                                                  joint_prob, int_V)
        P_likelihood = P_likelihood / n_samples

    elif estimator == 'enumerator':
        P_likelihood = P_likelihood_with_enumerator(jl, x, u_rooted_CPDAG, BN,
                                                    int_V, joint_prob)

    else:
        raise NotImplementedError

    return P_likelihood

# Calculate the likelihood through enumerating all the possible orientations over V
def P_likelihood_with_enumerator(jl, x, u_rooted_CPDAG, BN, int_V: set, joint_prob):
    # If neighborhood of V fully oriented, no need to iterate
    if u_rooted_CPDAG.undirected_degree_of(list(int_V)[0])==0:
        sample_DAG = DAG_sample_from_CPDAG(jl, u_rooted_CPDAG)
        P_likelihood = P_likelihood_given_DAG(x, sample_DAG, BN, joint_prob, int_V)
        return P_likelihood

    else:
        # Iterate through all orientations
        n_V_neighbors = u_rooted_CPDAG.undirected_degree_of( list(int_V)[0] )
        V_neighbors_list = list(u_rooted_CPDAG._undirected_neighbors[ list(int_V)[0] ])
        n_classes = 2**(n_V_neighbors)
        CPDAG_MECsize = 0
        P_likelihood = 0
        for i in range(n_classes):
            CPDAG_copy = copy.deepcopy(u_rooted_CPDAG)
            bin_class = get_bin_class(n_V_neighbors, i)
            # Get interventional PDAG
            V_oriented_PDAG = get_int_PDAG(CPDAG_copy, int_V,
                                                     V_neighbors_list, bin_class)
            # Get CPDAG
            V_oriented_CPDAG = copy.deepcopy( V_oriented_PDAG )
            V_oriented_CPDAG.to_complete_pdag()
            # Check if the class is valid, skip invalid ones
            if has_v_structure_cycle(jl, V_oriented_CPDAG ):
                continue
            # Calculate the MEC size of this CPDAG
            V_oriented_UCCG_MECsize = get_CPDAG_MECsize(jl, V_oriented_CPDAG)
            CPDAG_MECsize += V_oriented_UCCG_MECsize
            # Sample a DAG and get likelihood
            sample_DAG = DAG_sample_from_CPDAG(jl, V_oriented_CPDAG)
            P_likelihood_class = P_likelihood_given_DAG(x, sample_DAG, BN,
                                                        joint_prob, int_V)
            P_likelihood += V_oriented_UCCG_MECsize * P_likelihood_class

        P_likelihood = P_likelihood / CPDAG_MECsize
        return P_likelihood

# Get the binary code of the orientation
def get_bin_class(n_V_neighbors, i):
    n_classes = 2 ** (n_V_neighbors)
    bin_class = bin(i)[2:]
    bin_class = '0' * (n_V_neighbors - len(bin_class)) + bin_class
    return bin_class

# Get the PDAG according to the binary code
def get_int_PDAG(UCCG, int_V: set, V_neighbors_list, bin_class):
    v = list(int_V)[0]
    for j in range(len(V_neighbors_list)):
        neighbor = V_neighbors_list[j]
        if bin_class[j] == '0':
            UCCG.replace_edge_with_arc( (neighbor, v) )
        else:
            UCCG.replace_edge_with_arc( (v, neighbor) )
    return UCCG

# PDAG from intervention on a set
def get_int_PDAG_from_combination(UCCG:graphical_models.PDAG, int_V_list, combination):
    PDAG = copy.deepcopy(UCCG)
    for v, config_v in zip(int_V_list, combination):
        n_v_neighbor = len(config_v)
        v_neighbors_list = list(UCCG.neighbors[v])
        for i in range(n_v_neighbor):
            neighbor = v_neighbors_list[i]
            if config_v[i] == '0':
                PDAG.replace_edge_with_arc((neighbor, v))
            else:
                PDAG.replace_edge_with_arc((v, neighbor))
    return PDAG

def get_combination_name(int_V_list, combination):
    name = ''
    for v, config in zip(int_V_list, combination):
        if name == '':
            name = str(v) + '|' + config
        else:
            name = name + ';' + str(v) + '|' + config
    return name

# Check if a CPDAG has v structures
def has_v_structure_cycle(jl, CPDAG: graphical_models.PDAG ):
    try:
        DAG = DAG_sample_from_CPDAG(jl, CPDAG)
        if DAG.vstructs() != set() or DAG is None:
            return True
        else:
            DAG = PDAG_to_nx(DAG)
            try:
                nx.find_cycle(DAG, orientation='original')
                return True
            except:
                return False

    except Exception as e:
        return True

def get_Prior_of_sep_set(jl, BN, joint_prob, UCCG, int_V: set):
    PDAG = graphical_models.PDAG.from_nx(UCCG)
    # Initialize the dictionary to save priors
    P_prior = {}
    P_likelihood = {}
    P_posterior = {}
    P_int = {}
    UCCG_MECsize = 0

    # Get all possible orientations of sep set
    class_configurations = {}
    int_V_list = list(int_V)
    for v in int_V_list:
        class_configurations.setdefault(v, [])
        n_v_neighbors = PDAG.undirected_degree_of(v)
        V_neighbors_list = list(PDAG._undirected_neighbors[v])
        n_classes = 2 ** n_v_neighbors
        for i in range(n_classes):
            bin_class = get_bin_class(n_v_neighbors, i)
            class_configurations[v].append(bin_class)

    all_class_combinations = list(itertools.product(*class_configurations.values()))

    # Iterate through combinations
    for combination in all_class_combinations:
        combination_PDAG = get_int_PDAG_from_combination(PDAG, int_V_list, combination)
        combination_PDAG.to_complete_pdag()
        # Check if the class is valid, skip invalid ones
        if has_v_structure_cycle(jl, combination_PDAG):
            continue
        combination_MECsize = get_CPDAG_MECsize(jl, combination_PDAG)
        combination_DAG_sample = DAG_sample_from_CPDAG(jl, combination_PDAG)
        UCCG_MECsize += combination_MECsize
        combination_name = get_combination_name(int_V_list, combination)
        P_prior[combination_name] = combination_MECsize
        P_int[combination_name] = get_int_joint_prob_DAG(BN, joint_prob, combination_DAG_sample, int_V)

    for key in P_prior.keys():
        P_prior[key] = P_prior[key]/UCCG_MECsize
        P_posterior[key] = P_prior[key]
        P_likelihood[key] = []
    return P_prior, P_likelihood, P_posterior, P_int

# Return prior of vertex intervention
def get_Prior_of_class(jl, UCCG, int_v: set):
    # Initialize the dictionary to save priors
    P_prior = {}
    P_likelihood = {}
    P_posterior = {}
    UCCG_MECsize = 0
    for u in list(UCCG.nodes):
        if u == list(int_v)[0]:
            u_rooted_CPDAG = UCCG_to_V_rooted_CPDAG(UCCG, set([u]))
            u_rooted_CPDAG_MECsize = get_CPDAG_MECsize(jl, u_rooted_CPDAG)
            UCCG_MECsize += u_rooted_CPDAG_MECsize
            P_prior[str(u)] = u_rooted_CPDAG_MECsize
        else:
            v = list(int_v)[0]
            u_rooted_CPDAG = UCCG_to_V_rooted_CPDAG(UCCG, set([u]))
            # Iterate through all orientations
            n_V_neighbors = u_rooted_CPDAG.undirected_degree_of(v)
            V_neighbors_list = list(u_rooted_CPDAG._undirected_neighbors[v])
            n_classes = 2 ** n_V_neighbors
            for i in range(n_classes):
                CPDAG_copy = copy.deepcopy(u_rooted_CPDAG)
                bin_class = get_bin_class(n_V_neighbors, i)
                # Get interventional PDAG
                V_oriented_PDAG = get_int_PDAG(CPDAG_copy, int_v,
                                               V_neighbors_list, bin_class)
                # Get CPDAG
                V_oriented_CPDAG = copy.deepcopy(V_oriented_PDAG)
                V_oriented_CPDAG.to_complete_pdag()
                # Check if the class is valid, skip invalid ones
                if has_v_structure_cycle(jl, V_oriented_CPDAG):
                    continue
                # Calculate the MEC size of this CPDAG
                V_oriented_UCCG_MECsize = get_CPDAG_MECsize(jl, V_oriented_CPDAG)
                UCCG_MECsize += V_oriented_UCCG_MECsize
                P_prior[str(u)+'|'+bin_class] = V_oriented_UCCG_MECsize

    for key in P_prior.keys():
        P_prior[key] = P_prior[key]/UCCG_MECsize
        P_posterior[key] = []
        P_likelihood[key] = []
    return P_prior, P_likelihood, P_posterior

def get_class_likelihood_from_u_rooted_CPDAG(jl, x, BN, joint_prob, u_rooted_CPDAG,
                                             root_u, int_V: set, P_likelihood):
    V_neighbors_list = list(u_rooted_CPDAG._undirected_neighbors[list(int_V)[0]])
    for key in P_likelihood.keys():
        if key.split('|')[0] != str(root_u):
            continue
        bin_class = key.split('|')[1]
        CPDAG_copy = copy.deepcopy(u_rooted_CPDAG)
        V_oriented_PDAG = get_int_PDAG(CPDAG_copy, int_V,
                                       V_neighbors_list, bin_class)
        V_oriented_CPDAG = copy.deepcopy(V_oriented_PDAG)
        V_oriented_CPDAG.to_complete_pdag()
        sample_DAG = DAG_sample_from_CPDAG(jl, V_oriented_CPDAG)
        P_likelihood_class = P_likelihood_given_DAG(x, sample_DAG, BN,
                                                        joint_prob, int_V)
        P_likelihood[key] = P_likelihood_class

    return P_likelihood


def get_sep_set_class_likelihood(jl, x, BN, joint_prob, UCCG:nx.Graph,
                                int_V:set, P_likelihood):
    UCCG = graphical_models.PDAG.from_nx(UCCG)
    n_V_int = len(int_V)
    for sep_set_class in P_likelihood.keys():
        sep_set_class_PDAG = copy.deepcopy(UCCG)
        for i in range(n_V_int):
            int_v = list(int_V)[i]
            bin_class = sep_set_class.split(';')[i].split('|')[1]
            v_neighbors_list = list( UCCG.neighbors_of(int_v) )
            sep_set_class_PDAG = get_int_PDAG(sep_set_class_PDAG, set([int_v]),
                                           v_neighbors_list, bin_class)

        # Orient to CPDAG and sample DAG to estimate likelihood
        sep_set_class_PDAG.to_complete_pdag()
        sample_DAG = DAG_sample_from_CPDAG(jl, sep_set_class_PDAG)
        P_likelihood_class = P_likelihood_given_DAG(x, sample_DAG, BN,
                                                    joint_prob, int_V)
        P_likelihood[sep_set_class] = P_likelihood_class

    return P_likelihood
