import networkx as nx
import subprocess
import pyAgrum as gum
import graphical_models
import copy
import numpy as np
import matplotlib.pyplot as plt
#import graphtheory
import juliacall

# Write a nx graph for Julia loading, the nodes will be 1 to n, returns a dictionary that maps Julia graph id to nx node id
def write_graph(G, graph_dir = 'graph.gr'):
    nv = nx.number_of_nodes(G)
    ne = nx.number_of_edges(G)
    node_list = list(G.nodes)
    jl_id_to_nx_id_map = {}
    nx_id_to_jl_id_map = {}
    jl_ids = np.arange(nv) + 1
    for i in jl_ids:
        jl_id_to_nx_id_map[i] = node_list[i-1]
        nx_id_to_jl_id_map[node_list[i-1]] = i
    with open(graph_dir, 'w') as file:
        # Write nodes
        file.write("{} {}\n\n".format(nv, ne))
        for edge in G.edges:
            a, b = edge
            a_jl = nx_id_to_jl_id_map[a]
            b_jl = nx_id_to_jl_id_map[b]
            file.write("{} {}\n".format(a_jl, b_jl))

    return jl_id_to_nx_id_map

# Read a DAG sampled from Julia
def read_DAG(jl_id_to_nx_id_map, graph_dir = './samples/sample.lgz'):
    dag = nx.DiGraph()
    with open(graph_dir, "r") as file:
        # Read graph properties and arcs from each line
        for line in file:
            s = line.split(',')
            if len(s) > 2:
                n_nodes = np.int32(s[0])
                n_edges = np.int32(s[1])
                dag.add_nodes_from(np.arange(n_nodes) + 1)
            elif len(s)==2:
                src_jl = np.int64(s[0])
                dst_jl = np.int64(s[1])
                dag.add_edge(jl_id_to_nx_id_map[src_jl], jl_id_to_nx_id_map[dst_jl])
    return dag

# Calculate the MEC size of a CPDAG
def get_CPDAG_MECsize(jl, CPDAG: graphical_models.PDAG):
    MEC_size = 1
    UCCG_list = CPDAG.chain_components()
    if UCCG_list == []:
        MEC_size = 1
    else:
        for UCCG in UCCG_list:
            graph_dir = 'graph.gr'
            write_graph( UCCG.to_nx(), graph_dir )
            UCCG_MEC_size = call_JLMECsize(jl, graph_dir)
            MEC_size = MEC_size * UCCG_MEC_size

    return MEC_size

# Call the Julia MEC counting function
def call_JLMECsize(jl, graph_dir = 'graph.gr'):
    jl.seval(f'graph_dir = "{graph_dir}"')
    G_MECsize = jl.seval("call_MECCounting(graph_dir)")
    return G_MECsize

# Call the Julia MEC sampling function
def call_JLMECsampler(jl, jl_id_to_nx_id_map, graph_dir = 'graph.gr',
                      sample_dir = './samples/sample.lgz'):
    jl.seval(f'graph_dir = "{graph_dir}"')
    jl.seval(f'sample_dir = "{sample_dir}"')
    jl.seval("call_MECSampling(graph_dir, sample_dir)")
    DAG_sample = read_DAG(jl_id_to_nx_id_map, sample_dir)
    return DAG_sample

# transform graphical model PDAG to nx Digraph
def PDAG_to_nx(PDAG:graphical_models.PDAG):
    D = nx.DiGraph()
    D.add_edges_from(PDAG.arcs)
    return D

# Create a BN that matches a given nx DAG
def nxDAG_to_BN(nx_DAG, name = ''):
    BN = gum.BayesNet(name)

    # Add nodes to the Bayesian Network
    for i in range( len(nx_DAG.nodes() ) ):
        BN.add(str(i), 2)
    # Add directed edges to represent the DAG structure
    for edge in nx_DAG.edges():
        src, dst = edge
        BN.addArc(str(src), str(dst))

    return BN

# Get a dictionary that map from BN names to ids
def BN_names_to_id_map(BN):
    v_id_map = {}
    for name in list(BN.names()):
        v_id_map[name] = list( BN.nodeset( [name] ) )[0]
    return v_id_map

# Get the reverse map
def BN_id_to_names_map(BN):
    id_name_map = {}
    for name in list(BN.names()):
        id_name_map[list(BN.nodeset( [name] ))[0]] = name
    return id_name_map

# Ge skeleton from BN, use as essential graph for now
def Essential_from_BN(BN):
    UCCG = nx.Graph()
    UCCG.add_nodes_from(BN.nodes())
    UCCG.add_edges_from(BN.arcs())
    return UCCG

# Compute the UCCGs from an essential graph
def UCCG_from_BN_ess(BN_ess):
    CPDAG = graphical_models.PDAG(nodes=BN_ess.nodes(), edges=BN_ess.edges(), arcs=BN_ess.arcs())
    UCCG_list = CPDAG.chain_components()
    nx_UCCG_list = []
    for UCCG in UCCG_list:
        UCCG._arcs = set()
        nx_UCCG_list.append(UCCG.to_nx())

    return nx_UCCG_list

# Calculate the skeleton of a given DAG
def skeleton_from_DAG(DAG):
    skeleton = graphical_models.PDAG.from_nx(DAG)
    for arc in skeleton.arcs:
        skeleton = skeleton._replace_arc_with_edge(arc)
    return skeleton

def nodeset_to_nameset(nodeset: set):
    nameset = set()
    for v in list(nodeset):
        nameset = nameset.union( set([str(v)]) )
    return nameset

# Calculate the joint probablity of a BN
def get_joint_prob(BN):
    joint_prob = 1
    for var_name in list(BN.names()):
        joint_prob = joint_prob * BN.cpt(var_name)
    return joint_prob

# Calculate the interventional joint prob
def get_int_joint_prob_DAG(BN, joint_prob, DAG, int_V):
    int_joint_prob = 1
    int_V_names = nodeset_to_nameset(int_V)
    # Get joint prob of DAG
    for v in list(DAG.nodes):
        v_name = str(v)
        # If v is in intervetional set, change the distribution of v given Pa_v
        if v_name in int_V_names:
            v_var = BN.variableFromName(v_name)
            Pa_v = DAG.parents[v]
            Pa_v_names = nodeset_to_nameset(Pa_v)
            P_v_given_Pa_v = get_conditional_prob(joint_prob, set([v_name]), Pa_v_names)
            P_v_given_Pa_v = P_v_given_Pa_v.putFirst(v_name)
            int_prob = np.zeros(P_v_given_Pa_v.shape)
            int_prob[..., :] = [0, 1]
            P_v_given_Pa_v[:] = int_prob
            int_joint_prob = int_joint_prob * P_v_given_Pa_v

        else:
            # Get Potential for P(v|Pa_v)
            Pa_v = DAG.parents[v]
            Pa_v_names = nodeset_to_nameset(Pa_v)
            P_v_given_Pa_v = get_conditional_prob(joint_prob, set([v_name]), Pa_v_names)
            int_joint_prob = int_joint_prob * P_v_given_Pa_v

    # Extract array by order
    for i in range( len(DAG.nodes) ):
        int_joint_prob = int_joint_prob.putFirst(str(i))
    int_joint_prob = int_joint_prob.toarray()

    assert np.abs(int_joint_prob.sum() - 1) <= 0.001
    return int_joint_prob

# Estimate sample likelihood from P_int
def get_sample_likelihood_from_P_int(x, P_int:dict, P_likelihood:dict):
    for comb_name in list(P_int.keys()):
        int_joint_prob = P_int[comb_name]
        P_x_given_DAG = copy.copy(int_joint_prob)
        # Get Likelihood of x in joint prob of DAG
        length = len(int_joint_prob.shape)
        for i in range(length):
            var_name = str(i)
            label_i = int(x[var_name].values[0])
            P_x_given_DAG = P_x_given_DAG[label_i]
        P_likelihood[comb_name] = P_x_given_DAG

    return P_likelihood

# Calculate the prior of a given UCCG
def get_Prior_of_UCCG(jl, UCCG: nx.Graph, partition = 'root'):
    # Save for clique picking later
    if partition != 'root':
        raise NotImplementedError
    # Initialize the dictionary to save priors
    P_prior = {}
    # Get the MEC size for UCCG
    graph_dir = 'graph.gr'
    write_graph(UCCG, graph_dir)
    UCCG_MECsize = call_JLMECsize(jl, graph_dir)
    P_prior = {}  # Initialize prior for v as root
    for v in list(UCCG.nodes):
        v_name = str(v)
        # Retrieve the v-rooted CPDAG and UCCGs
        v_rooted_CPDAG = UCCG_to_V_rooted_CPDAG(UCCG, set([v]))  # CPDAG in graphical_models PDAG
        v_rooted_UCCG_list = v_rooted_CPDAG.chain_components()  # Induced subgraph of CC
        # If no UCCG left, it is a tree
        if v_rooted_UCCG_list == []:
            P_prior[v_name] = 1 / UCCG_MECsize
            continue

        # Else, for each UCCG in v-rooted CPDAG, calculate MEC size
        v_rooted_UCCG_MEC_size_l = []
        for v_rooted_UCCG in v_rooted_UCCG_list:
            # Find MECsize of sub UCCG
            graph_dir = 'graph.gr'
            write_graph(v_rooted_UCCG.to_nx(), graph_dir)
            v_MEC_size = call_JLMECsize(jl, graph_dir)
            v_rooted_UCCG_MEC_size_l.append(v_MEC_size)
        P_prior[v_name] = np.prod(v_rooted_UCCG_MEC_size_l) / UCCG_MECsize

    return P_prior


# Calculate the conditional probability from a given joint probability
def get_conditional_prob(joint_prob, v_name:set, Condition_v_names:set):
    copy_joint = copy.copy(joint_prob)
    # No condition, return marg prob
    if len(Condition_v_names)==0:
        return copy_joint.margSumOut(list(set(joint_prob.names)-v_name))
    else:
        marg_nominator_names = set(joint_prob.names)-v_name-Condition_v_names
        marg_denominator_names = set(joint_prob.names) - Condition_v_names
        P_denominator = copy_joint.margSumOut(list(marg_denominator_names))
        # No need to marg out
        if marg_nominator_names == set():
            P_nominator = joint_prob
        # Marg out rest
        else:
            P_nominator = copy_joint.margSumOut(list(marg_nominator_names))
        return P_nominator/P_denominator

# Given the UCCG and a set of source nodes, calculate the CPDAG
def UCCG_to_V_rooted_CPDAG(UCCG, V_ids:set):
    # Create PDAG to orient
    V_rooted_UCCG = graphical_models.PDAG(nodes=UCCG.nodes, edges=UCCG.edges)
    V_ids_list = list(V_ids)
    # Orient outgoing arcs from the source nodes
    for v in V_ids_list:
        neighbor_list = list(V_rooted_UCCG.neighbors_of(v) - V_ids)
        for u in neighbor_list:
            V_rooted_UCCG.replace_edge_with_arc((v, u))

    V_rooted_UCCG.to_complete_pdag()
    return V_rooted_UCCG

# Get an interventional BN, V_names is the intervention set
def get_intervened_BN(BN, V_names:set):
    BN_inter_V = copy.copy(BN)
    for v_name in list(V_names):
        p_v = BN_inter_V.cpt(v_name)
        a = np.zeros(p_v.shape)
        a[..., :] = [0, 1]
        BN_inter_V.cpt(v_name)[:] = a
    return BN_inter_V

# Sample a DAG that has V as source nodes
def DAG_sample_from_CPDAG(jl, v_rooted_CPDAG):
    v_rooted_UCCG_list = v_rooted_CPDAG.chain_components()
    # If fully oriented, return the CPDAG as DAG
    if v_rooted_UCCG_list == []:
        return v_rooted_CPDAG
    else:
        sample_DAG = copy.deepcopy(v_rooted_CPDAG)
        for v_rooted_UCCG in v_rooted_UCCG_list:
            # Sample a DAG for the UCCG
            graph_dir = 'graph.gr'
            jl_id_to_nx_id_map = write_graph(v_rooted_UCCG.to_nx(), graph_dir)
            sub_DAG = call_JLMECsampler(jl, jl_id_to_nx_id_map, graph_dir)
            sub_DAG_gm = graphical_models.PDAG(nodes=sub_DAG.nodes, arcs=sub_DAG.edges)
            # sub_DAG = graphical_models.PDAG.from_nx(sub_DAG)
            # Orient arcs in to match the sub DAG
            for arc in sub_DAG_gm.arcs:
                sample_DAG.replace_edge_with_arc(arc)
    return sample_DAG

# Get the name set from an id set
def get_V_names(id_name_map, V: set):
    if len(V)==1:
        return {id_name_map[list(V)[0]]}
    elif len(V)==0:
        return set()
    else:
        V_names = set()
        for v_id in list(V):
            V_names = V_names.union( [id_name_map[v_id]] )
        return V_names

# Calculate the likelihood when u==v
def get_P_X_given_V(x_sample, joint_prob_obs, int_V_names: set):
    # Get conditional prob of P(V\v|v)
    P_x_obs_given_int = copy.copy(joint_prob_obs)
    P_x_obs_given_int = get_conditional_prob( P_x_obs_given_int, set(joint_prob_obs.names) - int_V_names,
                                             int_V_names )
    # Get probability of x in joint prob of obs
    length = x_sample.shape[1]
    variable_order = P_x_obs_given_int.variablesSequence()
    for i in range(length):
        var_name = variable_order[length - i - 1].name()
        label_i = int(x_sample[var_name].values[0])
        P_x_obs_given_int = P_x_obs_given_int[label_i]
    return P_x_obs_given_int

# Calculate the likelihood from a sampled DAG
def P_likelihood_given_DAG(x, DAG, BN, joint_prob, int_V:set):
    joint_prob_DAG = 1
    int_V_names = nodeset_to_nameset(int_V)
    # Get joint prob of DAG
    for v in list(DAG.nodes):
        v_name = str(v)
        # If v is in intervetional set, change the distribution of v given Pa_v
        if v_name in int_V_names:
            v_var = BN.variableFromName(v_name)
            Pa_v = DAG.parents[v]
            Pa_v_names = nodeset_to_nameset( Pa_v )
            P_v_given_Pa_v = get_conditional_prob(joint_prob, set([v_name]), Pa_v_names)
            P_v_given_Pa_v = P_v_given_Pa_v.putFirst(v_name)
            int_prob = np.zeros(P_v_given_Pa_v.shape)
            int_prob[..., :] = [0, 1]
            P_v_given_Pa_v[:] = int_prob
            joint_prob_DAG = joint_prob_DAG * P_v_given_Pa_v

        else:
            # Get Potential for P(v|Pa_v)
            Pa_v = DAG.parents[v]
            Pa_v_names =  nodeset_to_nameset( Pa_v )
            P_v_given_Pa_v = get_conditional_prob(joint_prob, set([v_name]), Pa_v_names)
            joint_prob_DAG = joint_prob_DAG * P_v_given_Pa_v

    # Check if P given DAG is joint
    assert np.abs(joint_prob_DAG.sum() - 1) <= 0.0001
    P_x_given_DAG = copy.copy(joint_prob_DAG)
    # Get Likelihood of x in joint prob of DAG
    length = len(joint_prob_DAG.names)
    for i in range(length):
        var_name = joint_prob_DAG.variablesSequence()[length - i - 1].name()
        label_i = int(x[var_name].values[0])
        P_x_given_DAG = P_x_given_DAG[label_i]
    return P_x_given_DAG

# Normalize the posterior and update the prior
def Normalize_posterior(posterior:dict, likelihood:dict, prior:dict, trace = True):
    sum_post = 0
    normed_posterior = copy.copy(posterior)

    # Sum up likelihood*prior
    #for V_names in list(likelihood.keys()):
    #    sum_post += likelihood[V_names]*prior[V_names]
    sum_post = np.sum( np.multiply( list(likelihood.values()), list(prior.values())) )
    # Normalize posterior and update prior
    for V_names in list(posterior.keys()):
        if trace:
            normed_posterior[V_names].append(prior[V_names]*likelihood[V_names]/sum_post)
            prior[V_names] = normed_posterior[V_names][-1]
        else:
            normed_posterior[V_names] = prior[V_names] * likelihood[V_names] / sum_post
            prior[V_names] = normed_posterior[V_names]
    return normed_posterior, prior

# Check edge strength
def check_edge(joint_prob, u_name, v_name):
    print('Joint obs ')
    print(joint_prob)
    names = set(joint_prob.names)
    print('Joint {} {} '.format(u_name, v_name))
    print(joint_prob.margSumOut(list( names - set(u_name) - set(v_name) ) ))
    print('P {} given {}'.format(u_name, v_name))
    print(get_conditional_prob(joint_prob, set(u_name), set(v_name) ) )
    print('P {} given {}'.format(v_name, u_name))
    print(get_conditional_prob(joint_prob, set(v_name), set(u_name) ) )
    return None


# Plot the posterior vs samples
def plot_posterior_vs_samples(posterior, int_name, n_data_samples = 100):
    num_plot = len(posterior.keys())
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_plot)]
    x_axis = np.arange(n_data_samples)+1
    var_list = list(posterior.keys())
    plt.figure()
    for i in range(num_plot):
        V_names = var_list[i]
        plt.plot(x_axis, posterior[V_names], c=colors[i])
    plt.title('Posterior V rooted do {}'.format(int_name))
    plt.xlabel('# Samples')
    plt.ylabel('Posterior')
    plt.legend(var_list)
    plt.show()

    return None

# Plot root posterior with different number of DAG samples
def verify_root_consistency(posteriors, root_name, n_data_samples = 1000):
    num_plot = len(posteriors.keys())
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_plot)]
    x_axis = np.arange(n_data_samples)+1
    n_DAG_sample_list = list(posteriors.keys())
    plt.figure()
    for i in range(num_plot):
        n_DAG_sample = n_DAG_sample_list[i]
        plt.plot(x_axis, posteriors[n_DAG_sample], color=colors[i])
    plt.title('Posterior {} rooted do {}'.format(root_name, root_name))
    plt.xlabel('# Samples')
    plt.ylabel('Posteriors')
    plt.legend(n_DAG_sample_list)
    plt.show()
    return None

# Smooth plots
def smooth_array(a:np.ndarray):
    n = a.shape[0]
    i = 0
    while i<n:
        a[i: i + 100] = np.mean(a[i:i + 100])
        i = i + 100

    return a