import networkx as nx
import random
import numpy as np
import itertools as itr
from baselines.dct import *
from baselines.rnd import *
from baselines.radpt import *
import pickle

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

np.random.seed(42)

N_nodes = 5
density = 0.5
N_dag = 50
r_apa_l = []
dct_l = []
rnd_l = []

for i in range(N_dag):
    print("{} out of {}".format(i, N_dag))
    d = shanmugam_random_chordal(N_nodes, density)

    """
    j = 0
    while j == 0:
        try:
            a, s = DCT_discovery(d, N_nodes, density)
        except:
            continue
        else:
            dct_l.append(a[0])
            j = 1
    """

    a, s = Random_Interventions(d, N_nodes, density)
    rnd_l.append(a[0])

    a, s = r_apative(d , N_nodes, density)
    r_apa_l.append(a[0])

with open('baseline_{}_{}_data.pickle'.format(N_nodes, density), 'wb') as f:
    # Serialize and save the variables
    pickle.dump(dct_l, f)
    pickle.dump(rnd_l, f)
    pickle.dump(r_apa_l, f)

