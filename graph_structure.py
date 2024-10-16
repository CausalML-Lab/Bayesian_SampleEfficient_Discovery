import graphtheory
from graphtheory.structures.edges import Edge
from graphtheory.structures.graphs import Graph
from graphtheory.chordality.peotools import find_peo_lex_bfs
from graphtheory.chordality.peotools import find_peo_mcs
import networkx as nx
import numpy as np


def atomic_graph_sep(G:nx.Graph):
    sep_sys = {}
    for v in list(G.nodes):
        sep_sys[v + 1] = {v}
    return sep_sys

# Calculate the separating system of a given chordal graph
def chordal_graph_separating_sys(G:nx.Graph):
    #assert nx.is_chordal(G) == True
    G_gt = nx_to_gt(G)
    PEO = find_peo_mcs(G_gt)
    sep_sys = greedy_coloring(G, PEO)
    return sep_sys

def greedy_coloring(G:nx.Graph, PEO):
    # Reverse PEO
    PEO = PEO[::-1]
    coloring = {}
    # Find maximum clique
    max_clique = nx.find_cliques(G)
    omega = len( max(max_clique, key=len) )
    colors = set(np.arange(omega) + 1)
    color_set = {}
    for c in list(colors):
        color_set.setdefault(c, set() )

    visited_V_list = []
    for v in PEO:
        neighbor_colors = get_neighbor_colors(G, v, coloring, visited_V_list)
        c = min( colors - neighbor_colors )
        coloring[v] = c
        color_set[c] = color_set[c].union(set([v]))
        visited_V_list.append(v)
    return color_set

def get_neighbor_colors(G:nx.Graph, v, coloring, visited_V_list):
    N_c = set()
    for u in G.neighbors(v):
        if u in visited_V_list:
            N_c = N_c.union( set( [ coloring[u] ]) )
    return N_c

def nx_to_gt(G:nx.Graph):
    G_gt = Graph(directed=False)
    for u, v in list(G.edges):
        edge = Edge(u, v)
        G_gt.add_edge(edge)
    return G_gt