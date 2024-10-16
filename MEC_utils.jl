using LightGraphs
using GraphIO

#=
ex:
- Julia version: 
- Author: zhouz
- Date: 2024-02-20
=#

include("./CliquePicking/sampling.jl")
include("./CliquePicking/counting.jl")

function process_graph(graph)
    order_of_graph = size(vertices(graph))[1]
    return order_of_graph
end

function call_MECCounting(graph_dir)
    g = readgraph(graph_dir, true)
    return MECsize(g)
end

function call_MECSampling(graph_dir, sample_dir = "./samples/sample.lgz")
    chordal_g = readgraph(graph_dir, true)
    sample_G = sampleDAG(chordal_g)
    savegraph(sample_dir, sample_G)
    return sample_G
end