using SparseArrays
using Distributions
using LinearAlgebra
using Plots
using GLM
using DataFrames
using Statistics
using Convex
using SCS
using LightGraphs
using TimerOutputs

WEIGHT = Float32
Adj_Matrix = SparseMatrixCSC{WEIGHT, Int64}
OPINIONS = Union{Array{WEIGHT, 1}, Array{WEIGHT, 2}}
PSYCHOLOGICAL_FACTOR = Array{WEIGHT, 1}

WHITESPACE = r"\s+"
COMMA = ","

DATASETS = Dict("congress"=>WHITESPACE, "wikielections" => WHITESPACE,
                "cloister" => WHITESPACE, "highlandtribes" => WHITESPACE,
                "bitcoinotc" => COMMA, "bitcoinalpha" => COMMA)

CENTRALITIES = Dict(
    "pagerank" => pagerank,
    "betweeness" => betweenness_centrality,
    "closeness" => closeness_centrality,
    "stress" => stress_centrality,
    "radiality" => radiality_centrality
)
