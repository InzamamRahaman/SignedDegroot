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
                "bitcoinotc" => COMMA, "bitcoinalpha" => COMMA,
                "twitterreferendum"=>WHITESPACE)

CENTRALITIES = Dict(
    "pagerank" => pagerank,
    "betweeness" => betweenness_centrality,
    "closeness" => closeness_centrality,
    "stress" => stress_centrality,
    "radiality" => radiality_centrality
)

COLORS = Dict(
    "complete_centrality_pagerank"   => "#A6CEE3",
    "complete_knapsack"              => "#B2DF8A",
    "optimal"                        => "#1F78B4",
    "fractional_knapsack_approx"     => "#33A02C",
    "complete_centrality_radiality"  => "#FB9A99",
    "complete_centrality_stress"     => "#FDBF6F",
    "greedy_approx"                  => "#FF7F00",
    "complete_centrality_closeness"  => "#CAB2D6",
    "complete_centrality_betweeness" => "#6A3D9A",
    "complete_greedy"                => "#E31A1C"
)
