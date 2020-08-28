using SparseArrays
using Distributions
using LinearAlgebra
using Plots
using GLM
using DataFrames
using Statistics
using Convex
using SCS

WEIGHT = Float32
Adj_Matrix = SparseMatrixCSC{WEIGHT, Int64}
OPINIONS = Union{Array{WEIGHT, 1}, Array{WEIGHT, 2}}
PSYCHOLOGICAL_FACTOR = Array{WEIGHT, 1}