using SparseArrays
using Distributions
using LinearAlgebra
WEIGHT = Float16
Adj_Matrix = SparseMatrixCSC{WEIGHT, Int64}
OPINIONS = Array{WEIGHT, 1}
PSYCHOLOGICAL_FACTOR = Array{WEIGHT, 1}
