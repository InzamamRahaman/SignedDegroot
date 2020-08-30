include("util.jl")
include("graph.jl")
include("opinion.jl")

struct ProblemInstance
    α::PSYCHOLOGICAL_FACTOR
    β::PSYCHOLOGICAL_FACTOR
    γ::PSYCHOLOGICAL_FACTOR
    y::OPINIONS
    W::Adj_Matrix
    U::Adj_Matrix
    M::Adj_Matrix
    G::SimpleGraph{Int64}
end


function prep_instance(dataset="congress")
    delim = DATASETS[dataset]
    FILEPATH =  "./data/raw/$(dataset).edgelist"
    println("Reading file at ", FILEPATH)
    (W, U, n) = read_edgelist(FILEPATH, delim)
    normalize_edgelist!(W)
    normalize_edgelist!(U)
    y = generate_opinions(n)
    α, β, γ = generate_factors(n)
    M = (β .* W) - (γ .* U)
    G = construct_graph(M)
    problem_instance  = ProblemInstance(α, β, γ, y, W, U, M, G)
    return problem_instance
end
