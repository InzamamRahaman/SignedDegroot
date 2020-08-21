include("graph.jl")
include("opinion.jl")


function main()
    (W, U, n) = read_edgelist("../data/raw/congress.edgelist")
    normalize_edgelist!(pos)
    normalize_edgelist!(neg)
    y = generate_opinions(n)
    α, β, γ = generate_factors(n)
    Q = get_fundamental_matrix(β, W, γ, U)
    z∞ = get_equlibrium(α, β, γ, pos, neg, y)
end

main()
