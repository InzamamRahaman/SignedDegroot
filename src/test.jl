include("graph.jl")
include("opinion.jl")


function main()
    (W, U, n) = read_edgelist("../data/raw/cloister.edgelist")
    m = 2
    normalize_edgelist!(W)
    normalize_edgelist!(U)
    y = generate_opinions(n, m)
    α, β, γ = generate_factors(n)
    Q = get_fundamental_matrix(β, W, γ, U)
    z∞ = get_equlibrium(α, Q, y)
    return y, Q, z∞
end

main()
