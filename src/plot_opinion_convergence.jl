include("graph.jl")
include("opinion.jl")



function row_sum_max(M::Matrix{WEIGHT})
    (n, m) = size(M)
    acc = fill(0.0, n)
    for i = 1:n
        for j = 1:n
            acc[i] += abs(M[i, j])
        end
    end
    return maximum(acc)
end

function main()
    dataset = "cloister"
    filepath = "./data/raw/$(dataset).edgelist"
    @show filepath
    (W, U, n) = read_edgelist(filepath)
    m = 2
    normalize_edgelist!(W)
    normalize_edgelist!(U)
    y = generate_opinions(n, m)
    α, β, γ = generate_factors(n)
    M = (β .* W) - (γ .* U)
    Q = get_fundamental_matrix(β, W, γ, U)
    z = get_equlibrium(α, Q, y)
    z_next = get_next_opinion(α,
                              β,
                              γ,
                              W, U, y,z)
    M = Matrix(β .* W - γ .* U)

    y_trans = y'
    z_trans = z'

    p = scatter(y_trans[1,:], y_trans[2,:], fmt = :png)
    savefig("before_equlilbrium_$(dataset).png")
    p = scatter(z_trans[1,:], z_trans[2,:], fmt = :png)
    savefig("after_equlilbrium_$(dataset).png")





    return y, Q, z
end

y, Q, z = main()

df = DataFrame(X = Float64.(y[:,1]), Y = Float64.(y[:, 2]))

df1 = DataFrame(X = Float64.(z[:,1]), Y = Float64.(z[:, 2]))
