include("util.jl")

function test_if_balanced(M)
    (nrows, ncols) = size(M)
    for i = 1:nrows
        for j = 1:nrows
            for k = 1:nrows
                product = M[i, j] * M[j, k] * M[k, i]
                if product < -1
                    return false
                end
            end
        end
    end
    return true
end

function generate_factors(n::Int64)::Tuple{PSYCHOLOGICAL_FACTOR,
                                          PSYCHOLOGICAL_FACTOR,
                                          PSYCHOLOGICAL_FACTOR}

    α = WEIGHT.(rand(n))
    β = WEIGHT.(rand(n))
    γ = WEIGHT.(rand(n))

    totals = α + β + γ
    α ./= totals
    β ./= totals
    γ ./= totals

    return (α, β, γ)

end

function generate_opinions(n::Int64)
    base_dist = Uniform(-1.0, 1.0)
    arr = rand(base_dist, n)
    return WEIGHT.(arr)
end

function generate_opinions(n::Int64, m::Int64)
    base_dist = Uniform(-1.0, 1.0)
    arr = rand(base_dist, n, m)
    return WEIGHT.(arr)
end

function mynorm(M)
    denom  =
    return norm(M / denom)
end


# function generate_opinions(n::Int64)
#     selector = rand(0:1, n)
#     d1 = rand(truncated(Normal(-0.5, 0.1), -1, 0), n)
#     d2 = rand(truncated(Normal(0.5, 0.1), 0, 1), n)
#     arr = (selector .* d1) + ((1 .- selector) .* d2)
#     return WEIGHT.(arr)
# end
#
# function generate_opinions(n::Int64, m::Int64)
#     selector = rand(0:1, n)
#     d1 = rand(truncated(Normal(-0.5, 0.1), -1, 0), n, m)
#     d2 = rand(truncated(Normal(0.5, 0.1), 0, 1), n, m)
#     arr = (selector .* d1) + ((1 .- selector) .* d2)
#     return WEIGHT.(arr)
# end


function get_next_opinion(α::PSYCHOLOGICAL_FACTOR,
                          β::PSYCHOLOGICAL_FACTOR,
                          γ::PSYCHOLOGICAL_FACTOR,
                          W::Adj_Matrix, U::Adj_Matrix, y::OPINIONS,z::OPINIONS)
    inertia = α .* y
    change_mat = (β .* W) - (γ .* U)
    change_in_opinion = change_mat * z
    new_opinion = inertia + change_in_opinion
    return new_opinion
end

function get_next_opinion(α::PSYCHOLOGICAL_FACTOR,
                          β::PSYCHOLOGICAL_FACTOR,
                          γ::PSYCHOLOGICAL_FACTOR,
                          W::Adj_Matrix, U::Adj_Matrix, y::OPINIONS,
                          z::Nothing=nothing)
    return get_next_opinion(α, β, γ, W, U, y, y)
end

function get_sparse_identity(n::Int64)::SparseMatrixCSC{WEIGHT, Int64}
    return sparse(I, n, n)
end

function get_fundamental_matrix(β::PSYCHOLOGICAL_FACTOR, W::Adj_Matrix,
                                γ::PSYCHOLOGICAL_FACTOR, U::Adj_Matrix)
    (nrows, ncols) = size(W)
    identity_mat = get_sparse_identity(nrows)
    Q = -((β .* W)  + (γ  .* U))
    Q += identity_mat
    Q = Matrix(Q)
    Q = inv(Q)
    return Q
end

function get_equlibrium(α::PSYCHOLOGICAL_FACTOR, Q::Matrix, y::OPINIONS)
    z∞ = Q * (α .* y)
    return z∞
end

function get_equlibrium(α::PSYCHOLOGICAL_FACTOR,
                          β::PSYCHOLOGICAL_FACTOR,
                          γ::PSYCHOLOGICAL_FACTOR,
                          W::Adj_Matrix, U::Adj_Matrix, y::OPINIONS)
    Q = get_fundamental_matrix(β, W, γ, U)
    return get_equlibrium(α, Q, y)
end
