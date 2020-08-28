include("util.jl")
include("graph.jl")
include("opinion.jl")


# function convex_approach(Q::Array{WEIGHT, 2}, y::OPINIONS, budget::WEIGHT)
#     z = Q * y
#     N = size(y)
#     N = N[1]
#     δ = Convex.Variable(N)
#     problem = minimize(sumsquares(z + (Q * δ)))
#     problem.constraints += (z + δ) <= 1
#     problem.constraints += (z + δ) >= -1
#     problem.constraints += sum((dot(^)(δ,2))) <= budget
#     solve!(problem, SCS.Optimizer)
#     @show problem.status
#     @show (z + (Q * δ.value))
#     return δ.value
# end

function convex_approach(Q::Matrix{WEIGHT}, y::OPINIONS,
                        α::PSYCHOLOGICAL_FACTOR, budget::WEIGHT)
    z = Q * (α .* y)
    N = size(y)
    N = N[1]
    δ = Variable(N)
    problem = minimize(sumsquares(z + (Q * (α .* δ))))
    problem.constraints += (z + δ) <= 1
    problem.constraints += (z + δ) >= -1
    problem.constraints += sum((dot(^)(δ,2))) <= budget
    solve!(problem, SCS.Optimizer)
    @show problem.status
    @show Q * (α .* (y + δ))
    return δ.value
end


function mult_but_exclude(M::Matrix{WEIGHT}, y::OPINIONS,
    exclude::Array{Int64, 1})

    temp_M = M[:, setdiff(1:end, exclude)]
    temp_y = y[setdiff(1:end, exclude)]

    return (temp_M * temp_y)

end


function greedy_approach(Q::Matrix{WEIGHT}, y::OPINIONS, α::PSYCHOLOGICAL_FACTOR, budget::WEIGHT)

    cost_of_removing(x) = x * x
    (n, m) = size(Q)
    T = Q * diagm(α)
    total_cost = 0
    to_remove = fill(0, 0)
    while total_cost < budget
        current_cost = total_cost
        best_so_far = WEIGHT(Inf)
        current_to_remove = -1
        for i = 1:n
            if i ∉ to_remove && ((cost_of_removing(y[i]) + current_cost) <= budget)
                push!(to_remove, i)
                marginal_norm = norm(mult_but_exclude(T, y, to_remove))
                if marginal_norm < best_so_far
                    current_to_remove = i
                    best_so_far = marginal_norm
                else
                    pop!(to_remove)
                end
            end
        end
        if (current_cost == total_cost) || (length(to_remove) == n)
            break
        else
            total_cost = current_cost + cost_of_removing(y[to_remove[end]])
        end
    end

    δ = WEIGHT.(zeros(n))
    δ[to_remove] = -y[to_remove]

    return δ, total_cost, (T * (y + δ))



end


function calculate_ideal_deltas(Q::Matrix{WEIGHT}, y::OPINIONS, α::PSYCHOLOGICAL_FACTOR, budget::WEIGHT)

    (n, m) = size(Q)
    T = Q * diagm(α)
    new_vals = fill(WEIGHT(0.0), n)
    counts = fill(WEIGHT(0.0), n)
    for i = 1:n
        if y[i] != 0.0
            @simd for j = 1:m
                if (T[i,j] > 0) && (i != j)
                    counts[j] += 1
                    numer = -((T[i, :]' * y) - (T[i, j] * y[j]))
                    denom = T[i,j]
                    new_vals[j] += (numer / denom)
                    @show j
                    @show new_vals[j]
                end
            end
        end
    end

    @simd for j = 1:n
        if counts[j] > 0.0
            new_vals[j] = new_vals[j] / counts[j]
        else
            new_vals[j] = y[j]
        end
    end

    δ = new_vals - y
    cost = sum(δ .* δ)


end


function main()

    DATASET = "congress"
    BUDGET = WEIGHT(3.0)
    filepath = "./data/raw/$(DATASET).edgelist"


    @show filepath
    (W, U, n) = read_edgelist(filepath)
    normalize_edgelist!(W)
    normalize_edgelist!(U)

    y = generate_opinions(n)
    α, β, γ = generate_factors(n)
    M = (β .* W) - (γ .* U)

    M = Matrix(M)
    Q = inv(I - M)

    z = Q * (α .* y)



    (δ, total_cost, z_prime) =  greedy_approach(Q, y, α,  BUDGET)



    δ1 = convex_approach(Q, y, α,  BUDGET)
    res = Q * (α .* (y + δ1))

    @show norm(z)
    @show norm(z_prime)
    @show norm(res)



    return (δ, total_cost, z_prime)
    # return y, z, res
end

#y, z, res = main()
(δ, total_cost, z_prime) = main()
