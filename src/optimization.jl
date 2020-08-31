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

cost_of_removing(x) = x * x

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


    (n, m) = size(Q)
    T = Q * diagm(α)
    total_cost = 0
    prev_len = 0
    to_remove = fill(0, 0)

    # while we can still potentially remove a node's influence
    while total_cost < budget
        current_cost = total_cost
        best_so_far = WEIGHT(Inf)
        current_to_remove = -1

        # consider each node
        for i = 1:n
            # get cost of adding node
            cost_to_add_i = (cost_of_removing(y[i]) + current_cost)

            # if we have not considered adding that node and
            # our budget would allow us to add them
            if i ∉ to_remove && (cost_to_add_i <= budget)
                # consider them
                push!(to_remove, i)
                # get change to the norm
                marginal_norm = norm(mult_but_exclude(T, y, to_remove))

                # if the norm obtained is lower than our current best in this
                # iteration
                if marginal_norm < best_so_far
                    # flag it as the best to remove
                    current_to_remove = i
                    best_so_far = marginal_norm
                end
                # pop off to remove temporarily
                pop!(to_remove)
            end
        end

        # if we found a viable node to remove
        if current_to_remove != -1
            # add it to remove list
            push!(to_remove, current_to_remove)
             @show current_to_remove
        end

        # if (current_cost == total_cost) || (length(to_remove) == n)
        #     break
        if (length(to_remove) == n)
            break
        else
            total_cost = current_cost + cost_of_removing(y[to_remove[end]])
        end
    end

    δ = WEIGHT.(zeros(n))
    δ[to_remove] = -y[to_remove]

    return δ, total_cost, (T * (y + δ))
end

function select_using_centrality(Q::Matrix{WEIGHT}, y::OPINIONS,
    α::PSYCHOLOGICAL_FACTOR, budget::WEIGHT,
    G::SimpleGraph{Int64}, centrality::String)

    centralities = CENTRALITIES[centrality](G)
    (n, m) = size(Q)
    T = Q * diagm(α)

    value = fill(WEIGHT(0.0), n)
    cost = fill(WEIGHT(0.0), n)
    z_inf = T * y
    original = norm(z_inf)
    for i = 1:n
        value[i] = centralities[i]
        cost[i] = cost_of_removing(y[i])
    end
    perm = sortperm(-value)
    δ = fill(WEIGHT(0.0), n)
    total_cost = 0
    for i ∈ perm
        if (cost[i] + total_cost) <= budget
            total_cost += cost[i]
            δ[i] = -y[i]
        else
            remaining = budget - total_cost
            fraction = sqrt(remaining)
            δ[i] = -(sign(y[i])) * fraction
            break
        end
    end
    z = T * (y + δ)
    return (δ, total_cost, z)


end

function select_using_pagerank(Q::Matrix{WEIGHT}, y::OPINIONS,
    α::PSYCHOLOGICAL_FACTOR, budget::WEIGHT, A)

    g = construct_graph(A)
    centralities = closeness_centrality(g)

    (n, m) = size(Q)
    T = Q * diagm(α)
    value = fill(WEIGHT(0.0), n)
    cost = fill(WEIGHT(0.0), n)
    z_inf = T * y
    original = norm(z_inf)
    for i = 1:n
        value[i] = centralities[i]
        cost[i] = cost_of_removing(y[i])
    end



    perm = sortperm(-value)
    δ = fill(WEIGHT(0.0), n)
    total_cost = 0
    for i ∈ perm
        if (cost[i] + total_cost) <= budget
            total_cost += cost[i]
            δ[i] = -y[i]
        else
            remaining = budget - total_cost
            fraction = sqrt(remaining)
            δ[i] = -(sign(y[i])) * fraction
            break
        end
    end
    z = T * (y + δ)
    return (δ, total_cost, z)


end


function fractional_knapsack(Q::Matrix{WEIGHT}, y::OPINIONS,
    α::PSYCHOLOGICAL_FACTOR, budget::WEIGHT)
    (n, m) = size(Q)
    T = Q * diagm(α)
    value = fill(WEIGHT(0.0), n)
    cost = fill(WEIGHT(0.0), n)
    z_inf = T * y
    original = norm(z_inf)
    for i = 1:n
        v = mult_but_exclude(T, y, [i])
        v = norm(z_inf) - norm(v)
        value[i] = 0
        if value[i] > 0
            value[i] = v
        end
        cost[i] = cost_of_removing(y[i])
    end

    perm = sortperm(-value)
    δ = fill(WEIGHT(0.0), n)
    total_cost = 0
    for i ∈ perm
        if (cost[i] + total_cost) <= budget
            total_cost += cost[i]
            δ[i] = -y[i]
        else
            remaining = budget - total_cost
            fraction = sqrt(remaining)
            δ[i] = -(sign(y[i])) * fraction
            break
        end
    end
    z = T * (y + δ)
    return (δ, total_cost, z)
end




function plot_multiple_series(series)
    p = plot()
    n, m = size(series)
    x = 1:m
    for i = 1:n
        plot!(p, x, series[i, :])
    end
end

function approx_equlibrium_vector(M::SparseMatrixCSC, y::OPINIONS,
     α::PSYCHOLOGICAL_FACTOR, k::Int64)
    #k = Int64(ceil(compute_num_iters(p.y, p.M, epsilon, p.α)))
    z = y
    for i = 1:k
        z = (α .* y) + (M * z)
    end
    return z
 end

 function greedy_approach_approx(M::SparseMatrixCSC{WEIGHT}, y::OPINIONS,
     α::PSYCHOLOGICAL_FACTOR, budget::WEIGHT, epsilon::WEIGHT)

     k = Int64(ceil(compute_num_iters(y, M, epsilon, α)))

     (n, m) = size(M)
     #T = Q * diagm(α)
     total_cost = 0
     to_remove = fill(0, 0)

     # while we can still potentially remove a node's influence
     while (total_cost < budget) && (length(to_remove) < n)
         current_cost = total_cost
         best_so_far = WEIGHT(Inf)
         current_to_remove = -1

         # consider each node
         for i = 1:n
             # get cost of adding node
             cost_to_add_i = (cost_of_removing(y[i]) + current_cost)

             # if we have not considered adding that node and
             # our budget would allow us to add them
             if i ∉ to_remove && (cost_to_add_i <= budget)
                 # consider them
                 push!(to_remove, i)
                 # get change to the norm
                 temp_y = copy(y)
                 temp_y[to_remove] .= 0
                 temp_eqilibrium = approx_equlibrium_vector(M, temp_y, α, k)
                 marginal_norm = norm(temp_eqilibrium)

                 # if the norm obtained is lower than our current best in this
                 # iteration
                 if marginal_norm < best_so_far
                     # flag it as the best to remove
                     current_to_remove = i
                     best_so_far = marginal_norm
                 end
                 # pop off to remove temporarily
                 pop!(to_remove)
             end
         end

         # if we found a viable node to remove
         if current_to_remove != -1
             # add it to remove list
             push!(to_remove, current_to_remove)
             @show current_to_remove
         end

         # if (current_cost == total_cost) || (length(to_remove) == n)
         #     break
         total_cost = current_cost + cost_of_removing(y[to_remove[end]])
     end

     δ = WEIGHT.(zeros(n))
     δ[to_remove] = -y[to_remove]
     # @show M
     # @show y + δ
     z = approx_equlibrium_vector(M, y + δ, α, k)
     return δ, total_cost, z
 end


 function fractional_knapsack_approx(M::SparseMatrixCSC{WEIGHT}, y::OPINIONS,
     α::PSYCHOLOGICAL_FACTOR, budget::WEIGHT, epsilon::WEIGHT)

     k = Int64(ceil(compute_num_iters(y, M, epsilon, α)))

     (n, m) = size(M)
     value = fill(WEIGHT(0.0), n)
     cost = fill(WEIGHT(0.0), n)
     z_inf = approx_equlibrium_vector(M, y, α, k)
     original = norm(z_inf)
     for i = 1:n
         temp_y = copy(y)
         temp_y[i] = 0
         v = approx_equlibrium_vector(M, temp_y, α, k)
         v = norm(z_inf) - norm(v)
         value[i] = 0
         if value[i] > 0
             value[i] = v
         end
         cost[i] = cost_of_removing(y[i])
     end

     perm = sortperm(-value)
     δ = fill(WEIGHT(0.0), n)
     total_cost = 0
     for i ∈ perm
         if (cost[i] + total_cost) <= budget
             total_cost += cost[i]
             δ[i] = -y[i]
         else
             remaining = budget - total_cost
             fraction = sqrt(remaining)
             δ[i] = -(sign(y[i])) * fraction
             break
         end
     end
     z = approx_equlibrium_vector(M, y + δ, α, k)
     return (δ, total_cost, z)
 end



# function main()
#
#     DATASET = "congress"
#     BUDGET = WEIGHT(3.0)
#     filepath = "./data/raw/$(DATASET).edgelist"
#
#
#     @show filepath
#     (W, U, n) = read_edgelist(filepath)
#     normalize_edgelist!(W)
#     normalize_edgelist!(U)
#
#     y = generate_opinions(n)
#     α, β, γ = generate_factors(n)
#     M = (β .* W) - (γ .* U)
#     G = construct_graph(M)
#
#     M = Matrix(M)
#     Q = inv(I - M)
#
#     z = Q * (α .* y)
#
#
#
#     (δ, total_cost, z_prime) =  greedy_approach(Q, y, α,  BUDGET)
#
#
#
#     δ1 = convex_approach(Q, y, α,  BUDGET)
#     res = Q * (α .* (y + δ1))
#
#     (δ, total_cost, z_knap) = fractional_knapsack(Q, y, α,  BUDGET)
#
#     (δ, total_cost, z_page_rank) = select_using_pagerank(Q, y, α,  BUDGET, M)
#
#     @show norm(y)
#     @show norm(z)
#     @show norm(z_prime)
#     @show norm(z_knap)
#     @show norm(z_page_rank)
#     @show norm(res)
#
#     times = n * 2
#
#     trace = fill(WEIGHT(0.0), n, times)
#     trace[:,1] = y
#     for i = 2:times
#         next_opinion = get_next_opinion(α, β, γ,W, U, y, trace[:,i - 1])
#         trace[:,i] = next_opinion
#     end
#
#
#     # plot_multiple_series(trace)
#     # plot(1:times)
#     return (y, Q, α, δ, total_cost, z_prime, G, trace)
#     # return y, z, res
# end
#
# #y, z, res = main()
# (y, Q, α, δ, total_cost, z_prime, G, trace) = main()
