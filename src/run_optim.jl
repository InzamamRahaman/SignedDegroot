include("util.jl")
include("graph.jl")
include("opinion.jl")
include("optimization.jl")
include("problem.jl")


function apply_convex_approach(p::ProblemInstance,
        Q::Matrix{WEIGHT}, budget::WEIGHT)
    return WEIGHT.(convex_approach(Q, p.y, p.α, budget))
end

function apply_knapsack_approach(p::ProblemInstance,
    Q::Matrix{WEIGHT}, budget::WEIGHT)
    (δ, total_cost, z) = (fractional_knapsack(Q, p.y, p.α, budget))
    println("Total cost of knapsack approach ", total_cost)
    return WEIGHT.(δ)
end

function apply_selection_by_centrality(p::ProblemInstance,
    Q::Matrix{WEIGHT}, budget::WEIGHT, centrality::String)

    (δ, total_cost, z) = select_using_centrality(Q, p.y,
        p.α, budget, p.G, centrality)
    println("Total cost of centrality approach with ", centrality, " centrality is ",  total_cost)
    return WEIGHT.(δ)

end

function apply_greedy_approach(p::ProblemInstance,
        Q::Matrix{WEIGHT}, budget::WEIGHT)
    (δ, total_cost, z) = greedy_approach(Q, p.y, p.α, budget)
    println("Total cost for complete greedy ", total_cost)
    return WEIGHT.(δ)
end


function get_polarization(Q::Matrix{WEIGHT},
    α::PSYCHOLOGICAL_FACTOR, y::OPINIONS)
    z = Q * (α .* y)
    return norm(z)
end

function construct_approx(p::ProblemInstance, epsilon::WEIGHT)
    k = Int64(ceil(compute_num_iters(p.y, p.M, epsilon, p.α)))
    function f()
        z = p.y
        for i = 1:k
            z = (p.α .* p.y) + (p.M * z)
        end
        return z
    end
    return f
end

function apply_fractional_knapsack_approx(p::ProblemInstance, budget::WEIGHT, epsilon::WEIGHT)
    (δ, total_cost, z) = fractional_knapsack_approx(p.M, p.y, p.α, budget,
                        epsilon)
    println("Total cost for complete greedy ", total_cost)
    return WEIGHT.(δ), norm(z)
end

function apply_greedy_approx(p::ProblemInstance, budget::WEIGHT, epsilon::WEIGHT)
    (δ, total_cost, z) = greedy_approach_approx(p.M, p.y, p.α, budget,
                        epsilon)
    println("Total cost for complete greedy ", total_cost)
    return WEIGHT.(δ), norm(z)
end




function optimize_across_methods(p::ProblemInstance,
        test_complete=true, budget=WEIGHT(10.0), epsilon=WEIGHT(0.01))

    n = length(p.y)
    #return problem_instance
    timing_results = Dict{String, Float64}()
    polarization_results = Dict{String, WEIGHT}()

    #polarization_results["before"] = norm(p.y)/length(p.y)

    if test_complete
        Q = Matrix(I - p.M)
        Q = inv(Q)

        t = @elapsed z = get_polarization(Q, p.α, p.y)
        timing_results["original"] = t
        polarization_results["original"] = z/n


        t = @elapsed δ = apply_convex_approach(p, Q, budget)
        timing_results["optimal"] = t
        polarization_results["optimal"] = get_polarization(Q, p.α, p.y + δ)/n

        t = @elapsed δ = apply_knapsack_approach(p, Q, budget)
        timing_results["complete_knapsack"] = t
        polarization_results["complete_knapsack"] =
            get_polarization(Q, p.α, p.y + δ)/n

        t = @elapsed δ = apply_greedy_approach(p, Q, budget)
        timing_results["complete_greedy"] = t
        polarization_results["complete_greedy"] =
            get_polarization(Q, p.α, p.y + δ)/n

        for centrality in keys(CENTRALITIES)
            name = "complete_centrality_$(centrality)"

            t = @elapsed δ = apply_selection_by_centrality(p, Q, budget, centrality)
            timing_results[name] = t
            polarization_results[name] =
                get_polarization(Q, p.α, p.y + δ)/n
        end
    end

    t = @elapsed (δ, res) = apply_fractional_knapsack_approx(p, budget, epsilon)
    timing_results["fractional_knapsack_approx"] = t
    polarization_results["fractional_knapsack_approx"] = res/n

    t = @elapsed (δ, res) = apply_greedy_approx(p, budget, epsilon)
    timing_results["greedy_approx"] = t
    polarization_results["greedy_approx"] = res/n



    return (timing_results, polarization_results)
end

#(timing_results, polarization_results) = main("cloister")

function array_of_dicts_to_dict_of_arrays(d::Array{Dict{String, T}, 1}) where
        {T <: Number}
    ks = keys(d[1])
    ans = Dict{String, Array{T, 1}}()
    for k in ks
        ans[k] = Array{T, 1}()
    end

    for i = 1:length(d)
        for (k, v) in d[i]
            push!(ans[k], v)
        end
    end

    return ans

end

function mean_of_dict(d::Dict{String, Array{V, 1}}) where {V <: Number}

    ans = Dict{String, Array{Float64, 1}}()
    for (k, v) in d
        ans[k] = [Float64(mean(v)), Float64(std(v))]
    end

    return ans

end

function compute_improvement(d::Dict{String, Array{V, 1}}) where {V <: Number}

    original = d["original"]
    d1 = Dict{String, Array{V, 1}}()
    for (k, v) in d
        improvement = v
        if k != "original"
            temp = original - improvement
            temp = temp ./ original
            d1[k] = temp * 100
        end
    end

    return d1

end

function plot_improvements(scores::Dict{String, Array{WEIGHT,1}},
        budgets::Array{WEIGHT, 1}, path::String, dataset::String)
    p = plot(legend=:outertopright)
    plot!(size=(1000,900))
    plot!([1], [100], label = "", linealpha=1.0)
    for (k, v) in scores
        plot!(p, budgets, v, label=k, lw = 3, color=COLORS[k])
    end
    xlabel!("Budget")
    ylabel!("% reduction in polarization")
    name = dataset
    name = string(uppercase(name[1])) * name[2:end]
    title!("Improvement in Polarization by Reduction Method for $(name) Dataset")
    savefig(path)
end


function main(dataset, epsilon=WEIGHT(0.01))

    budgets = WEIGHT.(1:1:10)
    timings = Array{Dict{String, Float64}, 1}()
    scores = Array{Dict{String, WEIGHT}, 1}()
    p = prep_instance(dataset)

    for budget ∈ budgets
        println("Processing for ", budget)
        timing, score = optimize_across_methods(p, true, budget, epsilon)
        push!(timings, timing)
        push!(scores, score)
    end

    timings = array_of_dicts_to_dict_of_arrays(timings)
    timings = mean_of_dict(timings)
    scores = array_of_dicts_to_dict_of_arrays(scores)
    scores1 = compute_improvement(scores)
    mean_scores = mean_of_dict(scores)
    mean_score_improvements = mean_of_dict(scores1)

    plot_path = "./plots/optim_plots/$(dataset).png"
    plot_improvements(scores1, budgets, plot_path, dataset)

    stringdata = JSON.json(timings)

    open("./results/$(dataset)-timings.json", "w") do f
        write(f, stringdata)
    end

    stringdata = JSON.json(scores1)

    open("./results/$(dataset)-scores.json", "w") do f
        write(f, stringdata)
    end

    stringdata = JSON.json(mean_scores)

    open("./results/$(dataset)-mean-scores.json", "w") do f
        write(f, stringdata)
    end

    stringdata = JSON.json(mean_score_improvements)

    open("./results/$(dataset)-mean-scores-improvements.json", "w") do f
        write(f, stringdata)
    end








    return timings, scores1, mean_scores, mean_score_improvements

end

(timings, scores, mean_scores, mean_score_improvements ) = main("congress")
