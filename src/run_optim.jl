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


function get_polarization(Q::Matrix{WEIGHT},
    α::PSYCHOLOGICAL_FACTOR, y::OPINIONS)
    z = Q * (α .* y)
    return norm(z)
end


function main(dataset="cloister", test_complete=true, budget=WEIGHT(10.0))
    p = prep_instance(dataset)
    #return problem_instance
    timing_results = Dict{String, Float64}()
    polarization_results = Dict{String, WEIGHT}()

    if test_complete
        Q = Matrix(I - p.M)
        Q = inv(Q)

        t = @elapsed z = get_polarization(Q, p.α, p.y)
        timing_results["original"] = t
        polarization_results["original"] = z


        t = @elapsed δ = apply_convex_approach(p, Q, budget)
        timing_results["optimal"] = t
        polarization_results["optimal"] = get_polarization(Q, p.α, p.y + δ)

        t = @elapsed δ = apply_knapsack_approach(p, Q, budget)
        timing_results["complete_knapsack"] = t
        polarization_results["complete_knapsack"] =
            get_polarization(Q, p.α, p.y + δ)

        for centrality in keys(CENTRALITIES)
            name = "complete_centrality_$(centrality)"

            t = @elapsed δ = apply_selection_by_centrality(p, Q, budget, centrality)
            timing_results[name] = t
            polarization_results[name] =
                get_polarization(Q, p.α, p.y + δ)
        end

    end
    return (timing_results, polarization_results)
end

(timing_results, polarization_results) = main("congress")
