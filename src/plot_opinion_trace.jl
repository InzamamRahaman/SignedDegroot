include("util.jl")
include("graph.jl")
include("opinion.jl")


function main(dataset="congress")

    delim = DATASETS[dataset]
    DATASET = dataset
    FILEPATH =  "./data/raw/$(DATASET).edgelist"
    PLOT_PATH = "./plots/traces/$(DATASET).png"
    (W, U, n) = read_edgelist(FILEPATH, delim)
    normalize_edgelist!(W)
    normalize_edgelist!(U)
    y = generate_opinions(n)
    α, β, γ = generate_factors(n)
    M = (β .* W) - (γ .* U)
    #M = Matrix(M)
    #Q = inv(I - M)



    times  = 20

    trace = fill(WEIGHT(0.0), n, times)
    trace[:,1] = y

    println("Computing traces")
    for i = 2:times
        z = trace[:,i - 1]
        next_opinion = (α .* y) + (M * z)
        trace[:,i] = next_opinion
    end

    println("Ploting traces")
    p = plot(legend = false)
    for i = 1:n
        x = 1:times
        to_plot = trace[i, :]
        plot!(p, x, to_plot, legend = false)
    end

    println("Saving plots of traces")
    savefig(PLOT_PATH)


end

main("wikielections")
