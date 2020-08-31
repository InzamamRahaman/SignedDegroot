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
