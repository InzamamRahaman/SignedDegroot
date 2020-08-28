include("util.jl")

using Distributions

function clean_comment(s::String)
    replace(s, r"(#\s+)"=>"")
end

function parse_node_label(s::String)
    return parse(Int64, s) + 1
end

function normalize_edgelist!(mat::Adj_Matrix)
    rows, cols = size(mat)
    for i in 1:rows
        total = 0
        @simd for j in 1:cols
            if mat[i, j] != 0.0
                total += mat[i, j]
            end
        end

        @simd for j in 1:cols
            if mat[i, j] != 0.0
                mat[i, j] /= total
            end
        end
    end
end

function scale_mat!(f::PSYCHOLOGICAL_FACTOR, mat::Adj_Matrix)
    rows, cols = size(mat)
    for i = 1:rows
        @simd for j = 1:cols
            mat[i, j] *= f[i]
        end
    end
end

function scale_mat!(f::PSYCHOLOGICAL_FACTOR, mat::Array{WEIGHT, 2})
    rows, cols = size(mat)
    for i = 1:rows
        @simd for j = 1:cols
            mat[i, j] *= f[i]
        end
    end
end

function read_edgelist(s::String)::Tuple{Adj_Matrix, Adj_Matrix, Int64}
    I_pos = Array{Int64, 1}()
    J_pos = Array{Int64, 1}()
    K_pos = Array{WEIGHT, 1}()

    I_neg = Array{Int64, 1}()
    J_neg = Array{Int64, 1}()
    K_neg = Array{WEIGHT, 1}()

    n = 0
    open(s) do file
        for (index, line) in enumerate(readlines(file))
            if startswith(line, "#") && (index == 1)
                comment = clean_comment(line)
                n = parse(Int64, comment)
            elseif !startswith(line, "#")
                contents = map(x -> String(x), split(line))
                #(u, v, w) = (0, 0, WEIGHT(1.0))
                u = parse_node_label(contents[1])
                v = parse_node_label(contents[2])
                w = WEIGHT(1.0)
                if length(contents) > 2
                    w = parse(WEIGHT, contents[3])
                    if w > 0.0
                        push!(I_pos, u)
                        push!(J_pos, v)
                        push!(I_pos, v)
                        push!(J_pos, u)
                        push!(K_pos, w)
                        push!(K_pos, w)
                    elseif w < 0.0
                        push!(I_neg, u)
                        push!(J_neg, v)
                        push!(I_neg, v)
                        push!(J_neg, u)
                        push!(K_neg, w)
                        push!(K_neg, w)
                    end
                end
            end
        end
    end
    n_temp = max(maximum(I_pos), maximum(J_pos))
    n = max(n, n_temp)
    n_temp = max(maximum(I_neg), maximum(J_neg))
    n = max(n, n_temp)

    pos = sparse(I_pos, J_pos, K_pos, n, n)
    neg = sparse(I_neg, J_neg, K_neg, n, n)


    return (pos, neg, n)
end


function schatten_norm(A, p)
    inner = tr(A' * A)
    inner = inner ^ (p / 2)
    outer = inner ^ 1/p
end
