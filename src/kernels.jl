export NoiseKernel, compute_stationary_dist_matrix

abstract type NoiseKernel end

function compute_stationary_dist_matrix(t1, t2)
    n1 = length(t1)
    n2 = length(t2)
    out = zeros(n1, n2)
    for i ∈ eachindex(t1)
        for j ∈ eachindex(t2)
            out[i, j] = abs(t1[i] - t2[j])
        end
    end
    return out
end