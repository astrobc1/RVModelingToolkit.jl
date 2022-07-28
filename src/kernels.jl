export NoiseKernel, compute_stationary_dist_matrix

"""
Base type for noise kernels.
"""
abstract type NoiseKernel end

"""
    compute_stationary_dist_matrix(t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real})
Computes the stationary distance matrix D[i, j] = |t[i] - t[j]|
"""
function compute_stationary_dist_matrix(t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real})
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